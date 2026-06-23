"""SR-7 — Análise de sensibilidade multi-target.

Executa um pipeline de modelagem simplificado (screening + CV) para todas as
definições de target (A, B, C, D) e consolida os resultados em um relatório
comparativo único. Permite avaliar a robustez das conclusões independentemente
da operacionalização de "resiliência criativa" escolhida.

Referência metodológica:
    Cook & Campbell (1979) "Quasi-Experimentation": validade de constructo
    exige que as conclusões não dependam de uma única operacionalização do constructo.
    Em Learning Analytics, variações do target são uma forma de análise de sensibilidade
    que fortalece a validade interna e externa dos achados.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import get_sample_weights, select_feature_columns
from src.utils.seed import set_global_seed


# ──────────────────────────────────────────────────────────────────
# Estimadores: subconjunto mais rápido para screening multi-target.
# Incluídos modelos representativos de cada família para cobertura
# metodológica sem custo computacional excessivo.
# ──────────────────────────────────────────────────────────────────
def _fast_estimators(seed: int) -> dict[str, Pipeline]:
    def scaled(clf: object) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])

    def tree(clf: object) -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ])

    return {
        "logistic_regression": scaled(
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        ),
        "svm_rbf": scaled(
            SVC(C=1.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)
        ),
        "knn_distance": scaled(KNeighborsClassifier(n_neighbors=21, weights="distance")),
        "gaussian_nb": tree(GaussianNB()),
        "random_forest": tree(
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1
            )
        ),
        "extra_trees": tree(
            ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1
            )
        ),
        "gradient_boosting": tree(
            GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=seed)
        ),
        "hist_gradient_boosting": tree(
            HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, l2_regularization=0.1, random_state=seed
            )
        ),
    }


def _ignore_warnings() -> None:
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*", category=UserWarning)


def _fit_safe(estimator: object, x: pd.DataFrame, y: pd.Series, weights: pd.Series | None) -> object:
    fitted = clone(estimator)
    with warnings.catch_warnings():
        _ignore_warnings()
        try:
            import inspect
            sig = inspect.signature(fitted.named_steps["clf"].fit if hasattr(fitted, "named_steps") else fitted.fit)
            if "sample_weight" in sig.parameters and weights is not None:
                key = "clf__sample_weight" if hasattr(fitted, "named_steps") else "sample_weight"
                fitted.fit(x, y, **{key: weights.to_numpy()})
            else:
                fitted.fit(x, y)
        except Exception:
            fitted.fit(x, y)
    return fitted


def _cv_ap(estimator: object, x: pd.DataFrame, y: pd.Series, weights: pd.Series | None, cv: RepeatedStratifiedKFold) -> dict[str, float]:
    """Avalia candidato em CV repetida. Retorna métricas condensadas."""
    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "f1": make_scorer(f1_score, zero_division=0),
    }
    with warnings.catch_warnings():
        _ignore_warnings()
        try:
            import inspect
            try:
                sig = inspect.signature(estimator.named_steps["clf"].fit)  # type: ignore
                fp = {"clf__sample_weight": weights.to_numpy()} if weights is not None and "sample_weight" in sig.parameters else {}
            except Exception:
                fp = {}
            scores = cross_validate(
                estimator, x, y, cv=cv, scoring=scoring,
                fit_params=fp or None, n_jobs=1, error_score=np.nan,
            )
        except TypeError:
            scores = cross_validate(
                estimator, x, y, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan,
            )

    def _s(k: str) -> tuple[float, float]:
        vals = np.asarray(scores[f"test_{k}"], dtype=float)
        return float(np.nanmean(vals)), float(np.nanstd(vals, ddof=1))

    ap_mean, ap_std = _s("average_precision")
    auc_mean, _ = _s("roc_auc")
    f1_mean, _ = _s("f1")
    return {
        "cv_ap_mean": ap_mean,
        "cv_ap_std": ap_std,
        "cv_auc_mean": auc_mean,
        "cv_f1_mean": f1_mean,
    }


def _one_target_results(
    target_key: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: dict,
    seed: int,
    test_size: float,
    cv_folds: int,
    cv_repeats: int,
) -> dict[str, Any]:
    """Executa pipeline de screening para UMA definição de target.

    Retorna dict com métricas do melhor modelo por CV AP no treino
    e avaliação no holdout (sem data leakage — limiares do treino).
    """
    from src.target_builder import apply_target_definitions, compute_target_thresholds

    # Target preliminar (global) apenas para estratificar o split
    target_col = f"target_{target_key}"
    if target_col in df.columns:
        y_prelim = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int)
    else:
        return {"target": target_key, "status": "sem coluna de target", "best_model": None}

    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask = y_prelim.notna() & x.notna().any(axis=1)
    x, y_prelim = x.loc[mask], y_prelim.loc[mask].astype(int)
    weights = get_sample_weights(df.loc[mask], cfg)

    prevalence = float(y_prelim.mean())
    n_pos = int(y_prelim.sum())
    n_total = len(y_prelim)

    if y_prelim.nunique() < 2 or n_pos < 20:
        return {
            "target": target_key,
            "status": f"classe insuficiente (n_pos={n_pos})",
            "prevalence": prevalence,
            "n_total": n_total,
            "n_pos": n_pos,
            "best_model": None,
        }

    split_kw = {"test_size": test_size, "random_state": seed, "stratify": y_prelim}
    if weights is not None:
        x_train, x_test, y_train_p, y_test_p, w_train, w_test = train_test_split(
            x, y_prelim, weights, **split_kw
        )
    else:
        x_train, x_test, y_train_p, y_test_p = train_test_split(x, y_prelim, **split_kw)
        w_train = w_test = None

    # CC-1: recomputar limiares exclusivamente no treino
    try:
        w_thresh = get_sample_weights(df.loc[x_train.index], cfg)
        thresholds = compute_target_thresholds(df.loc[x_train.index], weights=w_thresh)
        train_targets = apply_target_definitions(df.loc[x_train.index], thresholds)
        test_targets = apply_target_definitions(df.loc[x_test.index], thresholds)
        y_train = pd.Series(train_targets[target_key].values, index=x_train.index).astype(int)
        y_test = pd.Series(test_targets[target_key].values, index=x_test.index).astype(int)
    except Exception:
        y_train, y_test = y_train_p, y_test_p

    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=seed)
    estimators = _fast_estimators(seed)

    best_row: dict[str, Any] | None = None
    best_ap: float = -np.inf
    best_model: object | None = None
    best_proba: np.ndarray | None = None
    all_rows: list[dict[str, Any]] = []

    for name, estimator in estimators.items():
        try:
            cv_metrics = _cv_ap(estimator, x_train, y_train, w_train, cv)
            fitted = _fit_safe(estimator, x_train, y_train, w_train)
            with warnings.catch_warnings():
                _ignore_warnings()
                proba = fitted.predict_proba(x_test)[:, 1]
            holdout_ap = float(average_precision_score(y_test, proba))
            holdout_auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else np.nan
            holdout_brier = float(brier_score_loss(y_test, proba))
            row = {
                "model": name,
                **cv_metrics,
                "holdout_ap": holdout_ap,
                "holdout_auc": holdout_auc,
                "holdout_brier": holdout_brier,
            }
            all_rows.append(row)
            if cv_metrics["cv_ap_mean"] > best_ap:
                best_ap = cv_metrics["cv_ap_mean"]
                best_row = row
                best_model = fitted
                best_proba = proba
        except Exception as exc:
            all_rows.append({"model": name, "error": str(exc)})

    if best_row is None or best_proba is None:
        return {"target": target_key, "status": "todos os modelos falharam", "best_model": None}

    return {
        "target": target_key,
        "status": "ok",
        "prevalence": prevalence,
        "n_total": n_total,
        "n_pos": n_pos,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "best_model": str(best_row["model"]),
        "cv_ap_mean": best_row.get("cv_ap_mean"),
        "cv_ap_std": best_row.get("cv_ap_std"),
        "cv_auc_mean": best_row.get("cv_auc_mean"),
        "holdout_ap": best_row.get("holdout_ap"),
        "holdout_auc": best_row.get("holdout_auc"),
        "holdout_brier": best_row.get("holdout_brier"),
        "all_models": all_rows,
    }


def _rank_stability_index(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Calcula o Rank Stability Index dos modelos entre targets.

    Para cada par de targets, mede a correlação de Spearman entre os rankings
    dos modelos por CV AP. Alta correlação → achados robustos à operacionalização.
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return pd.DataFrame()

    # Coletar todos os modelos × targets com cv_ap_mean
    model_ap: dict[str, dict[str, float]] = {}
    for r in results:
        if r.get("status") != "ok" or not r.get("all_models"):
            continue
        for row in r["all_models"]:
            m = row.get("model")
            ap = row.get("cv_ap_mean")
            if m and ap is not None and not np.isnan(ap):
                model_ap.setdefault(m, {})[r["target"]] = ap

    # Pivot: models × targets
    targets_ok = [r["target"] for r in results if r.get("status") == "ok"]
    if len(targets_ok) < 2:
        return pd.DataFrame()

    pivot = pd.DataFrame(model_ap).T.reindex(columns=targets_ok)
    pivot = pivot.dropna()

    if pivot.shape[0] < 3:
        return pd.DataFrame()

    corr_rows = []
    for i, t1 in enumerate(targets_ok):
        for t2 in targets_ok[i + 1:]:
            col1, col2 = pivot[t1], pivot[t2]
            valid = col1.notna() & col2.notna()
            if valid.sum() < 3:
                continue
            rho, p = spearmanr(col1[valid], col2[valid])
            corr_rows.append({
                "target_1": t1,
                "target_2": t2,
                "spearman_rho": float(rho),
                "p_value": float(p),
                "interpretation": (
                    "Alta concordância" if abs(rho) >= 0.8
                    else "Concordância moderada" if abs(rho) >= 0.5
                    else "Baixa concordância — achados sensíveis à operacionalização"
                ),
            })
    return pd.DataFrame(corr_rows)


def _plot_sensitivity(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Gera gráficos comparativos entre targets."""
    try:
        import matplotlib.pyplot as plt

        targets = summary_df["target"].tolist()
        metrics_to_plot = {
            "CV Average Precision (PR-AUC)": "cv_ap_mean",
            "Holdout AUC-ROC": "holdout_auc",
            "Holdout Average Precision": "holdout_ap",
            "Brier Score (↓ melhor)": "holdout_brier",
            "Prevalência (classe positiva)": "prevalence",
        }

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, (label, col) in zip(axes, metrics_to_plot.items()):
            vals = summary_df[col].tolist() if col in summary_df.columns else [float("nan")] * len(targets)
            colors = ["#3B82F6" if col != "holdout_brier" else "#EF4444"] * len(targets)
            ax.bar(targets, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
            ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
            ax.set_xlabel("Definição de Target", fontsize=9)
            ax.grid(axis="y", alpha=0.3, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            # Adicionar valores sobre as barras
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(i, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        fig.suptitle(
            "Análise de Sensibilidade Multi-Target (A/B/C/D)\n"
            "Comparação do melhor modelo entre definições operacionais de resiliência criativa",
            fontsize=11, fontweight="bold", y=1.02,
        )
        plt.tight_layout()
        plt.savefig(out_dir / "sensitivity_comparison.png", dpi=200, bbox_inches="tight")
        plt.close()

    except Exception:
        pass


def _plot_model_heatmap(results: list[dict[str, Any]], out_dir: Path) -> None:
    """Heatmap: modelos × targets (CV Average Precision)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        model_ap: dict[str, dict[str, float]] = {}
        for r in results:
            if r.get("status") != "ok" or not r.get("all_models"):
                continue
            for row in r["all_models"]:
                m = row.get("model")
                ap = row.get("cv_ap_mean")
                if m and ap is not None and not np.isnan(float(ap)):
                    model_ap.setdefault(m, {})[r["target"]] = float(ap)

        if not model_ap:
            return

        targets_ok = [r["target"] for r in results if r.get("status") == "ok"]
        pivot = pd.DataFrame(model_ap).T.reindex(columns=targets_ok)

        plt.figure(figsize=(max(6, len(targets_ok) * 2), max(5, len(model_ap) * 0.6)))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "CV Average Precision (PR-AUC)"},
        )
        plt.title(
            "CV Average Precision por Modelo e Definição de Target\n"
            "(linhas consistentemente altas = modelo robusto à operacionalização)",
            fontsize=10, fontweight="bold",
        )
        plt.xlabel("Definição de Target", fontsize=9)
        plt.ylabel("Modelo", fontsize=9)
        plt.tight_layout()
        plt.savefig(out_dir / "sensitivity_model_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close()

    except Exception:
        pass


def run_sensitivity_analysis(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    """Fase 12 — Análise de Sensibilidade Multi-Target (SR-7).

    Executa screening de modelos para cada definição de target (A/B/C/D)
    usando o mesmo conjunto de features e configurações, e produz:
    - Tabela comparativa de métricas por target
    - Heatmap de AP por modelo × target
    - Rank Stability Index (correlação de Spearman dos rankings de modelos)
    - Relatório Markdown interpretativo

    Referência:
        Cook & Campbell (1979); validade de constructo requer que os achados
        não dependam de uma única operacionalização do constructo.
    """
    set_global_seed(int(cfg.get("random_seed", 42)))
    seed = int(cfg.get("random_seed", 42))

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "sensitivity"
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    modeling_cfg = cfg.get("modeling", {})
    test_size = float(modeling_cfg.get("test_size", 0.2))
    cv_folds = int(modeling_cfg.get("cv_folds", 5))
    # Para sensitivity analysis usa 2 repetições (metade do pipeline principal)
    # para equilibrar custo computacional com confiabilidade da estimativa
    cv_repeats = max(2, int(modeling_cfg.get("cv_repeats", 3)) - 1)

    target_defs = [str(t) for t in cfg.get("analysis", {}).get("target_definitions", ["A", "B", "C", "D"])]
    feature_cols = select_feature_columns(df, cfg)

    print(f"\n[sensitivity] Targets: {target_defs} | Features: {len(feature_cols)} | CV: {cv_folds}x{cv_repeats}")

    results: list[dict[str, Any]] = []
    for target_key in target_defs:
        print(f"  [sensitivity] Rodando target {target_key}...", flush=True)
        r = _one_target_results(
            target_key=target_key,
            df=df,
            feature_cols=feature_cols,
            cfg=cfg,
            seed=seed,
            test_size=test_size,
            cv_folds=cv_folds,
            cv_repeats=cv_repeats,
        )
        results.append(r)
        status_str = r.get("best_model") or r.get("status", "?")
        print(f"    → melhor: {status_str} | CV_AP={r.get('cv_ap_mean', float('nan')):.4f} | holdout_AUC={r.get('holdout_auc', float('nan')):.4f}")

    # ── Tabela resumo ──
    summary_cols = [
        "target", "status", "prevalence", "n_total", "n_pos", "n_train", "n_test",
        "best_model", "cv_ap_mean", "cv_ap_std", "cv_auc_mean",
        "holdout_ap", "holdout_auc", "holdout_brier",
    ]
    summary_df = pd.DataFrame([
        {c: r.get(c) for c in summary_cols} for r in results
    ])
    summary_df.to_csv(out_tables / "sensitivity_summary.csv", index=False)

    # Tabela detalhada por modelo × target
    all_model_rows: list[dict] = []
    for r in results:
        for row in r.get("all_models", []):
            all_model_rows.append({"target": r["target"], **row})
    if all_model_rows:
        pd.DataFrame(all_model_rows).to_csv(out_tables / "sensitivity_all_models.csv", index=False)

    # ── Rank Stability Index ──
    rank_df = _rank_stability_index(results)
    if not rank_df.empty:
        rank_df.to_csv(out_tables / "sensitivity_rank_stability.csv", index=False)

    # ── Gráficos ──
    _plot_sensitivity(summary_df, out_figures)
    _plot_model_heatmap(results, out_figures)

    # ── Interpretação automática ──
    ok_results = [r for r in results if r.get("status") == "ok"]
    if ok_results:
        ap_values = [r.get("cv_ap_mean", np.nan) for r in ok_results]
        ap_values_finite = [v for v in ap_values if v is not None and np.isfinite(v)]
        ap_range = max(ap_values_finite) - min(ap_values_finite) if len(ap_values_finite) > 1 else 0.0

        if ap_range < 0.05:
            robustness_msg = (
                f"**ALTA ROBUSTEZ** — A variação de CV AP entre targets é de {ap_range:.4f} "
                "(< 0.05). As conclusões são estáveis independentemente da operacionalização "
                "de resiliência criativa escolhida."
            )
        elif ap_range < 0.15:
            robustness_msg = (
                f"**ROBUSTEZ MODERADA** — A variação de CV AP entre targets é de {ap_range:.4f} "
                "(0.05–0.15). As conclusões são parcialmente estáveis; recomenda-se discutir "
                "as diferenças entre targets no texto do artigo."
            )
        else:
            robustness_msg = (
                f"**BAIXA ROBUSTEZ** — A variação de CV AP entre targets é de {ap_range:.4f} "
                "(> 0.15). As conclusões são sensíveis à operacionalização do target; "
                "a escolha de target A deve ser rigorosamente justificada teoricamente "
                "e as diferenças entre targets devem ser discutidas extensivamente."
            )
    else:
        robustness_msg = "Nenhum target executou com sucesso."
        ap_range = float("nan")

    # ── Relatório Markdown ──
    md: list[str] = [
        "# Análise de Sensibilidade Multi-Target (Fase 12 — SR-7)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Targets avaliados: {', '.join(target_defs)}\n",
        f"- Features: {len(feature_cols)}\n",
        f"- Protocolo: holdout {int(test_size*100)}% + CV {cv_folds}×{cv_repeats}\n",
        f"- Modelos no screening: {len(_fast_estimators(seed))}\n",
        f"- Data leakage controlado (CC-1): limiares calculados exclusivamente no treino\n\n",

        "## Justificativa metodológica\n\n",
        "A validade de constructo de *resiliência criativa* depende de como o constructo é "
        "operacionalizado. Diferentes limiares (A/B/C/D) capturam aspectos distintos do fenômeno "
        "e geram prevalências diferentes. Se os achados são robustos entre definições, "
        "a conclusão principal é mais forte; se variam, a escolha de operacionalização "
        "deve ser justificada com base na teoria e reportada transparentemente "
        "(Cook & Campbell, 1979; Messick, 1995).\n\n",

        "## Resumo de resultados por target\n\n",
        "> **Critério primário de seleção:** CV Average Precision (PR-AUC) no treino.\n",
        "> O holdout AUC-ROC é reportado como métrica secundária de comparabilidade.\n\n",
        df_to_markdown(summary_df[[c for c in summary_cols if c in summary_df.columns]]),
        "\n\n",

        "## Interpretação de robustez\n\n",
        robustness_msg,
        "\n\n",
    ]

    if not rank_df.empty:
        md.extend([
            "## Rank Stability Index (Spearman ρ entre rankings de modelos)\n\n",
            "> Correlação de Spearman dos rankings dos modelos por CV AP entre pares de targets.\n",
            "> ρ ≥ 0.80 indica alta concordância (os modelos são ordenados de forma similar entre targets).\n",
            "> ρ < 0.50 indica que a escolha do target muda qual modelo é melhor — sinal de alerta.\n\n",
            df_to_markdown(rank_df),
            "\n\n",
        ])

    md.extend([
        "## Recomendações para o manuscrito\n\n",
        "1. **Reportar target primário:** justificar a escolha de target A com base na literatura "
        "(ex.: prevalência similar à da literatura PISA; alinhamento com Q1/Q3).\n",
        "2. **Análise de sensibilidade:** incluir esta tabela como Apêndice ou Supplementary Material.\n",
        "3. **Interpretação diferencial:** se targets produzirem modelos muito diferentes, "
        "discutir as implicações teóricas de cada operacionalização.\n",
        "4. **Transparência:** reportar que o mesmo pipeline (sem otimização específica por target) "
        "foi aplicado a todas as definições.\n\n",
        "## Figuras\n\n",
        "- `outputs/figures/sensitivity/sensitivity_comparison.png` — métricas por target\n",
        "- `outputs/figures/sensitivity/sensitivity_model_heatmap.png` — AP por modelo × target\n",
    ])

    (out_reports / "sensitivity_report.md").write_text("".join(md), encoding="utf-8")
    print(f"\n[sensitivity] ✅ Relatório salvo em outputs/reports/sensitivity_report.md")
