from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import find_sensitive_columns, get_sample_weights, get_target_series
from src.utils.seed import set_global_seed


def _safe_tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """True Positive Rate (Recall) para o grupo."""
    pos_mask = y_true == 1
    if pos_mask.sum() == 0:
        return float("nan")
    return float(y_pred[pos_mask].mean())


def _safe_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """False Positive Rate para o grupo."""
    neg_mask = y_true == 0
    if neg_mask.sum() == 0:
        return float("nan")
    return float(y_pred[neg_mask].mean())


def _fairness_aggregate(fair_df: pd.DataFrame) -> pd.DataFrame:
    """Calcula métricas formais de equidade por variável sensível (SR-3).

    Métricas:
    - Equal Opportunity Difference (EOD): max(TPR) - min(TPR) por grupo
    - Demographic Parity Difference (DPD): max(prevalência) - min(prevalência)
    - Disparate Impact Ratio (DIR): min(prevalência) / max(prevalência)
    - Max AUC Gap: max(AUC) - min(AUC) entre grupos (se disponível)

    Referências:
    - Hardt et al. (2016) "Equality of opportunity in supervised learning"
    - Feldman et al. (2015) "Certifying and removing disparate impact"
    """
    rows = []
    for var, grp_df in fair_df.groupby("sensitive_var"):
        n_groups = len(grp_df)
        if n_groups < 2:
            continue

        # Demographic Parity
        prev = grp_df["prevalence"].dropna()
        dpd = float(prev.max() - prev.min()) if len(prev) >= 2 else float("nan")
        dir_ratio = float(prev.min() / prev.max()) if len(prev) >= 2 and prev.max() > 0 else float("nan")

        # Equal Opportunity (TPR gap)
        if "tpr" in grp_df.columns:
            tpr = grp_df["tpr"].dropna()
            eod = float(tpr.max() - tpr.min()) if len(tpr) >= 2 else float("nan")
        else:
            eod = float("nan")

        # AUC gap
        if "group_roc_auc" in grp_df.columns:
            aucs = grp_df["group_roc_auc"].dropna()
            auc_gap = float(aucs.max() - aucs.min()) if len(aucs) >= 2 else float("nan")
        else:
            auc_gap = float("nan")

        rows.append({
            "sensitive_var": var,
            "n_groups": n_groups,
            "demographic_parity_difference": dpd,
            "disparate_impact_ratio": dir_ratio,
            "equal_opportunity_difference": eod,
            "max_auc_gap": auc_gap,
            # Interpretação: DPD < 0.10 e DIR > 0.80 são limiares comuns na literatura
            "dpd_ok_lt010": (dpd < 0.10) if np.isfinite(dpd) else None,
            "dir_ok_gt080": (dir_ratio > 0.80) if np.isfinite(dir_ratio) else None,
        })
    return pd.DataFrame(rows)


def run_fairness(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    """Fase 10 — Análise de fairness algorítmica.

    SR-3: Métricas formais de equidade adicionadas:
    - Equal Opportunity Difference (TPR gap entre grupos)
    - Demographic Parity Difference
    - Disparate Impact Ratio
    Referências: Hardt et al. (2016), Feldman et al. (2015).
    """
    set_global_seed(int(cfg.get("random_seed", 42)))

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    y_raw, target_key = get_target_series(df, cfg)
    if isinstance(y_raw, pd.DataFrame):
        y = (y_raw.mean(axis=1) >= 0.5).astype(int)
    else:
        y = y_raw
    sensitive_cols = find_sensitive_columns(df, cfg)
    if not sensitive_cols:
        md = (
            "# Fairness (Fase 10)\n\n"
            f"- Dataset: `{csv_path.name}`\n"
            "- Nenhuma variável sensível encontrada entre os candidatos em `config.yaml`.\n"
        )
        (out_reports / "fairness_report.md").write_text(md, encoding="utf-8")
        return

    model_path = Path(cfg["paths"]["models_dir"]) / "best_model.joblib"
    model = joblib.load(model_path) if model_path.exists() else None

    meta_path = Path(cfg["paths"]["models_dir"]) / "modeling_meta.json"
    feature_cols = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = meta.get("feature_cols")

    rows: list[dict] = []
    weights = get_sample_weights(df, cfg)

    for col in sensitive_cols:
        groups = df[col].astype(str).fillna("NA")
        for g in groups.unique():
            idx = groups[groups == g].index
            y_g = y.loc[idx]
            if len(y_g) < 30:
                continue
            row: dict = {
                "sensitive_var": col,
                "group": g,
                "n": int(len(y_g)),
                "n_pos": int((y_g == 1).sum()),
                "prevalence": float(y_g.mean()),
            }
            if weights is not None:
                w = weights.loc[idx].fillna(0).to_numpy()
                if w.sum() > 0:
                    base_prev = float(np.average(y_g, weights=w))
                    row["weighted_prevalence"] = base_prev
                    
                    sum_sq = 0.0
                    count_brr = 0
                    for i in range(1, 81):
                        w_col = f"W_FSTURWT{i}"
                        if w_col in df.columns:
                            w_rep = df.loc[idx, w_col].fillna(0).to_numpy()
                            if w_rep.sum() > 0:
                                rep_prev = float(np.average(y_g, weights=w_rep))
                                sum_sq += (rep_prev - base_prev) ** 2
                                count_brr += 1
                    
                    if count_brr == 80:
                        row["weighted_prevalence_se"] = float(np.sqrt(0.05 * sum_sq))
                x_g = df.loc[idx, feature_cols].apply(pd.to_numeric, errors="coerce")
                try:
                    proba_g = model.predict_proba(x_g)[:, 1]
                    y_g_arr = y_g.to_numpy()

                    if y_g.nunique() > 1:
                        row["group_roc_auc"] = float(roc_auc_score(y_g_arr, proba_g))

                    # Limiar padrão 0.5 para métricas de classificação
                    y_pred_g = (proba_g >= 0.5).astype(int)
                    row["mean_predicted_proba"] = float(proba_g.mean())
                    # SR-3: TPR e FPR por grupo (base para Equal Opportunity)
                    row["tpr"] = _safe_tpr(y_g_arr, y_pred_g)
                    row["fpr"] = _safe_fpr(y_g_arr, y_pred_g)
                except Exception:
                    pass
            rows.append(row)

    fair_df = pd.DataFrame(rows).sort_values(["sensitive_var", "group"])
    fair_df.to_csv(out_tables / "fairness_by_group.csv", index=False)

    # SR-3: Métricas formais de equidade agregadas por variável sensível
    fairness_summary = _fairness_aggregate(fair_df)
    if not fairness_summary.empty:
        fairness_summary.to_csv(out_tables / "fairness_summary.csv", index=False)

    md = [
        "# Fairness (Fase 10)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Target: **{target_key}**\n",
        f"- Variáveis sensíveis analisadas: {', '.join(sensitive_cols)}\n\n",
        "## Métricas formais de equidade (SR-3)\n\n",
        "> **DPD** (Demographic Parity Difference): diferença máxima de prevalência entre grupos. "
        "Limiar recomendado: < 0.10.\n",
        "> **DIR** (Disparate Impact Ratio): razão min/max de prevalência. "
        "Limiar recomendado: > 0.80 (EEOC 4/5 rule).\n",
        "> **EOD** (Equal Opportunity Difference): diferença de TPR entre grupos. "
        "Limiar recomendado: < 0.10.\n\n",
    ]
    if not fairness_summary.empty:
        md.append(df_to_markdown(fairness_summary))
        md.append("\n\n")
    md.extend([
        "## Métricas por grupo\n\n",
        df_to_markdown(fair_df.head(60)),
        "\n\n",
        "## Nota metodológica\n",
        "Grupos com n < 30 foram excluídos por instabilidade estatística. "
        "AUCs de grupos pequenos têm alta variância e devem ser interpretadas com cautela. "
        "Para publicação, reportar também intervalos de confiança por grupo via bootstrap.\n",
    ])
    (out_reports / "fairness_report.md").write_text("".join(md), encoding="utf-8")
