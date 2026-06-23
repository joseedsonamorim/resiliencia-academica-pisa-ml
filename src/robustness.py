from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import get_target_series
from src.utils.seed import set_global_seed



def run_robustness(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))
    seed = int(cfg.get("random_seed", 42))

    meta_path = Path(cfg["paths"]["models_dir"]) / "modeling_meta.json"
    predictions_path = Path(cfg["outputs"]["tables_dir"]) / "modeling_holdout_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(
            "Predições do holdout não encontradas. "
            "Execute primeiro: python3 -m src.main --stage modeling"
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    target_key = str(meta.get("target_key", get_target_series(df, cfg)[1]))

    pred_df = pd.read_csv(predictions_path)
    required_cols = {"y_true", "proba_resilient", "row_id"}
    if not required_cols.issubset(pred_df.columns):
        raise ValueError(
            f"modeling_holdout_predictions.csv deve conter as colunas {required_cols}. "
            f"Colunas encontradas: {list(pred_df.columns)}"
        )

    y = pred_df["y_true"].astype(int).to_numpy()
    proba = pred_df["proba_resilient"].to_numpy(dtype=float)
    row_ids = pred_df["row_id"].to_numpy()

    if len(y) == 0 or np.unique(y).size < 2:
        raise ValueError("holdout insuficiente ou sem variação de classe para bootstrap.")

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "robustness"
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    n_boot = int(cfg.get("robustness", {}).get("n_bootstrap", 200))
    rng = np.random.default_rng(seed)

    # ── Bootstrap exclusivamente no holdout ──
    aucs: list[float] = []
    aps: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y[idx]
        p_b = proba[idx]
        if len(np.unique(y_b)) < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, p_b)))
        aps.append(float(average_precision_score(y_b, p_b)))

    def _boot_row(metric: str, values: list[float]) -> dict:
        if not values:
            return {"metric": metric, "mean": float("nan"), "std": float("nan"),
                    "ci_low": float("nan"), "ci_high": float("nan"), "n_boot": 0}
        return {
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_low": float(np.percentile(values, 2.5)),
            "ci_high": float(np.percentile(values, 97.5)),
            "n_boot": len(values),
        }

    boot_df = pd.DataFrame([
        _boot_row("roc_auc", aucs),
        _boot_row("average_precision", aps),
    ])
    boot_df.to_csv(out_tables / "modeling_best_bootstrap.csv", index=False)
    boot_df.to_csv(out_tables / "robustness_bootstrap.csv", index=False)

    # ── BRR FAY REPLICATE WEIGHTS (Nível 1A) ──
    # Calcular o Erro Padrão do ROC-AUC usando os 80 pesos de replicação
    df_holdout = df.loc[row_ids]
    brr_se_auc = float("nan")
    brr_se_ap = float("nan")
    base_auc_w = float("nan")
    base_ap_w = float("nan")
    
    if "W_FSTUWT" in df_holdout.columns:
        w_base = df_holdout["W_FSTUWT"].fillna(0).to_numpy()
        if w_base.sum() > 0:
            try:
                base_auc_w = roc_auc_score(y, proba, sample_weight=w_base)
                base_ap_w = average_precision_score(y, proba, sample_weight=w_base)
                
                sum_sq_auc = 0.0
                sum_sq_ap = 0.0
                count_brr = 0
                for i in range(1, 81):
                    w_col = f"W_FSTURWT{i}"
                    if w_col in df_holdout.columns:
                        w_rep = df_holdout[w_col].fillna(0).to_numpy()
                        rep_auc = roc_auc_score(y, proba, sample_weight=w_rep)
                        rep_ap = average_precision_score(y, proba, sample_weight=w_rep)
                        sum_sq_auc += (rep_auc - base_auc_w) ** 2
                        sum_sq_ap += (rep_ap - base_ap_w) ** 2
                        count_brr += 1
                
                if count_brr == 80:
                    brr_se_auc = np.sqrt(0.05 * sum_sq_auc)
                    brr_se_ap = np.sqrt(0.05 * sum_sq_ap)
            except Exception:
                pass


    brier = float(brier_score_loss(y, proba))
    try:
        prob_true, prob_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        cal_df = pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})
        cal_df.to_csv(out_tables / "robustness_calibration.csv", index=False)
    except Exception:
        cal_df = None

    try:
        import matplotlib.pyplot as plt

        if cal_df is not None:
            plt.figure(figsize=(6, 5))
            plt.plot([0, 1], [0, 1], "k--", label="perfeita")
            plt.plot(cal_df["prob_pred"], cal_df["prob_true"], marker="o", label="modelo")
            plt.xlabel("Probabilidade prevista")
            plt.ylabel("Frequência observada")
            plt.title(f"Calibração — target {target_key}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_figures / "calibration_curve.png", dpi=200)
            plt.close()
    except Exception:
        pass

    md = [
        "# Robustez e Calibração (Fase 11)\\n\\n",
        f"- Dataset: `{csv_path.name}`\\n",
        f"- Target: **{target_key}**\\n",
        f"- Bootstrap iterations: {n_boot} (somente sobre holdout)\\n",
        f"- N holdout: {n}\\n",
        f"- Brier score (holdout): {brier:.4f}\\n\\n",
        "## Bootstrap de métricas (holdout)\\n\\n",
        "> As estimativas abaixo são calculadas **exclusivamente** sobre o conjunto de teste "
        "(holdout), nunca sobre os dados de treino. Cada iteração reamostraliza com "
        "reposição o holdout para estimar a variabilidade das métricas.\\n\\n",
        df_to_markdown(boot_df),
        "\\n\\n",
        "## Estimativa de Erro Padrão via Regras de Replicação BRR (Fay)\\n\\n",
        "> O erro padrão da performance é estimado recalculando a métrica em 80 amostras "
        "modificadas segundo os pesos de replicação fornecidos pela OCDE (`W_FSTURWT1` a `W_FSTURWT80`), "
        "garantindo que o desenho amostral complexo do PISA foi levado em conta (Fay's method).\\n\\n",
        f"- **ROC-AUC (Base Weight)**: {base_auc_w:.4f}\\n",
        f"- **ROC-AUC (BRR Standard Error)**: {brr_se_auc:.4f}\\n",
        f"- **PR-AUC (Base Weight)**: {base_ap_w:.4f}\\n",
        f"- **PR-AUC (BRR Standard Error)**: {brr_se_ap:.4f}\\n",
        "\\n",
    ]
    (out_reports / "robustness_report.md").write_text("".join(md), encoding="utf-8")
