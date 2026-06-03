from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.utils.pipeline_data import get_target_series, select_feature_columns
from src.utils.seed import set_global_seed


def run_robustness(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))
    seed = int(cfg.get("random_seed", 42))

    model_path = Path(cfg["paths"]["models_dir"]) / "best_model.joblib"
    meta_path = Path(cfg["paths"]["models_dir"]) / "modeling_meta.json"
    if not model_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute primeiro: python3 -m src.main --stage modeling"
        )

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    feature_cols = meta.get("feature_cols") or select_feature_columns(df, cfg)

    y, target_key = get_target_series(df, cfg)
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask = y.notna()
    x, y = x.loc[mask], y.loc[mask].astype(int)

    proba = model.predict_proba(x)[:, 1]

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "robustness"
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    n_boot = int(cfg.get("robustness", {}).get("n_bootstrap", 200))
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_b = y.iloc[idx]
        p_b = proba[idx]
        if y_b.nunique() < 2:
            continue
        aucs.append(float(roc_auc_score(y_b, p_b)))

    boot_df = pd.DataFrame(
        {
            "metric": ["roc_auc"],
            "mean": [float(np.mean(aucs)) if aucs else float("nan")],
            "std": [float(np.std(aucs)) if aucs else float("nan")],
            "ci_low": [float(np.percentile(aucs, 2.5)) if aucs else float("nan")],
            "ci_high": [float(np.percentile(aucs, 97.5)) if aucs else float("nan")],
        }
    )
    boot_df.to_csv(out_tables / "robustness_bootstrap.csv", index=False)

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
        "# Robustness (Fase 11)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Target: **{target_key}**\n",
        f"- Bootstrap iterations: {n_boot}\n",
        f"- Brier score: {brier:.4f}\n\n",
        boot_df.to_markdown(index=False),
        "\n",
    ]
    (out_reports / "robustness_report.md").write_text("".join(md), encoding="utf-8")
