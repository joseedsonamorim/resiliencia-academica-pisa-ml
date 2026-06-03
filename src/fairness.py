from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.utils.pipeline_data import find_sensitive_columns, get_sample_weights, get_target_series
from src.utils.seed import set_global_seed


def run_fairness(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    y, target_key = get_target_series(df, cfg)
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

    import json

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
                "prevalence": float(y_g.mean()),
            }
            if weights is not None:
                w = weights.loc[idx]
                row["weighted_prevalence"] = float(np.average(y_g, weights=w))

            if model is not None and feature_cols:
                x_g = df.loc[idx, feature_cols].apply(pd.to_numeric, errors="coerce")
                try:
                    proba = model.predict_proba(x_g)[:, 1]
                    if y_g.nunique() > 1:
                        row["group_roc_auc"] = float(roc_auc_score(y_g, proba))
                    row["mean_predicted_proba"] = float(proba.mean())
                except Exception:
                    pass
            rows.append(row)

    fair_df = pd.DataFrame(rows).sort_values(["sensitive_var", "group"])
    fair_df.to_csv(out_tables / "fairness_by_group.csv", index=False)

    md = [
        "# Fairness (Fase 10)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Target: **{target_key}**\n",
        f"- Variáveis sensíveis analisadas: {', '.join(sensitive_cols)}\n\n",
        fair_df.head(40).to_markdown(index=False),
        "\n",
    ]
    (out_reports / "fairness_report.md").write_text("".join(md), encoding="utf-8")
