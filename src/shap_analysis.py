from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.markdown import df_to_markdown
from src.utils.pipeline_data import get_target_series, select_feature_columns
from src.utils.seed import set_global_seed


def run_shap_analysis(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))

    models_dir = Path(cfg["paths"]["models_dir"])
    model_path = models_dir / "best_model.joblib"
    meta_path = models_dir / "modeling_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute primeiro: python3 -m src.main --stage modeling"
        )

    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    feature_cols = meta.get("feature_cols") or select_feature_columns(df, cfg)
    model = joblib.load(model_path)

    y, target_key = get_target_series(df, cfg)
    x = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    mask = y.notna()
    x, y = x.loc[mask], y.loc[mask].astype(int)

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "shap"
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    importance_df: pd.DataFrame | None = None
    method = "permutation_importance"

    # Amostra para SHAP (dataset grande)
    max_rows = int(cfg.get("shap", {}).get("max_rows", 800))
    if len(x) > max_rows:
        sample_idx = x.sample(max_rows, random_state=int(cfg.get("random_seed", 42))).index
        x_sample = x.loc[sample_idx]
        y_sample = y.loc[sample_idx]
    else:
        x_sample, y_sample = x, y

    try:
        import shap

        clf = model.named_steps.get("clf", model)
        if hasattr(model, "named_steps") and "imputer" in model.named_steps:
            x_imp = model.named_steps["imputer"].transform(x_sample)
            if "scaler" in model.named_steps:
                x_imp = model.named_steps["scaler"].transform(x_imp)
            x_matrix = x_imp
        else:
            x_matrix = x_sample.values

        if hasattr(clf, "feature_importances_"):
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(x_matrix)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            mean_abs = np.abs(shap_values).mean(axis=0)
            method = "shap_tree"
        else:
            explainer = shap.LinearExplainer(clf, x_matrix)
            shap_values = explainer.shap_values(x_matrix)
            mean_abs = np.abs(shap_values).mean(axis=0)
            method = "shap_linear"

        importance_df = pd.DataFrame(
            {"feature": feature_cols, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False)
    except Exception:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            x_sample,
            y_sample,
            n_repeats=5,
            random_state=int(cfg.get("random_seed", 42)),
            n_jobs=-1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "mean_abs_shap": result.importances_mean,
            }
        ).sort_values("mean_abs_shap", ascending=False)
        method = "permutation_importance"

    importance_df.to_csv(out_tables / "shap_feature_importance.csv", index=False)

    try:
        import matplotlib.pyplot as plt

        top = importance_df.head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top["feature"][::-1], top["mean_abs_shap"][::-1])
        plt.title(f"Feature importance ({method}) — target {target_key}")
        plt.tight_layout()
        plt.savefig(out_figures / "shap_top20.png", dpi=200)
        plt.close()
    except Exception:
        pass

    md = [
        "# SHAP / importância (Fase 9)\n\n",
        f"- Dataset: `{csv_path.name}`\n",
        f"- Target: **{target_key}**\n",
        f"- Método: `{method}`\n\n",
        df_to_markdown(importance_df.head(25)),
        "\n",
    ]
    (out_reports / "shap_report.md").write_text("".join(md), encoding="utf-8")
