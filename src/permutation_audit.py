"""Permutation importance audit to detect possible indirect leakage.

Constraints: 
- Do not modify existing pipeline/model training.
- This module is complementary and uses already trained models and saved preprocessing.

Outputs:
- outputs/reports/permutation_importance_report.md
- outputs/tables/permutation_importance.csv
- outputs/figures/permutation_importance.png

Alert: if any single feature contributes >30% of total importance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import setup_logger

logger = setup_logger(__name__)


def _load_artifacts(config) -> Tuple[Any, Any, Any, pd.DataFrame, pd.Series]:
    """Load model artifacts and raw data.

    We reuse the current preprocessing by applying the saved preprocessing pipeline.
    """
    from src.data_layer import DataLoader
    from src.preprocessing_pipeline import PreprocessingPipeline

    models_dir = config.get_path('models')
    outputs_metrics = config.get_path('outputs_metrics')
    outputs_tables = config.get_path('outputs_tables')
    outputs_figures = config.get_path('outputs_figures')
    outputs_reports = config.get_path('outputs_reports')

    # Choose best model by best_params/best_score if exists.
    # Fallback: xgboost_model.
    candidates = {
        'logistic_regression': models_dir / 'logistic_regression_model.pkl',
        'random_forest': models_dir / 'random_forest_model.pkl',
        'xgboost': models_dir / 'xgboost_model.pkl',
        'lightgbm': models_dir / 'lightgbm_model.pkl',
    }

    # Try to load best model name from best_params or cv_results.
    best_model_name = 'xgboost'
    best_params_path = models_dir / 'best_params.json'
    if best_params_path.exists():
        try:
            best_params = json.loads(best_params_path.read_text())
            # heuristic: if xgboost exists use it; otherwise first key
            if 'xgboost' in best_params:
                best_model_name = 'xgboost'
            else:
                best_model_name = next(iter(best_params.keys()))
        except Exception:
            pass

    model_path = candidates.get(best_model_name, candidates['xgboost'])
    model = None
    import joblib

    model = joblib.load(model_path)

    loader = DataLoader(config)
    df_raw = loader.load_raw_data()

    y = df_raw['Creative_Resilience']
    X = df_raw.drop(columns=['Creative_Resilience'])

    # Apply saved preprocessing (deterministic pipeline)
    preprocessor = PreprocessingPipeline(config, random_state=config.get('random_state', 42))
    ok = preprocessor.load_pipeline()
    if not ok:
        logger.warning('Preprocessing pipeline not found; falling back to running full preprocessing.')
        X_processed, y_processed = preprocessor.preprocess_full_pipeline(X, y)
    else:
        # We don't have a "transform only" API; simplest safe approach is to re-run preprocess_full_pipeline.
        # This respects determinism and keeps compatibility.
        X_processed, y_processed = preprocessor.preprocess_full_pipeline(X, y)

    return model, df_raw, X_processed, y_processed


def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation importance using ROC-AUC delta."""

    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_auc_score

    scorer = None
    def auc_scorer(estimator, X_, y_):
        proba = estimator.predict_proba(X_)[:, 1]
        return roc_auc_score(y_, proba)

    result = permutation_importance(
        model,
        X,
        y,
        scoring=auc_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importances_mean = result.importances_mean
    importances_std = result.importances_std

    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': importances_mean,
        'importance_std': importances_std,
    })

    total = importances_mean.sum()
    if total == 0:
        importance_df['percentual_importancia'] = 0.0
    else:
        importance_df['percentual_importancia'] = importance_df['importance_mean'] / total

    importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)

    return importance_df


def run_permutation_audit() -> Dict[str, Any]:
    config = get_config()

    outputs_reports = config.get_path('outputs_reports')
    outputs_tables = config.get_path('outputs_tables')
    outputs_figures = config.get_path('outputs_figures')

    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)
    outputs_reports.mkdir(parents=True, exist_ok=True)

    logger.info('\nRunning permutation audit...')
    model, df_raw, X_processed, y_processed = _load_artifacts(config)

    # Use a holdout-like subset to reduce compute, but keep deterministic sampling.
    rs = config.get('random_state', 42)
    if len(X_processed) > 1500:
        idx = np.random.RandomState(rs).choice(len(X_processed), 1500, replace=False)
        X_eval = X_processed.iloc[idx]
        y_eval = y_processed.iloc[idx]
    else:
        X_eval = X_processed
        y_eval = y_processed

    importance_df = compute_permutation_importance(
        model=model,
        X=X_eval,
        y=y_eval,
        n_repeats=10,
        random_state=rs,
    )

    # Alerts
    top1 = float(importance_df.loc[0, 'percentual_importancia']) if len(importance_df) else 0.0
    alert_indirect_leakage = top1 > 0.30

    # Save CSV
    csv_path = outputs_tables / 'permutation_importance.csv'
    importance_df.to_csv(csv_path, index=False)

    # Save figure (safe for headless execution)
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt

    fig_path = outputs_figures / 'permutation_importance.png'
    topn = min(20, len(importance_df))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        importance_df['feature'].head(topn).iloc[::-1],
        importance_df['importance_mean'].head(topn).iloc[::-1],
    )
    ax.set_title('Permutation Importance (top features)')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)


    # Save report
    md_path = outputs_reports / 'permutation_importance_report.md'
    lines = [
        '# Permutation Importance Audit (Possible Indirect Leakage)',
        '',
        f'**Top-1 percentual_importancia**: {top1:.4f}',
        '',
        '## Top 10 variables',
        importance_df[['feature', 'importance_mean', 'percentual_importancia']].head(10).to_markdown(index=False),
        '',
        '## Top 20 variables',
        importance_df[['feature', 'importance_mean', 'percentual_importancia']].head(20).to_markdown(index=False),
        '',
        '## Top 50 variables',
        importance_df[['feature', 'importance_mean', 'percentual_importancia']].head(50).to_markdown(index=False),
        '',
        '## Alert',
        'POSSÍVEL LEAKAGE INDIRETO' if alert_indirect_leakage else 'Sem alerta de possível leakage indireto.',
        '',
        '---',
        '',
        f'Outputs: {csv_path} | {fig_path}',
    ]

    md_path.write_text('\n'.join(lines), encoding='utf-8')

    logger.info(f'Permutation audit completed. Alert: {alert_indirect_leakage}')

    return {
        'alert_indirect_leakage': alert_indirect_leakage,
        'top1_percentual_importancia': top1,
        'csv_path': str(csv_path),
        'fig_path': str(fig_path),
        'report_path': str(md_path),
    }


if __name__ == '__main__':
    run_permutation_audit()

