"""Interpretable clustering report.

This project has a clustering page placeholder. We generate a complementary
scientific interpretation report using already processed embeddings/cluster
assignments if present; otherwise we generate a best-effort report based on
k-means in preprocessed space.

Outputs:
- outputs/reports/cluster_interpretation.md
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import setup_logger

logger = setup_logger(__name__)


def run_cluster_interpretation(n_clusters: int = 4) -> Dict[str, Any]:
    from src.data_layer import DataLoader
    from src.preprocessing_pipeline import PreprocessingPipeline

    config = get_config()
    outputs_reports = config.get_path('outputs_reports')
    outputs_reports.mkdir(parents=True, exist_ok=True)

    logger.info('\nRunning cluster interpretation...')

    loader = DataLoader(config)
    df_raw = loader.load_raw_data()

    y = df_raw['Creative_Resilience']
    X = df_raw.drop(columns=['Creative_Resilience'])

    preprocessor = PreprocessingPipeline(config, random_state=config.get('random_state', 42))
    X_proc, y_proc = preprocessor.preprocess_full_pipeline(X, y)

    from sklearn.cluster import KMeans

    rs = config.get('random_state', 42)
    km = KMeans(n_clusters=n_clusters, random_state=rs, n_init='auto')
    labels = km.fit_predict(X_proc)

    # Try to map CRT/ESCS means if available in raw df
    cols_for_stats = ['CRT_SCORE', 'ESCS']
    cols_for_stats = [c for c in cols_for_stats if c in df_raw.columns]

    report_lines = ['# Cluster Interpretation (Scientific)']

    # Determine cluster characteristics in raw numeric columns (limited set)
    for c in range(n_clusters):
        mask = labels == c
        size = int(mask.sum())
        pct = size / len(labels)

        stats = {'cluster': c, 'size': size, 'pct': pct}
        for col in cols_for_stats:
            stats[f'mean_{col}'] = float(df_raw.loc[mask, col].astype(float).mean())

        # Cohen-like naming heuristic
        crt_mean = stats.get('mean_CRT_SCORE', 0.0)
        escs_mean = stats.get('mean_ESCS', 0.0)

        if 'mean_CRT_SCORE' in stats and 'mean_ESCS' in stats:
            # simplistic mapping
            name = 'Criativamente Resilientes'
            if crt_mean < np.median(df_raw['CRT_SCORE']):
                name = 'Baixa Criatividade e Baixos Recursos'
            if crt_mean >= np.median(df_raw['CRT_SCORE']) and escs_mean >= np.median(df_raw['ESCS']):
                name = 'Alta Criatividade e Alto Capital Cultural'
            if crt_mean >= np.median(df_raw['CRT_SCORE']) and escs_mean < np.median(df_raw['ESCS']):
                name = 'Potencial Criativo Emergente'
        else:
            name = f'Cluster {c}'

        report_lines.append('')
        report_lines.append(f"## Cluster {c} — {name}")
        report_lines.append(f"- Tamanho: {size} ({pct*100:.2f}%)")
        for k, v in stats.items():
            if k.startswith('mean_'):
                report_lines.append(f"- {k.replace('mean_', 'Média ')}: {v:.4f}")

        # Top processed features by kmeans centroid distance proxy
        centroid = km.cluster_centers_[c]
        feat_names = X_proc.columns.tolist()
        top_idx = np.argsort(np.abs(centroid))[::-1][:5]
        top_feats = [feat_names[i] for i in top_idx]
        report_lines.append(f"- Principais features (proxy por centroid): {', '.join(top_feats)}")

    md_path = outputs_reports / 'cluster_interpretation.md'
    md_path.write_text('\n'.join(report_lines), encoding='utf-8')

    logger.info(f'Cluster interpretation saved: {md_path}')

    return {
        'report_path': str(md_path),
        'n_clusters': n_clusters,
    }


if __name__ == '__main__':
    run_cluster_interpretation()

