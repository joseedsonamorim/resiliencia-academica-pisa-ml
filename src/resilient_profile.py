"""Resilient student profile analysis.

Constraints: add complementary analysis only.

Outputs:
- outputs/reports/resilient_profile.md
- outputs/tables/resilient_profile.csv
- outputs/figures/resilient_profile_radar.png
- outputs/figures/resilient_profile_heatmap.png
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import pandas as pd

from src.config import get_config
from src.utils import setup_logger

logger = setup_logger(__name__)


def _cohen_d(x1: np.ndarray, x2: np.ndarray) -> float:
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / max(n1 + n2 - 2, 1)
    sd = np.sqrt(pooled)
    return float((np.mean(x1) - np.mean(x2)) / sd) if sd > 0 else float('nan')


def run_resilient_profile() -> Dict[str, Any]:
    from src.data_layer import DataLoader
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt


    config = get_config()
    rs = config.get('random_state', 42)

    outputs_reports = config.get_path('outputs_reports')
    outputs_tables = config.get_path('outputs_tables')
    outputs_figures = config.get_path('outputs_figures')
    outputs_reports.mkdir(parents=True, exist_ok=True)
    outputs_tables.mkdir(parents=True, exist_ok=True)
    outputs_figures.mkdir(parents=True, exist_ok=True)

    logger.info('\nRunning resilient profile analysis...')
    loader = DataLoader(config)
    df = loader.load_raw_data()

    if 'Creative_Resilience' not in df.columns:
        raise ValueError('Creative_Resilience column not found')

    resilient = df[df['Creative_Resilience'] == 1]
    non_resilient = df[df['Creative_Resilience'] == 0]

    # Heuristic variable groups: use columns that exist among known names.
    candidate_cols = [
        # Tech / engagement-ish
        'HOMEPOS', 'ICTRES', 'ST004D01T',
        # Socioeconomic-ish
        'ESCS', 'HISCED',
        # Creativity/reading proxies (if present)
        'CRT_SCORE',
    ]

    candidate_cols = [c for c in candidate_cols if c in df.columns]

    rows = []
    for col in candidate_cols:
        x1 = resilient[col].astype(float).values
        x0 = non_resilient[col].astype(float).values
        rows.append({
            'variable': col,
            'mean_resilient': float(np.mean(x1)),
            'median_resilient': float(np.median(x1)),
            'std_resilient': float(np.std(x1, ddof=1)) if len(x1) > 1 else float('nan'),
            'mean_non_resilient': float(np.mean(x0)),
            'median_non_resilient': float(np.median(x0)),
            'std_non_resilient': float(np.std(x0, ddof=1)) if len(x0) > 1 else float('nan'),
            'cohen_d': _cohen_d(x1, x0),
        })

    out_df = pd.DataFrame(rows).sort_values('cohen_d', key=lambda s: s.abs(), ascending=False)

    csv_path = outputs_tables / 'resilient_profile.csv'
    out_df.to_csv(csv_path, index=False)

    # Radar chart: use top 6 variables by |d|
    topk = min(6, len(out_df))
    radar_vars = out_df.head(topk)['variable'].tolist()

    if topk > 1:
        means1 = out_df.head(topk)['mean_resilient'].values
        means0 = out_df.head(topk)['mean_non_resilient'].values

        # normalize to [0,1] for visualization
        all_means = np.concatenate([means1, means0])
        mn, mx = np.min(all_means), np.max(all_means)
        if mx - mn == 0:
            m1n = means1
            m0n = means0
        else:
            m1n = (means1 - mn) / (mx - mn)
            m0n = (means0 - mn) / (mx - mn)

        angles = np.linspace(0, 2 * np.pi, topk, endpoint=False).tolist()
        angles += angles[:1]

        m1n = m1n.tolist(); m0n = m0n.tolist()
        m1n += m1n[:1]; m0n += m0n[:1]

        fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={'polar': True})
        ax.plot(angles, m1n, label='Resiliente')
        ax.fill(angles, m1n, alpha=0.25)
        ax.plot(angles, m0n, label='Não-resiliente')
        ax.fill(angles, m0n, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_vars, fontsize=9)
        ax.set_title('Perfil do aluno criativamente resiliente (Radar)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
        radar_path = outputs_figures / 'resilient_profile_radar.png'
        fig.tight_layout()
        fig.savefig(radar_path, dpi=200)
        plt.close(fig)


    # Heatmap of cohen_d
    fig, ax = plt.subplots(figsize=(10, 3))
    heat_df = out_df[['variable', 'cohen_d']].head(12).set_index('variable')
    im = ax.imshow(heat_df.values, aspect='auto', cmap='coolwarm')
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)
    ax.set_xticks([0])
    ax.set_xticklabels(['Cohen d'])
    fig.colorbar(im, ax=ax, label='Cohen d (|d| maior = efeito maior)')
    ax.set_title('Efeito (Cohen\'s d) — Resiliente vs Não-resiliente')
    heat_path = outputs_figures / 'resilient_profile_heatmap.png'
    fig.tight_layout()
    fig.savefig(heat_path, dpi=200)
    plt.close(fig)


    # Markdown interpretation
    md_lines = [
        '# Perfil do Aluno Criativamente Resiliente',
        '',
        '## Comparação Resiliente vs Não-resiliente',
        '',
        out_df[['variable', 'mean_resilient', 'mean_non_resilient', 'cohen_d']].head(10).to_markdown(index=False),
        '',
        '## Interpretação (automática, português)',
        'Em geral, as variáveis com |Cohen\'s d| mais alto indicam diferenças mais fortes entre os grupos. Variáveis com Cohen\'s d positivo tendem a ser maiores no grupo resiliente (na escala original do dataset).',
        '',
        '## Outputs',
        f'- {csv_path}',
        f'- {radar_path if topk > 1 else "(sem radar)"}',
        f'- {heat_path}',
    ]

    md_path = outputs_reports / 'resilient_profile.md'
    md_path.write_text('\n'.join(md_lines), encoding='utf-8')

    logger.info(f'Resilient profile saved: {md_path}')

    return {
        'report_path': str(md_path),
        'csv_path': str(csv_path),
        'radar_path': str(outputs_figures / 'resilient_profile_radar.png'),
        'heatmap_path': str(outputs_figures / 'resilient_profile_heatmap.png'),
    }


if __name__ == '__main__':
    run_resilient_profile()

