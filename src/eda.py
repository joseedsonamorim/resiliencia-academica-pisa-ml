from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.markdown import df_to_markdown


def run_eda(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    """Fase 5 — EDA completa (aproximação robusta para dataset tabular grande).

    Gera:
    - outputs/reports/eda_report.md
    - outputs/tables/eda_numeric_describe.csv
    - outputs/tables/outliers_iqr_top20.csv
    - outputs/figures/eda/*.png
    """

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "eda"
    out_figures.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)

    report_path = out_reports / "eda_report.md"

    # Targets esperados: por construção do target, podem estar no dataframe ou somente em outputs/tables.
    target_cols = [
        c
        for c in df.columns
        if c.lower() in {"target_a", "target_b", "target_c", "target_d"}
        or c.lower().startswith("target_")
    ]

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    dupes = int(df.duplicated().sum())

    # Missingness
    missing_ratio = df.isna().mean().sort_values(ascending=False)

    # Descritivas (numéricas)
    desc = df[numeric_cols].describe(
        percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    ).T

    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_tables.mkdir(parents=True, exist_ok=True)
    desc_path = out_tables / "eda_numeric_describe.csv"
    desc.to_csv(desc_path, index=True)

    md: list[str] = []
    md.append("# EDA report\n\n")
    md.append(f"- Dataset: `{csv_path.name}`\n")
    md.append(f"- Linhas: {df.shape[0]}\n")
    md.append(f"- Colunas: {df.shape[1]}\n")
    md.append(f"- Duplicatas (linhas inteiras): {dupes}\n\n")

    md.append("## Missingness (top 25 colunas)\n\n")
    md.append(df_to_markdown(missing_ratio.head(25).to_frame("missing_ratio"), index=True))
    md.append("\n\n")

    md.append("## Descritivas (numéricas)\n\n")
    md.append(f"- Tabela: `{desc_path.name}`\n\n")

    # Histogramas (amostragem)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Top 12 por cardinalidade
        card = []
        for c in numeric_cols:
            try:
                card.append((c, df[c].nunique(dropna=True)))
            except Exception:
                continue
        card.sort(key=lambda x: x[1], reverse=True)
        top_cols = [c for c, n in card if n > 1][:12]

        for c in top_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) == 0:
                continue
            s = s.sample(min(len(s), 5000), random_state=int(cfg.get("random_seed", 42)))
            plt.figure(figsize=(7, 4))
            plt.hist(s, bins=30)
            plt.title(f"Histogram: {c}")
            plt.tight_layout()
            plt.savefig(out_figures / f"hist_{c}.png", dpi=200)
            plt.close()
    except Exception:
        pass

    # Heatmap de correlação (Pearson) com subset
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Filtra constantes/quase constantes
        cand = []
        for c in numeric_cols:
            s = pd.to_numeric(df[c], errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) < 20:
                continue
            if s.nunique(dropna=True) <= 1:
                continue
            cand.append((c, float(s.std(ddof=0))))
        cand.sort(key=lambda x: x[1], reverse=True)
        corr_cols = [c for c, _ in cand[:35]]

        corr_df = df[corr_cols].apply(pd.to_numeric, errors="coerce")
        corr_df = corr_df.replace([np.inf, -np.inf], np.nan)
        corr_df = corr_df.dropna(axis=0, how="any")
        if len(corr_df) >= 20:
            corr = corr_df.corr(method="pearson")
            plt.figure(figsize=(14, 10))
            sns.heatmap(corr, cmap="coolwarm", center=0)
            plt.title("Correlation heatmap (Pearson)")
            plt.tight_layout()
            plt.savefig(out_figures / "corr_heatmap.png", dpi=200)
            plt.close()
    except Exception:
        pass

    # Outliers por IQR (top 20)
    outlier_counts: list[tuple[str, int]] = []
    for c in numeric_cols:
        s = pd.to_numeric(df[c], errors="coerce")
        s = s.replace([float("inf"), float("-inf")], pd.NA).dropna()
        if len(s) < 20:
            continue
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        cnt = int(((s < low) | (s > high)).sum())
        outlier_counts.append((c, cnt))

    outlier_counts.sort(key=lambda x: x[1], reverse=True)
    outlier_top = outlier_counts[:20]
    outlier_df = pd.DataFrame(outlier_top, columns=["column", "outlier_iqr_count"])

    outlier_path = out_tables / "outliers_iqr_top20.csv"
    outlier_df.to_csv(outlier_path, index=False)

    md.append("## Outliers (IQR) — top 20\n\n")
    md.append(df_to_markdown(outlier_df))
    md.append("\n\n")

    md.append("## Target distribution (se existir no dataframe)\n\n")
    if target_cols:
        rows = []
        for c in target_cols:
            y = pd.to_numeric(df[c], errors="coerce")
            rows.append({"target_col": c, "mean": float(y.mean()), "sum": int(y.sum())})
        md.append(df_to_markdown(pd.DataFrame(rows)))
        md.append("\n\n")
    else:
        md.append(
            "- Nenhuma coluna `target_*` encontrada no dataframe nesta etapa "
            "(os targets A/B/C/D podem existir como artefato em `outputs/tables`).\n\n"
        )

    md.append(
        "## Artefatos de visualização\n\n"
        "- Missing values: `outputs/figures/missing_values.png` (fase 1)\n"
        "- Data quality: `outputs/figures/data_quality.png` (fase 1)\n"
        "- EDA figures: `outputs/figures/eda/*`\n"
    )

    md.append("\n> Próximas etapas: Fase 6 (perfil dos resilientes) e Fase 7 (clusterização).\n")

    report_path.write_text("".join(md), encoding="utf-8")
