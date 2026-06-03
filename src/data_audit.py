from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.seed import set_global_seed
from src.utils.io import ensure_parent


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    # try parse numeric
    try:
        pd.to_numeric(series.dropna().astype(str), errors="raise")
        return "numeric"
    except Exception:
        return "categorical"


def _cardinality_no_na(series: pd.Series) -> int:
    return series.dropna().nunique(dropna=True)


def _missing_stats(df: pd.DataFrame) -> pd.Series:
    return df.isna().mean()


def run_data_audit(cfg: dict, csv_path: Path) -> None:
    set_global_seed(int(cfg.get("random_seed", 42)))

    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])

    for p in [out_reports, out_figures, out_tables]:
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, low_memory=False)

    # --- Basic artifacts for inventory
    records = []

    missing_ratio = _missing_stats(df)
    dupes = int(df.duplicated().sum())

    # heuristic IDs / weights
    id_candidates = [c for c in df.columns if "id" in c.lower() or "cntstu" in c.lower() or "stuid" in c.lower()]
    weight_candidates = [c for c in df.columns if c.lower().startswith("w_") or "weight" in c.lower() or c.lower().endswith("wt")]

    const_threshold = 0.999
    near_const_threshold = 0.995

    # compute for every column
    for col in df.columns:
        s = df[col]
        non_na = s.dropna()
        miss = float(missing_ratio[col])
        card = int(_cardinality_no_na(s))

        inferred_type = _infer_type(s)

        # constant / near constant
        top = None
        is_constant = False
        is_near_constant = False
        if len(non_na) > 0:
            vc = non_na.value_counts(dropna=True)
            if len(vc) > 0:
                top_val, top_cnt = vc.index[0], int(vc.iloc[0])
                top = (top_val, top_cnt)
                frac = top_cnt / len(non_na)
                is_constant = frac >= const_threshold
                is_near_constant = (frac >= near_const_threshold) and not is_constant

        category_func = "Unknown"
        lcol = col.lower()
        if any(k in lcol for k in ["crt", "creative"]):
            category_func = "Criatividade"
        elif any(k in lcol for k in ["escs", "hom" ,"edu", "parent", "status"]):
            category_func = "Socioeconômicas/Educacionais"
        elif any(k in lcol for k in ["w_", "wt", "weight"]):
            category_func = "Pesos amostrais"
        elif any(k in lcol for k in ["sexo", "gender"]):
            category_func = "Demográficas"

        records.append({
            "nome": col,
            "tipo": inferred_type,
            "missing": miss,
            "cardinalidade": card,
            "categoria_funcional": category_func,
            "duplicatas_totalmente_iguais": "",
            "constante": is_constant,
            "quase_constante": is_near_constant,
        })

    inv = pd.DataFrame(records)

    # save dataset_inventory.md
    inv_md_path = out_reports / "dataset_inventory.md"
    inv_sorted = inv.sort_values(["missing", "cardinalidade"], ascending=[False, True])
    # pandas.to_markdown requires optional dependency `tabulate`.
    # If unavailable, fallback to a simple CSV/tsv-like markdown.
    try:
        inv_md = inv_sorted.to_markdown(index=False)
    except ImportError:
        inv_md = inv_sorted.head(500).to_csv(index=False, sep="|")
        inv_md = "(fallback) " + inv_md

    ensure_parent(inv_md_path)
    inv_md_path.write_text(inv_md, encoding="utf-8")


    # --- data_audit.md (narrative)
    audit_path = out_reports / "data_audit.md"
    missing_top = missing_ratio.sort_values(ascending=False).head(15)

    audit_txt = [
        "# Data Audit\n",
        f"- Dataset: {csv_path.name}\n",
        f"- Linhas: {df.shape[0]}\n",
        f"- Colunas: {df.shape[1]}\n",
        f"- Duplicatas (linhas inteiras): {dupes}\n",
        "\n## Missing (top 15)\n",
        missing_top.to_frame("missing_ratio").to_markdown(),
        "\n## Heurísticas\n",
        f"- Possíveis IDs: {id_candidates[:30]}\n",
        f"- Possíveis pesos amostrais: {weight_candidates[:30]}\n",
    ]

    ensure_parent(audit_path)
    audit_path.write_text("".join(audit_txt), encoding="utf-8")

    # --- Figures
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # missing values heatmap (subsample columns for readability)
        cols = df.columns
        # pick up to 80 columns with highest missingness
        top_cols = missing_ratio.sort_values(ascending=False).head(min(80, len(cols))).index
        sub = df[top_cols]
        plt.figure(figsize=(14, 6))
        sns.heatmap(sub.isna(), cbar=False)
        plt.title("Missing values (heatmap - top missing columns)")
        fig_path = out_figures / "missing_values.png"
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()

        # data quality overview
        plt.figure(figsize=(12, 4))
        missing_ratio_plot = missing_ratio.sort_values(ascending=False)
        missing_ratio_plot.head(50).plot(kind="bar")
        plt.ylabel("Missing ratio")
        plt.title("Data quality: missingness (top 50)")
        fig2_path = out_figures / "data_quality.png"
        plt.tight_layout()
        plt.savefig(fig2_path, dpi=200)
        plt.close()

    except Exception:
        # Figures are best-effort.
        pass

