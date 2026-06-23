from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.markdown import df_to_markdown


def _select_targets(df: pd.DataFrame) -> dict[str, pd.Series]:
    # Prefer explicit target columns already present
    targets = {}
    for k in ["A", "B", "C", "D"]:
        for cand in [f"target_{k}", f"Target_{k}", f"target_{k.lower()}", f"target{k}"]:
            if cand in df.columns:
                s = pd.to_numeric(df[cand], errors="coerce")
                targets[k] = (s.fillna(0) > 0).astype(int)
                break
    return targets


def _mannwhitney_stat(
    x_pos: pd.Series,
    x_neg: pd.Series,
) -> tuple[float, float, float]:
    """Retorna (U, p_value, effect_size_r).

    Effect size r = Z / sqrt(N) — convenção para Mann-Whitney (Cohen 1988).
    Se scipy não estiver disponível, retorna NaN.
    """
    try:
        from scipy import stats

        xp = x_pos.dropna().to_numpy(dtype=float)
        xn = x_neg.dropna().to_numpy(dtype=float)
        n_total = len(xp) + len(xn)
        if len(xp) < 5 or len(xn) < 5 or n_total == 0:
            return float("nan"), float("nan"), float("nan")
        res = stats.mannwhitneyu(xp, xn, alternative="two-sided", method="auto")
        u_stat = float(res.statistic)
        p_val = float(res.pvalue)
        # Z approximation for effect size
        n1, n2 = len(xp), len(xn)
        mu = n1 * n2 / 2.0
        sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u_stat - mu) / sigma if sigma > 0 else 0.0
        r = abs(z) / np.sqrt(n_total)
        return u_stat, p_val, float(r)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def _fdr_correction(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction.

    Tenta usar statsmodels; fallback para implementação própria.
    """
    p = np.array(p_values, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return list(p)

    try:
        from statsmodels.stats.multitest import multipletests

        reject, p_adj, _, _ = multipletests(p[finite], method="fdr_bh")
        p_out = p.copy()
        idx = np.where(finite)[0]
        for i, j in enumerate(idx):
            p_out[j] = float(p_adj[i])
        return list(p_out)
    except Exception:
        # Implementação manual de BH
        m = int(finite.sum())
        idx_sorted = np.argsort(p[finite])
        p_sorted = p[finite][idx_sorted]
        p_adj_sorted = np.minimum.accumulate(
            (p_sorted * m / (np.arange(m) + 1))[::-1]
        )[::-1]
        p_adj_sorted = np.minimum(p_adj_sorted, 1.0)
        p_out = p.copy()
        finite_idx = np.where(finite)[0]
        for i, j in enumerate(finite_idx[idx_sorted]):
            p_out[j] = float(p_adj_sorted[i])
        return list(p_out)


def run_resilient_profile(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "resilient_profile"
    out_figures.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    targets = _select_targets(df)
    if not targets:
        from src.target_builder import build_target_definitions

        targets_int, _ = build_target_definitions(df)
        targets = {k: pd.Series(v, index=df.index) for k, v in targets_int.items()}

    y_key = cfg.get("analysis", {}).get("target_profile_key", "A")
    if y_key not in targets:
        y_key = sorted(targets.keys())[0]

    y = pd.Series(targets[y_key], index=df.index)

    # features: numeric columns excluding obvious target/ids/weights
    feature_cols = []
    for c in df.columns:
        lc = c.lower()
        if any(s in lc for s in ["target_a", "target_b", "target_c", "target_d"]):
            continue
        if "target_" in lc or lc.startswith("target"):
            continue
        if lc.startswith("w_") or "weight" in lc:
            continue
        if lc.startswith("st") or "cntstu" in lc or lc.endswith("id"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    # ── CC-5: Teste Mann-Whitney por variável + effect size ──
    rows_raw: list[tuple[str, float, float, float, float, float, float]] = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        x_pos = x[y == 1]
        x_neg = x[y == 0]
        if x_pos.dropna().empty or x_neg.dropna().empty:
            continue
        diff = float(x_pos.mean() - x_neg.mean())
        u_stat, p_val, r = _mannwhitney_stat(x_pos, x_neg)
        rows_raw.append((c, diff, float(x_pos.mean()), float(x_neg.mean()), u_stat, p_val, r))

    # Ordenar por magnitude do effect size
    rows_raw.sort(key=lambda t: abs(t[6]) if not np.isnan(t[6]) else abs(t[1]) / (abs(t[1]) + 1), reverse=True)

    # ── CC-5: Correção FDR de Benjamini-Hochberg ──
    p_values_raw = [r[5] for r in rows_raw]
    p_adj_list = _fdr_correction(p_values_raw)

    rows: list[dict] = []
    for i, (c, diff, m_pos, m_neg, u, p_raw, r) in enumerate(rows_raw):
        rows.append({
            "feature": c,
            "mean_resilientes": m_pos,
            "mean_nao_resilientes": m_neg,
            "diff_means": diff,
            "mann_whitney_U": u,
            "p_value": p_raw,
            "p_adjusted_fdr": p_adj_list[i],
            "effect_size_r": r,
            "significativo_fdr05": (p_adj_list[i] < 0.05) if np.isfinite(p_adj_list[i]) else False,
        })

    top = rows[:30]
    result_df = pd.DataFrame(rows)

    out_csv = out_tables / "resilient_profile.csv"
    result_df.to_csv(out_csv, index=False)

    # Resumo de significância
    n_sig = int(result_df["significativo_fdr05"].sum()) if "significativo_fdr05" in result_df.columns else 0
    n_tested = len(result_df)

    # radar/heatmap — best-effort
    try:
        import matplotlib.pyplot as plt

        top_sig = [r for r in rows if r.get("significativo_fdr05", False)][:10]
        if not top_sig:
            top_sig = rows[:10]
        labels = [r["feature"] for r in top_sig]
        vals = np.array([r["diff_means"] for r in top_sig], dtype=float)

        if np.nanmax(np.abs(vals)) != 0:
            vals = vals / np.nanmax(np.abs(vals))

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        vals_plot = np.concatenate([vals, vals[:1]])

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, vals_plot)
        ax.fill(angles, vals_plot, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"Resilient profile radar — target {y_key} (p_adj < 0.05)")
        plt.tight_layout()
        plt.savefig(out_figures / "resilient_profile_radar.png", dpi=200)
        plt.close()
    except Exception:
        pass

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        top_feats = [r["feature"] for r in rows[:20]]
        mat = []
        for group_label, mask in [("resilientes", y == 1), ("nao_resilientes", y == 0)]:
            vals_row = []
            for c in top_feats:
                vals_row.append(float(pd.to_numeric(df.loc[mask, c], errors="coerce").mean()))
            mat.append(vals_row)

        plt.figure(figsize=(12, 3.5))
        sns.heatmap(
            pd.DataFrame(mat, index=["resilientes", "nao_resilientes"], columns=top_feats),
            cmap="RdBu_r",
            center=0,
        )
        plt.title(f"Resilient profile heatmap — target {y_key}")
        plt.tight_layout()
        plt.savefig(out_figures / "resilient_profile_heatmap.png", dpi=200)
        plt.close()
    except Exception:
        pass

    # markdown report
    md = []
    md.append(f"# Resilient profile\n\n")
    md.append(f"- Dataset: `{csv_path.name}`\n")
    md.append(f"- Target usada para perfil: `{y_key}`\n")
    md.append(f"- N resilientes: {n_pos:,} | N não-resilientes: {n_neg:,}\n")
    md.append(f"- Variáveis testadas: {n_tested} | Significativas (FDR < 0.05): **{n_sig}**\n")
    md.append("\n## Nota metodológica (CC-5)\n")
    md.append(
        "As diferenças de médias são acompanhadas de teste **Mann-Whitney U** "
        "(não paramétrico, sem assumir normalidade) e effect size **r = Z/√N** (Cohen 1988). "
        "Os p-valores foram corrigidos para comparações múltiplas pelo método "
        "**Benjamini-Hochberg FDR** (α = 0.05). Apenas variáveis com `p_adjusted_fdr < 0.05` "
        "devem ser interpretadas como evidência de diferença real entre grupos.\n\n"
    )
    md.append("## Top diferenças (ordenadas por |effect_size_r|)\n\n")
    display_cols = [
        "feature", "mean_resilientes", "mean_nao_resilientes",
        "diff_means", "p_value", "p_adjusted_fdr", "effect_size_r", "significativo_fdr05",
    ]
    md.append(df_to_markdown(pd.DataFrame(top)[display_cols]))
    md.append("\n")
    md.append("\n## Figuras\n\n")
    md.append(f"- Radar: `outputs/figures/resilient_profile/resilient_profile_radar.png`\n")
    md.append(f"- Heatmap: `outputs/figures/resilient_profile/resilient_profile_heatmap.png`\n")

    (out_reports / "resilient_profile.md").write_text("".join(md), encoding="utf-8")
