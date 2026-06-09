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
        if k not in targets:
            # also allow pattern columns from target_comparison outputs
            # (common naming in some pipelines)
            pass
    return targets


def run_resilient_profile(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_figures = Path(cfg["outputs"]["figures_dir"]) / "resilient_profile"
    out_figures.mkdir(parents=True, exist_ok=True)
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    targets = _select_targets(df)
    if not targets:
        # fallback: if no target columns exist, try compute from target_builder parameters
        from src.target_builder import build_target_definitions

        targets_int, _ = build_target_definitions(df)
        targets = {k: pd.Series(v, index=df.index) for k, v in targets_int.items()}

    # Heurística: pick one target definition to profile (A by default)
    y_key = cfg.get("analysis", {}).get("target_profile_key", "A")
    if y_key not in targets:
        y_key = sorted(targets.keys())[0]

    y = pd.Series(targets[y_key], index=df.index)

    # features: numeric columns excluding obvious target/ids/weights
    drop_sub = {"id", "cntstu", "stuid", "weight", "w_fstuwt", "w_", "target_", "status"}
    feature_cols = []
    for c in df.columns:
        lc = c.lower()
        if any(s in lc for s in ["target_a", "target_b", "target_c", "target_d"]):
            continue
        if "target_" in lc or lc.startswith("target"):
            continue
        if lc in drop_sub or lc.startswith("w_") or "weight" in lc:
            continue
        if lc.startswith("st") or "cntstu" in lc or "id" in lc:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    # compare means for top N differences
    diffs = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        x_pos = x[y == 1]
        x_neg = x[y == 0]
        if x_pos.dropna().empty or x_neg.dropna().empty:
            continue
        diff = float(x_pos.mean() - x_neg.mean())
        diffs.append((c, diff, float(x_pos.mean()), float(x_neg.mean())))

    diffs.sort(key=lambda t: abs(t[1]), reverse=True)
    top = diffs[:30]

    # table CSV
    rows = []
    for c, diff, m_pos, m_neg in top:
        rows.append(
            {"feature": c, "mean_resilientes": m_pos, "mean_nao_resilientes": m_neg, "diff": diff}
        )

    out_csv = out_tables / "resilient_profile.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # radar/heatmap are best-effort
    try:
        import matplotlib.pyplot as plt

        labels = [r["feature"] for r in rows[:10]]
        vals = [r["diff"] for r in rows[:10]]

        # normalize for radar-like plot
        v = np.array(vals, dtype=float)
        if np.nanmax(np.abs(v)) != 0:
            v = v / np.nanmax(np.abs(v))

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        v = np.concatenate([v, v[:1]])

        plt.figure(figsize=(7, 6))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, v)
        ax.fill(angles, v, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"Resilient profile radar — target {y_key}")
        plt.tight_layout()
        plt.savefig(out_figures / "resilient_profile_radar.png", dpi=200)
        plt.close()
    except Exception:
        pass

    try:
        # heatmap of means for top 20 features
        top_feats = [r["feature"] for r in rows[:20]]
        mat = []
        for group_label, mask in [("resilientes", y == 1), ("nao_resilientes", y == 0)]:
            vals = []
            for c in top_feats:
                vals.append(float(pd.to_numeric(df.loc[mask, c], errors="coerce").mean()))
            mat.append(vals)

        import seaborn as sns

        plt.figure(figsize=(12, 3.5))
        sns.heatmap(pd.DataFrame(mat, index=["resilientes", "nao_resilientes"], columns=top_feats), cmap="RdBu_r", center=0)
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
    md.append("\n## Top diferenças (média) — resilient vs não\n\n")
    md.append(df_to_markdown(pd.DataFrame(rows).head(25)))
    md.append("\n")

    md.append("\n## Figuras\n\n")
    md.append(f"- Radar: `outputs/figures/resilient_profile/resilient_profile_radar.png`\n")
    md.append(f"- Heatmap: `outputs/figures/resilient_profile/resilient_profile_heatmap.png`\n")

    (out_reports / "resilient_profile.md").write_text("".join(md), encoding="utf-8")

