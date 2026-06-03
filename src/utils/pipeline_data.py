from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.target_builder import build_target_definitions

# Colunas nunca usadas como features (proxy de target, IDs, pesos).
_HARD_EXCLUDE_SUBSTR = (
    "cntstu",
    "stuid",
    "target_",
    "creative_resilience",
    "crt_score",
    "grupo_escs",
)
_HARD_EXCLUDE_EXACT = {"w_fstuwt"}


def load_dataframe(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def merge_saved_targets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    targets_path = Path(cfg["outputs"]["tables_dir"]) / "targets_definitions.csv"
    if not targets_path.exists():
        return df
    tdf = pd.read_csv(targets_path)
    if len(tdf) != len(df):
        return df
    out = df.copy()
    for c in tdf.columns:
        out[c] = tdf[c].values
    return out


def get_target_key(cfg: dict) -> str:
    return str(cfg.get("analysis", {}).get("target_profile_key", "A"))


def build_targets_dict(df: pd.DataFrame) -> dict[str, pd.Series]:
    targets: dict[str, pd.Series] = {}
    for k in ["A", "B", "C", "D"]:
        col = f"target_{k}"
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0)
            targets[k] = (s > 0).astype(int)
    if targets:
        return targets
    built, _ = build_target_definitions(df)
    return {k: pd.Series(v, index=df.index) for k, v in built.items()}


def get_target_series(df: pd.DataFrame, cfg: dict) -> tuple[pd.Series, str]:
    targets = build_targets_dict(df)
    key = get_target_key(cfg)
    if key not in targets:
        key = sorted(targets.keys())[0]
    return pd.Series(targets[key], index=df.index), key


def load_leakage_exclude(cfg: dict) -> set[str]:
    excluded: set[str] = set()
    for name in [
        "CRT_SCORE",
        "Creative_Resilience",
        "CRT",
        "ESCS",
        "Grupo_ESCS",
        "CNTSTUID",
        "W_FSTUWT",
    ]:
        excluded.add(name)

    cand_path = Path(cfg["outputs"]["tables_dir"]) / "leakage_candidates.csv"
    if cand_path.exists():
        leak = pd.read_csv(cand_path)
        if "nome" in leak.columns:
            high = leak["severity"].isin(["CRÍTICO", "ALTO"]) if "severity" in leak.columns else False
            corr = leak["score_corr_abs"] >= 0.95 if "score_corr_abs" in leak.columns else False
            mask = high | corr
            excluded.update(leak.loc[mask, "nome"].astype(str).tolist())

    return excluded


def select_feature_columns(
    df: pd.DataFrame,
    cfg: dict,
    *,
    max_features: int | None = None,
) -> list[str]:
    excluded = load_leakage_exclude(cfg)
    max_features = max_features or int(cfg.get("modeling", {}).get("max_features", 80))
    max_missing = float(cfg.get("modeling", {}).get("max_missing_ratio", 0.5))

    candidates: list[tuple[str, float, float]] = []
    for c in df.columns:
        if c in excluded:
            continue
        lc = c.lower()
        if lc in _HARD_EXCLUDE_EXACT:
            continue
        if any(s in lc for s in _HARD_EXCLUDE_SUBSTR):
            continue
        if lc.startswith("w_") or "weight" in lc:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        miss = float(df[c].isna().mean())
        if miss > max_missing:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.nunique(dropna=True) <= 1:
            continue
        var = float(s.var(ddof=0))
        if var != var or var == 0:  # NaN or zero variance
            continue
        candidates.append((c, var, miss))

    candidates.sort(key=lambda t: t[1], reverse=True)
    return [c for c, _, _ in candidates[:max_features]]


def get_sample_weights(df: pd.DataFrame, cfg: dict) -> pd.Series | None:
    for name in cfg.get("weighted_analysis", {}).get("sample_weight_candidates", ["W_FSTUWT"]):
        if name in df.columns:
            w = pd.to_numeric(df[name], errors="coerce")
            if w.notna().sum() > 0:
                return w.fillna(w.median())
    return None


def find_sensitive_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    found: list[str] = []
    for cand in cfg.get("fairness", {}).get("sensitive_feature_candidates", []):
        if cand in df.columns:
            found.append(cand)
    return found
