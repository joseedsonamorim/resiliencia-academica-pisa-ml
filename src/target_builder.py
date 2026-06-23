from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.markdown import df_to_markdown

@dataclass
class TargetDef:
    name: str
    escs_quantile: float
    crt_quantile: float

def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Nenhuma das colunas {candidates} foi encontrada no dataset")

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _weighted_quantile(values: pd.Series, quantile: float, weights: pd.Series | None = None) -> float:
    if weights is None or weights.isna().all():
        return float(values.quantile(quantile))
    mask = values.notna() & weights.notna() & (weights > 0)
    v = values[mask].to_numpy(dtype=float)
    w = weights[mask].to_numpy(dtype=float)
    if len(v) == 0:
        return float(values.quantile(quantile))
    sorter = np.argsort(v, kind="stable")
    v = v[sorter]
    w = w[sorter]
    cumw = np.cumsum(w)
    cumw = cumw / cumw[-1]
    return float(np.interp(quantile, cumw, v))

def compute_target_thresholds(df_train: pd.DataFrame, weights: pd.Series | None = None) -> dict:
    escs_col = _find_first_col(df_train, ["ESCS", "Grupo_ESCS", "HISCED", "homepos", "HOMEPOS"])
    
    # 1A Architecture: Support 10 Plausible Values
    pv_cols = [f"PV{i}CRTH_NC" for i in range(1, 11)]
    valid_pv_cols = [c for c in pv_cols if c in df_train.columns]
    
    # Se não achar os PVs, tenta buscar o genérico
    if not valid_pv_cols:
        crt_col = _find_first_col(df_train, ["CRT_SCORE", "Creative_Resilience", "CRT"])
        valid_pv_cols = [crt_col]

    es = _safe_numeric(df_train[escs_col])
    w_train: pd.Series | None = None
    if weights is not None:
        w_train = weights.loc[df_train.index] if hasattr(weights, "loc") else weights
        w_train = pd.to_numeric(w_train, errors="coerce")
        if w_train.notna().sum() == 0:
            w_train = None

    q1_escs = _weighted_quantile(es, 0.25, w_train)
    p30_escs = _weighted_quantile(es, 0.30, w_train)
    es_mean = float(es.mean())
    es_std = float(es.std(ddof=0)) or 1.0

    pv_params = {}
    for pv_col in valid_pv_cols:
        cr = _safe_numeric(df_train[pv_col])
        q3 = _weighted_quantile(cr, 0.75, w_train)
        p70 = _weighted_quantile(cr, 0.70, w_train)
        p90 = _weighted_quantile(cr, 0.90, w_train)
        
        cr_mean = float(cr.mean())
        cr_std = float(cr.std(ddof=0)) or 1.0
        
        es_z_train = (es - es_mean) / es_std
        cr_z_train = (cr - cr_mean) / cr_std
        score_train = cr_z_train - es_z_train
        score_q70 = _weighted_quantile(score_train, 0.70, w_train)
        
        pv_params[pv_col] = {
            "q3": q3, "p70": p70, "p90": p90,
            "cr_mean": cr_mean, "cr_std": cr_std,
            "score_q70": score_q70
        }

    return {
        "escs_col": escs_col,
        "valid_pv_cols": valid_pv_cols,
        "q1_escs": q1_escs,
        "p30_escs": p30_escs,
        "es_mean": es_mean,
        "es_std": es_std,
        "pv_params": pv_params,
        "n_train": int(df_train.shape[0]),
        "weighted": w_train is not None,
    }


def apply_target_definitions(df: pd.DataFrame, params: dict) -> dict[str, pd.DataFrame]:
    escs_col = params["escs_col"]
    es = _safe_numeric(df[escs_col]) if escs_col in df.columns else pd.Series(np.nan, index=df.index)
    
    es_z = (es - params["es_mean"]) / params["es_std"]
    
    targets_by_def = {"A": {}, "B": {}, "C": {}, "D": {}}
    
    for pv_col in params["valid_pv_cols"]:
        cr = _safe_numeric(df[pv_col]) if pv_col in df.columns else pd.Series(np.nan, index=df.index)
        p = params["pv_params"][pv_col]
        
        target_a = ((es <= params["q1_escs"]) & (cr >= p["q3"])).astype(int)
        target_b = ((es <= params["p30_escs"]) & (cr >= p["p70"])).astype(int)
        target_c = ((es <= params["q1_escs"]) & (cr >= p["p90"])).astype(int)
        
        cr_z = (cr - p["cr_mean"]) / p["cr_std"]
        score = cr_z - es_z
        target_d = (score >= p["score_q70"]).astype(int)
        
        targets_by_def["A"][pv_col] = target_a
        targets_by_def["B"][pv_col] = target_b
        targets_by_def["C"][pv_col] = target_c
        targets_by_def["D"][pv_col] = target_d

    # Retorna DataFrame para cada definição (cada coluna é um PV)
    # Se houver apenas 1 PV (fallback), é igual, mas numa DF com 1 coluna
    return {
        "A": pd.DataFrame(targets_by_def["A"]),
        "B": pd.DataFrame(targets_by_def["B"]),
        "C": pd.DataFrame(targets_by_def["C"]),
        "D": pd.DataFrame(targets_by_def["D"]),
    }


def build_target_definitions(df: pd.DataFrame, df_train: pd.DataFrame | None = None, weights: pd.Series | None = None) -> tuple[dict[str, pd.DataFrame], dict]:
    ref = df_train if df_train is not None else df
    params = compute_target_thresholds(ref, weights)
    targets = apply_target_definitions(df, params)
    return targets, params


def compare_targets(df: pd.DataFrame, targets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df_pv in targets.items():
        # Pooling over PVs for prevalence
        mean_prevalence = df_pv.mean().mean()
        mean_n_pos = df_pv.sum().mean()
        n = int(df_pv.shape[0])
        rows.append({
            "target_def": name, 
            "n": n, 
            "n_pos": round(mean_n_pos, 1), 
            "prevalence": mean_prevalence
        })
    return pd.DataFrame(rows).sort_values("target_def")


def run_target_comparison(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    weights: pd.Series | None = None
    for wname in cfg.get("weighted_analysis", {}).get("sample_weight_candidates", ["W_FSTUWT"]):
        if wname in df.columns:
            w = pd.to_numeric(df[wname], errors="coerce")
            if w.notna().sum() > 0:
                weights = w.where(w > 0).fillna(w.median())
                break

    targets, params = build_target_definitions(df, weights=weights)

    comp = compare_targets(df, targets)
    (out_tables / "targets_definitions_prevalence.csv").write_text(comp.to_csv(index=False), encoding="utf-8")

    # Save canonical PV average target for compatibility in other simple scripts if needed
    targets_df = pd.DataFrame({f"target_{k}": v.mean(axis=1) >= 0.5 for k, v in targets.items()}).astype(int)
    (out_tables / "targets_definitions.csv").write_text(targets_df.to_csv(index=False), encoding="utf-8")

    header = [
        "# Target comparison (A/B/C/D) - Plausible Values Pooled\n\n",
        f"- Dataset: {csv_path.name}\n",
        f"- ESCS proxy: **{params['escs_col']}**\n",
        f"- CRT proxy: **{len(params['valid_pv_cols'])} Plausible Values (Rubin's Rules)**\n",
        f"- Pesos amostrais (W_FSTUWT): **{'sim' if params['weighted'] else 'não'}**\n",
        f"- N referência: {params['n_train']:,} estudantes\n",
        "\n## Estatísticas de prevalência (Média dos PVs)\n",
    ]

    comp_md = df_to_markdown(comp)

    notes = [
        "\n\n## Nota metodológica 1A\n",
        "A resiliência foi computada independentemente para cada um dos 10 *Plausible Values* da OCDE. ",
        "O limiar (P75, P90, etc) é recalculado 10 vezes. A estatística final apresentada é o "
        "pooled average das 10 definições. No treinamento de Machine Learning, os modelos devem prever e combinar "
        "esses 10 vetores de resposta separadamente.\n"
    ]

    out_path = out_reports / "target_comparison.md"
    out_path.write_text("".join(header) + comp_md + "".join(notes), encoding="utf-8")
