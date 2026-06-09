from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.markdown import df_to_markdown


@dataclass
class LeakageCandidate:
    nome: str
    score_corr_abs: float
    corr_sign: float
    spearman_abs: float
    n_unique: int
    missing_ratio: float
    reason: str


def _safe_to_numeric(s: pd.Series) -> pd.Series:
    # tries to coerce; keeps non-convertible as NaN
    return pd.to_numeric(s, errors="coerce")


def _rank_corr_abs(x: pd.Series, y: pd.Series) -> float:
    # Spearman as rank-based proxy
    try:
        return float(x.corr(y, method="spearman").__abs__())
    except Exception:
        return float("nan")


def find_leakage_candidates(
    df: pd.DataFrame,
    target_proxy_col: str,
    candidate_cols: Iterable[str],
) -> pd.DataFrame:
    if target_proxy_col not in df.columns:
        raise KeyError(f"Target proxy col '{target_proxy_col}' not found in dataframe")

    y = _safe_to_numeric(df[target_proxy_col])

    out: list[LeakageCandidate] = []

    for c in candidate_cols:
        if c == target_proxy_col:
            continue

        if c not in df.columns:
            continue

        x = _safe_to_numeric(df[c])

        mask = x.notna() & y.notna()
        if mask.sum() < 30:
            continue

        x2 = x[mask]
        y2 = y[mask]

        try:
            corr = float(np.corrcoef(x2, y2)[0, 1])
        except Exception:
            corr = float("nan")

        spearman_abs = _rank_corr_abs(x2, y2)

        reason = []
        if np.isfinite(corr) and abs(corr) >= 0.95:
            reason.append("corr_abs>=0.95")
        if "status" in c.lower() or "grupo_escs" in c.lower():
            reason.append("name contains known target-proxy families")
        if "creative" in c.lower() or "resili" in c.lower():
            reason.append("name contains creative/resilience")
        if "crt" in c.lower():
            reason.append("name contains crt")

        if not reason:
            # still keep but reason as generic high corr based on spearman
            if np.isfinite(spearman_abs) and spearman_abs >= 0.9:
                reason.append("spearman_abs>=0.90")
            else:
                # keep only if it looks related by magnitude
                if not (np.isfinite(corr) and abs(corr) >= 0.7):
                    continue
                reason.append("corr_abs>=0.70")

        out.append(
            LeakageCandidate(
                nome=c,
                score_corr_abs=float(abs(corr)) if np.isfinite(corr) else float("nan"),
                corr_sign=float(corr) if np.isfinite(corr) else float("nan"),
                spearman_abs=float(spearman_abs) if np.isfinite(spearman_abs) else float("nan"),
                n_unique=int(df[c].nunique(dropna=True)),
                missing_ratio=float(df[c].isna().mean()),
                reason=";".join(reason) if reason else "",
            )
        )

    res = pd.DataFrame([o.__dict__ for o in out])
    if not res.empty:
        res = res.sort_values(["score_corr_abs", "spearman_abs"], ascending=False)
    return res


def run_leakage_audit(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    # Este detector é um gate inicial (proxy-based). A versão científica completa virá nas fases seguintes.
    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    # target proxy: CRT_SCORE se existir, senão Creative_Resilience
    proxy = None
    for cand in ["CRT_SCORE", "Creative_Resilience", "CRT"]:
        if cand in df.columns:
            proxy = cand
            break

    if proxy is None:
        raise KeyError("Não foi possível encontrar colunas de proxy de target (CRT_SCORE/Creative_Resilience/CRT)")

    # candidatos: todas menos IDs/weights
    drop = set()
    for c in df.columns:
        lc = c.lower()
        if "weight" in lc or lc.startswith("w_"):
            drop.add(c)
        if "id" in lc or "cntstu" in lc or lc.startswith("st"):
            drop.add(c)

    candidate_cols = [c for c in df.columns if c not in drop]

    res = find_leakage_candidates(df, proxy, candidate_cols)

    # heurística de severidade
    critical_corr = float(cfg["analysis"]["leakage_severity_thresholds"]["high_corr_abs"])
    res["severity"] = "MÉDIO"
    res.loc[res["score_corr_abs"] >= critical_corr, "severity"] = "ALTO"
    res.loc[res["nome"].str.contains(r"crt_score|creative_resilience|grupo_escs|status", case=False, na=False), "severity"] = "CRÍTICO"

    # salvar
    audit_path = out_reports / "leakage_audit.md"
    cand_csv = out_tables / "leakage_candidates.csv"
    cand_md = ""
    if not res.empty:
        cand_md = df_to_markdown(res.head(50))

    audit_md = []
    audit_md.append("# Leakage Audit (gate inicial)\n\n")
    audit_md.append(f"- Proxy de target usada: **{proxy}**\n")
    audit_md.append(f"- Dataset: {csv_path.name}\n")
    audit_md.append("\n## Top leakage candidates (top 50)\n")
    audit_md.append(cand_md if cand_md else "(nenhum candidato encontrado)\n")

    audit_path.write_text("".join(audit_md), encoding="utf-8")
    res.to_csv(cand_csv, index=False)

    # Gate rule: falhar somente se houver candidatos que pareçam DERIVADOS direta/indiretamente do proxy do target.
    # Observação: em datasets já tratados, podem existir colunas-alvo explícitas (p.ex. Creative_Resilience)
    # que são inevitáveis para o objetivo (mas não devem ser usadas como features). Por isso não falhamos só por match de nome.
    fail_mask = (res["severity"] == "CRÍTICO")
    fail_mask &= ~res["nome"].str.contains("creative_resilience|grupo_escs", case=False, na=False)

    crit = res[fail_mask]
    if not crit.empty:
        raise RuntimeError(
            "Leakage gate falhou: variáveis CRÍTICAS encontradas (não previstas/inerentes ao target). "
            + ", ".join(crit["nome"].head(20).tolist())
        )

