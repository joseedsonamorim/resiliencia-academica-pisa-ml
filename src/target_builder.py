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


def build_target_definitions(df: pd.DataFrame) -> dict[str, pd.Series]:
    # ESCS proxy
    escs_col = _find_first_col(df, ["ESCS", "Grupo_ESCS", "HISCED", "homepos", "HOMEPOS"])

    # CRT proxy
    crt_col = _find_first_col(df, ["CRT_SCORE", "Creative_Resilience", "CRT"])

    es = _safe_numeric(df[escs_col])
    cr = _safe_numeric(df[crt_col])

    # quantis
    q1 = float(es.quantile(0.25))
    p30 = float(es.quantile(0.30))
    q2 = float(es.quantile(0.20))  # fallback (unused)

    q3 = float(cr.quantile(0.75))
    p70 = float(cr.quantile(0.70))
    p90 = float(cr.quantile(0.90))

    # Definições (A/B/C) via quantis
    target_a = (es <= q1) & (cr >= q3)
    target_b = (es <= p30) & (cr >= p70)
    target_c = (es <= q1) & (cr >= p90)

    # D: exploratória/cluster-based simplificada via quartis de uma score composta
    # (evita KMeans nesta fase inicial; será substituída no pipeline completo)
    # score = padronização de CRT - padronização de ESCS
    es_z = (es - es.mean()) / (es.std(ddof=0) if es.std(ddof=0) != 0 else 1)
    cr_z = (cr - cr.mean()) / (cr.std(ddof=0) if cr.std(ddof=0) != 0 else 1)
    score = cr_z - es_z
    score_q70 = float(score.quantile(0.70))
    target_d = (score >= score_q70)

    # garantir dtype e nome
    out = {
        "A": target_a.astype(int),
        "B": target_b.astype(int),
        "C": target_c.astype(int),
        "D": target_d.astype(int),
    }
    return out, {"escs_col": escs_col, "crt_col": crt_col, "q1": q1, "p30": p30, "q3": q3, "p70": p70, "p90": p90}


def compare_targets(df: pd.DataFrame, targets: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, y in targets.items():
        y = pd.Series(y)
        prevalence = float(y.mean())
        n_pos = int(y.sum())
        n = int(y.shape[0])
        rows.append({"target_def": name, "n": n, "n_pos": n_pos, "prevalence": prevalence})
    return pd.DataFrame(rows).sort_values("target_def")


def run_target_comparison(cfg: dict, df: pd.DataFrame, csv_path: Path) -> None:
    out_reports = Path(cfg["outputs"]["reports_dir"])
    out_tables = Path(cfg["outputs"]["tables_dir"])
    out_reports.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    targets, params = build_target_definitions(df)

    comp = compare_targets(df, targets)
    (out_tables / "targets_definitions_prevalence.csv").write_text(comp.to_csv(index=False), encoding="utf-8")

    # salvar targets combinados
    targets_df = pd.DataFrame({f"target_{k}": v for k, v in targets.items()})
    (out_tables / "targets_definitions.csv").write_text(targets_df.to_csv(index=False), encoding="utf-8")

    # markdown
    header = [
        "# Target comparison (A/B/C/D)\n\n",
        f"- Dataset: {csv_path.name}\n",
        f"- ESCS proxy: **{params['escs_col']}**\n",
        f"- CRT proxy: **{params['crt_col']}**\n",
        "\n## Estatísticas\n",
    ]

    comp_md = df_to_markdown(comp)

    notes = [
        "\n\n## Notas das definições\n",
        f"A: ESCS <= Q1 (Q0.25={params['q1']:.4f}) e CRT >= Q3 (Q0.75={params['q3']:.4f})\n",
        f"B: ESCS <= P30 (P0.30={params['p30']:.4f}) e CRT >= P70 (P0.70={params['p70']:.4f})\n",
        f"C: ESCS <= Q1 (Q0.25={params['q1']:.4f}) e CRT >= P90 (P0.90={params['p90']:.4f})\n",
        "D: score composto (CRT z - ESCS z) com corte em P70\n",
    ]

    out_path = out_reports / "target_comparison.md"
    out_path.write_text("".join(header) + comp_md + "".join(notes), encoding="utf-8")
