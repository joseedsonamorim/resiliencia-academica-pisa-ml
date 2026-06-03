from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class VarRecord:
    nome: str
    categoria: str
    familia: str
    regras: str


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def discover_variable_categories(df: pd.DataFrame) -> pd.DataFrame:
    # Heurísticas baseadas em prefixos e nomes usuais do PISA.
    # Esta fase é propositalmente simples: serve como data-driven starting point.
    patterns: list[tuple[str, str, list[str]]] = [
        # Criatividade / resiliência
        (r"(^|_)crt($|_)|crt_", "Criatividade", ["creative", "crt", "resili" ]),
        (r"creative|resili", "Criatividade", ["creative", "resili"]),
        (r"grupo_escs|escs", "Socioeconômicas", ["escs", "grupo_escs"]),
        (r"hom(e|)pos|homepos", "Socioeconômicas", ["homepos", "homepos"]),
        (r"escs", "Socioeconômicas", ["escs"]),
        # Contexto escolar / escola
        (r"school|tipo_de_escola|school_type|st(_|$)", "Escola", ["school", "school_type"]),
        (r"uf|regi|country|estado", "Contexto geográfico", ["uf", "regi", "região"]),
        (r"sexo|gender|st_gender|n_sexo|female|male", "Demográficas", ["sexo", "gender"]),
        # Educacionais / escolaridade / motivação
        (r"hisc_ed|hisc ed|hisc_ed|hised|wised|hisc", "Educacionais", ["hised", "wised"]),
        (r"motiv|attit|engaj", "Motivação", ["motiv", "attit"]),
        (r"ict|computer|digital", "Tecnológicas", ["ict", "digital", "computer"]),
        (r"well|wellbeing|bemestar|satisf", "Bem-estar", ["well", "bemestar", "satisf"]),
    ]

    out: list[VarRecord] = []

    for col in df.columns:
        lc = _norm(col)
        categoria = "Outras"
        familia = "Outras"
        regras: list[str] = []

        # ID / peso ficam à parte
        if "id" in lc or "cntstu" in lc or lc.startswith("st"):
            familia = "Identificadores"
            categoria = "Identificadores"
            regras.append("id heurística")
        elif lc.startswith("w_") or "weight" in lc or "_wt" in lc:
            familia = "Pesos"
            categoria = "Pesos"
            regras.append("peso heurística")
        else:
            matched = False
            for pat, fam, rs in patterns:
                if re.search(pat, lc):
                    familia = fam
                    categoria = fam
                    regras.extend(rs)
                    matched = True
                    break
            if not matched:
                # Classificação por prefixos CRxx
                if lc.startswith("cr"):
                    familia = "Criatividade/Conteúdo CR"
                    categoria = "Variável CR"
                    regras.append("prefixo cr")

        out.append(
            VarRecord(
                nome=col,
                categoria=categoria,
                familia=familia,
                regras="; ".join(sorted(set(regras))),
            )
        )


    return pd.DataFrame([r.__dict__ for r in out]).sort_values(["familia", "nome"])


def run_variable_discovery(cfg: dict, df: pd.DataFrame, out_dir: Path | None = None) -> None:
    """Fase 2: classifica variáveis por família/categoria.

    Saídas:
      - outputs/reports/variable_catalog.md
      - outputs/tables/variable_catalog.csv
    """
    _ = out_dir  # compatibilidade com chamadas antigas
    reports = Path(cfg["outputs"]["reports_dir"])
    tables = Path(cfg["outputs"]["tables_dir"])
    reports.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    catalog = discover_variable_categories(df)
    (tables / "variable_catalog.csv").write_text(catalog.to_csv(index=False), encoding="utf-8")

    top = catalog.groupby("familia").size().sort_values(ascending=False).head(30)
    md = ["# Variable Catalog (heurístico)\n\n", "## Top famílias\n", top.to_frame("qtd").to_markdown(), "\n"]
    (reports / "variable_catalog.md").write_text("".join(md), encoding="utf-8")

