from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.markdown import df_to_markdown


def make_data_dictionary(df: pd.DataFrame, out_path: Path, max_rows: int | None = None) -> None:
    rows = []
    for c in df.columns:
        s = df[c]
        missing = float(s.isna().mean())
        dtype = str(s.dtype)
        nunique = int(s.dropna().nunique())

        # categoria simples por nome
        lc = c.lower()
        if "id" in lc or "cntstu" in lc or lc.startswith("st"):
            cat = "ID"
        elif lc.startswith("w_") or "weight" in lc:
            cat = "Peso"
        elif lc.startswith("crt") or "creative" in lc or "resili" in lc:
            cat = "Resiliência/Criatividade"
        elif "escs" in lc or "homepos" in lc:
            cat = "Socioeconômica"
        else:
            cat = "Geral"

        rows.append({
            "variavel": c,
            "tipo_pandas": dtype,
            "missing_ratio": missing,
            "cardinalidade_na": nunique,
            "categoria_nome": cat,
        })

    dd = pd.DataFrame(rows).sort_values("missing_ratio", ascending=False)
    if max_rows is not None:
        dd = dd.head(max_rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(df_to_markdown(dd), encoding="utf-8")
