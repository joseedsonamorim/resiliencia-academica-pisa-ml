from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame, *, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```csv\n" + df.to_csv(index=index) + "```"
