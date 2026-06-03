"""Dashboard Streamlit — PISA 2022 Creative Resilience."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Garante imports de `src` ao rodar via Streamlit
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from src.utils.paths import METADATA_PATH, PROJECT_ROOT

ROOT = PROJECT_ROOT
REPORTS = ROOT / "outputs" / "reports"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"


def _read_md(path: Path, max_chars: int = 12000) -> str:
    if not path.exists():
        return f"_(arquivo ausente: `{path.relative_to(ROOT)}`)_"
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n… _(truncado)_"
    return text


def _read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


st.set_page_config(page_title="PISA Resiliência Criativa", layout="wide")
st.title("PISA 2022 — Resiliência Criativa (Brasil)")

meta_path = METADATA_PATH
if meta_path.exists():
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st.caption(meta.get("project", "PISA Creative Resilience"))
    if meta.get("dataset", {}).get("csv_detected"):
        st.info(f"Dataset: `{meta['dataset']['csv_detected']}`")

tab_overview, tab_targets, tab_model, tab_reports = st.tabs(
    ["Visão geral", "Targets", "Modelagem", "Relatórios"]
)

with tab_overview:
    st.subheader("Auditoria e EDA")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(_read_md(REPORTS / "data_audit.md", 4000))
    with c2:
        st.markdown(_read_md(REPORTS / "eda_report.md", 4000))
    leak = _read_csv(TABLES / "leakage_candidates.csv")
    if leak is not None:
        st.subheader("Leakage (top 15)")
        st.dataframe(leak.head(15), use_container_width=True)

with tab_targets:
    prev = _read_csv(TABLES / "targets_definitions_prevalence.csv")
    if prev is not None:
        st.dataframe(prev, use_container_width=True)
    st.markdown(_read_md(REPORTS / "target_comparison.md", 6000))
    prof = _read_csv(TABLES / "resilient_profile.csv")
    if prof is not None:
        st.subheader("Perfil resilientes (top features)")
        st.dataframe(prof.head(20), use_container_width=True)
    radar = FIGURES / "resilient_profile" / "resilient_profile_radar.png"
    if radar.exists():
        st.image(str(radar), caption="Radar — perfil resilientes")

with tab_model:
    metrics = _read_csv(TABLES / "modeling_metrics.csv")
    if metrics is not None:
        st.dataframe(metrics, use_container_width=True)
    else:
        st.warning("Execute `python3 -m src.main --stage modeling` para gerar métricas.")
    shap_img = FIGURES / "shap" / "shap_top20.png"
    if shap_img.exists():
        st.image(str(shap_img), caption="Importância de features (SHAP / permutation)")
    fair = _read_csv(TABLES / "fairness_by_group.csv")
    if fair is not None:
        st.subheader("Fairness por grupo")
        st.dataframe(fair, use_container_width=True)

with tab_reports:
    report_files = sorted(REPORTS.glob("*.md")) if REPORTS.exists() else []
    choice = st.selectbox(
        "Relatório",
        [p.name for p in report_files] or ["(nenhum)"],
    )
    if report_files:
        st.markdown(_read_md(REPORTS / choice))

st.sidebar.header("Pipeline")
st.sidebar.code("python3 -m src.main --stage <nome>", language="bash")
stages = [
    "data_audit",
    "data_dictionary",
    "variable_discovery",
    "leakage_audit",
    "target_comparison",
    "eda",
    "resilient_profile",
    "clusterer",
    "modeling",
    "shap",
    "fairness",
    "robustness",
]
st.sidebar.markdown("\n".join(f"- `{s}`" for s in stages))

if st.sidebar.button("Abrir instruções Streamlit"):
    st.sidebar.write(f"`{sys.executable} -m streamlit run dashboard/app.py`")
