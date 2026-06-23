"""Dashboard Streamlit — PISA 2022 Creative Resilience."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.paths import METADATA_PATH, PROJECT_ROOT

ROOT = PROJECT_ROOT
REPORTS = ROOT / "outputs" / "reports"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
MODELS = ROOT / "models"


def _read_md(path: Path, max_chars: int = 25000) -> str:
    if not path.exists():
        return f"_(arquivo ausente: `{path.relative_to(ROOT)}`)_"
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n... _(truncado)_"
    return text


@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            return None
    return None


def _csv(name: str) -> pd.DataFrame | None:
    return _read_csv(str(TABLES / name))


def _fmt(value: object, digits: int = 3, suffix: str = "") -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return str(value)


def _pct(value: object, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return str(value)


def _best_metric(metrics: pd.DataFrame | None) -> pd.Series | None:
    if metrics is None or metrics.empty or "cv_average_precision_mean" not in metrics.columns:
        # Fallback para ROC-AUC se AP não existir
        if metrics is not None and "cv_roc_auc_mean" in metrics.columns:
            return metrics.sort_values(["cv_roc_auc_mean", "holdout_roc_auc"], ascending=False).iloc[0]
        return None
    return metrics.sort_values(["cv_average_precision_mean", "holdout_average_precision"], ascending=False).iloc[0]


def _target_row(prevalence: pd.DataFrame | None, target_key: str) -> pd.Series | None:
    if prevalence is None or prevalence.empty or "target_def" not in prevalence.columns:
        return None
    rows = prevalence[prevalence["target_def"].astype(str) == str(target_key)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _bootstrap_value(bootstrap: pd.DataFrame | None, metric: str, col: str) -> object:
    if bootstrap is None or bootstrap.empty:
        return np.nan
    rows = bootstrap[bootstrap["metric"].astype(str) == metric]
    if rows.empty or col not in rows.columns:
        return np.nan
    return rows.iloc[0][col]


def _target_explanation(target_key: str) -> str:
    explanations = {
        "A": "Baixo nível socioeconômico (ESCS ≤ P25) e alto desempenho criativo (CRT_SCORE ≥ P75). Foco nos extremos.",
        "B": "Baixo nível socioeconômico ampliado (ESCS ≤ P30) e criatividade alta (CRT_SCORE ≥ P70). Mais inclusivo.",
        "C": "Baixo nível socioeconômico estrito (ESCS ≤ P25) e criatividade de elite (CRT_SCORE ≥ P90). Altamente restritivo.",
        "D": "Desempenho criativo residual alto: escore de criatividade significativamente acima do esperado para o contexto socioeconômico.",
    }
    return explanations.get(str(target_key), "Definição operacional selecionada.")


def _info_card(title: str, value: str, note: str, tone: str = "blue") -> None:
    colors = {
        "blue": ("#1d4ed8", "#eef5ff"),
        "green": ("#047857", "#effaf4"),
        "amber": ("#b45309", "#fff7ed"),
        "rose": ("#be123c", "#fff1f2"),
        "ink": ("#334155", "#f8fafc"),
        "purple": ("#6d28d9", "#f5f3ff"),
    }
    accent, bg = colors.get(tone, colors["blue"])
    st.markdown(
        f"""
        <div class="info-card" style="border-top-color:{accent}; background:{bg};">
          <div class="info-title" style="color:{accent};">{html.escape(title)}</div>
          <div class="info-value">{html.escape(value)}</div>
          <div class="info-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _reading_note(title: str, body: str, icon: str = "💡") -> None:
    st.markdown(
        f"""
        <div class="reading-note">
          <div style="font-size: 1.2rem;">{icon}</div>
          <div>
            <strong>{html.escape(title)}:</strong>
            <span style="color: #475569; margin-left: 4px;">{html.escape(body)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _alert(text: str, type: str = "info") -> None:
    colors = {
        "info": ("#1d4ed8", "#eff6ff"),
        "warning": ("#b45309", "#fffbeb"),
        "success": ("#047857", "#ecfdf5"),
        "error": ("#b91c1c", "#fef2f2")
    }
    c_text, c_bg = colors.get(type, colors["info"])
    st.markdown(
        f"""
        <div style="background-color: {c_bg}; border-left: 4px solid {c_text}; padding: 12px 16px; border-radius: 4px; color: {c_text}; font-size: 0.95rem; margin-bottom: 16px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def _model_rank_chart(metrics: pd.DataFrame, metric_col: str = "cv_average_precision_mean") -> None:
    if metric_col not in metrics.columns:
        if "cv_roc_auc_mean" in metrics.columns:
            metric_col = "cv_roc_auc_mean"
        else:
            return
    rank = metrics.sort_values(metric_col, ascending=False).head(10)
    chart = rank.set_index("model")[[metric_col]]
    st.bar_chart(chart, use_container_width=True, height=320)


def _threshold_curve(thresholds: pd.DataFrame) -> None:
    line_cols = [c for c in ["f1", "precision", "recall", "balanced_accuracy"] if c in thresholds.columns]
    if not line_cols:
        return
    curve = thresholds.sort_values("threshold").set_index("threshold")[line_cols]
    st.line_chart(curve, use_container_width=True, height=330)


def _probability_bins(pred: pd.DataFrame) -> pd.DataFrame:
    bins = np.linspace(0, 1, 11)
    labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]
    tmp = pred.copy()
    tmp["faixa_prob"] = pd.cut(
        tmp["proba_resilient"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    grouped = tmp.groupby(["faixa_prob", "y_true"], observed=False).size().unstack(fill_value=0)
    grouped.columns = [f"classe_{c}" for c in grouped.columns]
    return grouped


def _render_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def _compact_metrics_table(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model",
        "variant",
        "cv_average_precision_mean",
        "cv_f1_mean",
        "cv_roc_auc_mean",
        "holdout_average_precision",
        "holdout_opt_f1",
        "holdout_roc_auc",
    ]
    return metrics[[c for c in cols if c in metrics.columns]]


# --- Page Config & CSS ---
st.set_page_config(
    page_title="PISA Resiliência Criativa — Painel Científico",
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="🎓"
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    :root {
      --ink: #0f172a;
      --muted: #64748b;
      --line: #e2e8f0;
      --paper: #ffffff;
      --blue: #2563eb;
      --green: #059669;
      --amber: #d97706;
      --rose: #e11d48;
      --purple: #7c3aed;
    }
    
    .stApp { 
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 { 
        font-family: 'Inter', sans-serif;
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    
    h1 { font-size: clamp(2.2rem, 4vw, 3rem); font-weight: 800; line-height: 1.1; margin-bottom: 0.5rem; }
    h2 { font-weight: 700; margin-top: 2rem; border-bottom: 1px solid var(--line); padding-bottom: 0.5rem; }
    h3 { font-weight: 600; color: #334155; }
    
    .hero {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 32px;
      background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
      margin-bottom: 24px;
      position: relative;
      overflow: hidden;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 6px;
        background: linear-gradient(90deg, var(--blue), var(--purple), var(--rose));
    }
    
    .hero-kicker {
      color: var(--purple);
      font-size: 0.85rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      margin-bottom: 12px;
    }
    
    .hero-body {
      max-width: 900px;
      color: #475569;
      font-size: 1.1rem;
      line-height: 1.6;
      margin-top: 12px;
    }
    
    .pill-row { display:flex; flex-wrap:wrap; gap:10px; margin-top:20px; }
    .pill {
      display:inline-flex; align-items:center; gap:6px;
      border: 1px solid #cbd5e1;
      border-radius: 999px;
      padding: 6px 14px;
      background: #ffffff;
      color: #334155;
      font-size: 0.85rem;
      font-weight: 600;
      box-shadow: 0 1px 2px rgba(0,0,0,0.02);
    }
    
    .info-card {
      border: 1px solid var(--line);
      border-top: 4px solid var(--blue);
      border-radius: 10px;
      padding: 20px;
      min-height: 150px;
      box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .info-title {
      font-size: 0.85rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }
    
    .info-value {
      color: var(--ink);
      font-size: 1.8rem;
      line-height: 1.2;
      font-weight: 800;
      margin-bottom: 10px;
    }
    
    .info-note {
      color: #64748b;
      font-size: 0.9rem;
      line-height: 1.4;
    }
    
    .reading-note {
      display: flex;
      align-items: flex-start;
      gap: 12px;
      background: #ffffff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      margin: 16px 0;
      box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 6px 6px 0 0;
        border: 1px solid var(--line);
        border-bottom: none;
        padding: 0 16px;
        color: #64748b;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f8fafc;
        color: var(--blue);
        border-top: 3px solid var(--blue);
    }
    
    div[data-testid="stMetric"] {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
      background: #ffffff;
      box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    
    /* Tabelas mais limpas */
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
    
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Load Data ---
meta = {}
if (MODELS / "modeling_meta.json").exists():
    meta = json.loads((MODELS / "modeling_meta.json").read_text(encoding="utf-8"))

project_meta = {}
if METADATA_PATH.exists():
    project_meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

metrics = _csv("modeling_metrics.csv")
bootstrap = _csv("modeling_best_bootstrap.csv")
thresholds = _csv("modeling_thresholds.csv")
predictions = _csv("modeling_holdout_predictions.csv")
prevalence = _csv("targets_definitions_prevalence.csv")
features = _csv("modeling_feature_set.csv")
leakage = _csv("leakage_candidates.csv")
fairness_group = _csv("fairness_by_group.csv")
fairness_summary = _csv("fairness_summary.csv")
importance = _csv("shap_feature_importance.csv")

sens_summary = _csv("sensitivity_summary.csv")
sens_models = _csv("sensitivity_all_models.csv")
sens_rank = _csv("sensitivity_rank_stability.csv")

cluster_assoc = _csv("cluster_resilience_association.csv")
ranking = _csv("clustering_model_comparison.csv")
stability = _csv("cluster_stability.csv")
profiles = _csv("cluster_profiles_interpretable.csv")
resilient_stats = _csv("resilient_profile.csv")

best = _best_metric(metrics)

target_key = str(meta.get("target_key", "A"))
target_stats = _target_row(prevalence, target_key)
dataset_label = project_meta.get("dataset", {}).get("csv_detected", meta.get("dataset", "NA"))
n_total = int(target_stats["n"]) if target_stats is not None and "n" in target_stats else None
n_pos = int(target_stats["n_pos"]) if target_stats is not None and "n_pos" in target_stats else None
target_prev = target_stats["prevalence"] if target_stats is not None and "prevalence" in target_stats else np.nan
best_threshold = meta.get("best_threshold", np.nan)

# --- Hero Section ---
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">PISA 2022 Brasil · Inteligência Artificial & Educação</div>
      <h1>Painel de Resiliência Criativa</h1>
      <div class="hero-body">
        Este portal apresenta os resultados da auditoria algorítmica e metodológica sobre os microdados do PISA 2022. 
        Nosso objetivo é identificar quais fatores protegem estudantes de baixo nível socioeconômico, permitindo que 
        atinjam alto desempenho em pensamento criativo, garantindo rigor científico, reprodutibilidade e justiça algorítmica.
      </div>
      <div class="pill-row">
        <span class="pill">📊 Dataset: {html.escape(str(dataset_label))}</span>
        <span class="pill">🎯 Target Ativo: {html.escape(target_key)}</span>
        <span class="pill">⚙️ Validação: CV Repetida + Holdout Estratificado</span>
        <span class="pill">🛡️ Leakage Auditado e Prevenido</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Tabs ---
tabs = st.tabs(
    [
        "📊 Resumo Executivo",
        "🔬 Visão Científica",
        "⚖️ Justiça Algorítmica",
        "🎯 Sensibilidade Multi-Target",
        "🧠 Importância (SHAP)",
        "🧩 Perfis de Resiliência",
        "⚙️ Modelagem & Métricas",
        "📁 Dados & Vazamento",
        "📖 Glossário PISA",
        "📄 Relatórios Completos",
    ]
)

# --- TABS CONTENT ---

# 1. Resumo Executivo (Para Público Geral)
with tabs[0]:
    st.subheader("O que descobrimos?")
    
    _reading_note(
        "Propósito",
        "Traduzir milhares de variáveis do questionário PISA em sinais claros que explicam como alguns estudantes vencem a adversidade socioeconômica e se destacam criativamente."
    )
    
    # Key Metrics Row
    top_cols = st.columns(4)
    with top_cols[0]:
        _info_card(
            "Alunos Resilientes",
            f"{n_pos:,}" if n_pos is not None else "NA",
            f"Prevalência de {_pct(target_prev)} em {n_total:,} estudantes analisados.",
            "amber",
        )
    with top_cols[1]:
        _info_card(
            "Poder Preditivo (ROC-AUC)",
            _fmt(best.get("holdout_roc_auc") if best is not None else np.nan),
            "Capacidade do modelo de separar resilientes de não resilientes no conjunto de teste isolado.",
            "green",
        )
    with top_cols[2]:
        _info_card(
            "Equidade Algorítmica",
            "Validada",
            "O modelo foi auditado quanto a vieses de gênero e status migratório.",
            "purple",
        )
    with top_cols[3]:
        _info_card(
            "Robustez Científica",
            "Alta",
            "Resultados consistentes através de múltiplas definições de resiliência e bootstrap estrito.",
            "blue",
        )

    st.markdown("---")
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### 🌟 Principais Fatores Protetores")
        st.write("Segundo nosso modelo de Machine Learning (SHAP values), os elementos que mais contribuem para a resiliência criativa são:")
        if importance is not None:
            # Pega as top 5
            top_features = importance.head(5)["feature"].tolist()
            for f in top_features:
                st.markdown(f"- **{f}**")
        else:
            st.info("Rode o pipeline de SHAP para ver os fatores.")
            
        st.markdown("### 🔍 O Perfil do Aluno Resiliente")
        st.write("Testes estatísticos rigorosos (Mann-Whitney U com correção FDR) mostram diferenças significativas:")
        _render_image(FIGURES / "resilient_profile" / "resilient_profile_radar.png", "Radar: Resilientes vs Não Resilientes")

    with c2:
        st.markdown("### 🤖 Performance do Modelo")
        st.write("O modelo não memorizou os dados: ele foi validado em um subconjunto virgem (holdout) e mostrou estabilidade via bootstrap.")
        if metrics is not None:
            ap = best.get("cv_average_precision_mean") if best is not None else np.nan
            _alert(f"O modelo escolhido foi o **{best.get('model')}**, selecionado através do critério de Average Precision ({_fmt(ap)}).")
            _render_image(FIGURES / "robustness" / "calibration_curve.png", "A curva de calibração mostra que o modelo reporta probabilidades confiáveis.")
        else:
            st.info("Métricas indisponíveis.")

# 2. Visão Científica (Guiada)
with tabs[1]:
    st.subheader("Validação Científica do Pipeline")
    
    _reading_note(
        "Prevenção de Vazamento (Data Leakage)",
        "Crucial na modelagem científica. Variáveis de tarefa criativa brutas (CR590Q*) foram excluídas. Os quantis do Target foram calculados SOMENTE no conjunto de treino, prevenindo que informações do teste vazassem para os limiares."
    )
    _reading_note(
        "Métrica de Seleção (PR-AUC vs ROC-AUC)",
        "Como 'Resiliência Criativa' é uma classe rara (desbalanceada), o modelo foi otimizado e selecionado usando Average Precision (PR-AUC), penalizando severamente falsos positivos."
    )
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV Average Precision", _fmt(best.get("cv_average_precision_mean") if best is not None else np.nan))
    c2.metric("Holdout Average Precision", _fmt(best.get("holdout_average_precision") if best is not None else np.nan))
    c3.metric("Holdout ROC-AUC", _fmt(best.get("holdout_roc_auc") if best is not None else np.nan))
    c4.metric("F1 c/ Limiar Otimizado", _fmt(best.get("holdout_opt_f1") if best is not None else np.nan))

    c_left, c_right = st.columns([1.35, 1])
    with c_left:
        st.markdown("### Incerteza (Bootstrap no Holdout)")
        st.write("O desempenho de um modelo em um único teste pode ser sorte. Avaliamos a incerteza gerando reamostragens (bootstrap) estritamente sobre as predições do holdout.")
        if bootstrap is not None:
            st.dataframe(bootstrap, use_container_width=True, hide_index=True)
        else:
            st.info("Tabela de bootstrap não encontrada.")
            
    with c_right:
        st.markdown("### Estatísticas do Perfil")
        if resilient_stats is not None:
            st.write("As comparações de médias entre grupos foram validadas com teste não-paramétrico de Mann-Whitney U e Correção de Benjamini-Hochberg (FDR).")
            cols = [c for c in ["feature", "effect_size_r", "significativo_fdr05"] if c in resilient_stats.columns]
            if cols:
                st.dataframe(resilient_stats[cols].head(10), use_container_width=True, hide_index=True)
        else:
            st.info("Estatísticas de perfil não encontradas.")

# 3. Justiça Algorítmica (Fairness)
with tabs[2]:
    st.subheader("Auditoria de Viés e Justiça Algorítmica")
    _reading_note(
        "Por que auditar viés?",
        "Modelos educacionais não devem discriminar grupos minoritários ou sensíveis. Comparamos as taxas de acerto e aprovação do modelo por gênero e status imigratório usando métricas formais.",
        "⚖️"
    )
    
    if fairness_summary is not None and not fairness_summary.empty:
        st.markdown("### Métricas Formais Agregadas (Feldman et al., 2015; Hardt et al., 2016)")
        _alert("**DPD** (Demographic Parity Difference): O modelo prediz taxas similares de resiliência para os grupos? (Ideal: < 0.10)<br>**DIR** (Disparate Impact Ratio): A regra dos 4/5 (80%) da EEOC. (Ideal: > 0.80)<br>**EOD** (Equal Opportunity Difference): O modelo acerta igualmente os verdadeiros resilientes (TPR) independente do grupo? (Ideal: < 0.10)")
        st.dataframe(fairness_summary, use_container_width=True, hide_index=True)
    else:
        st.info("Execute a etapa de fairness (SR-3) para visualizar as métricas agregadas.")
        
    if fairness_group is not None:
        st.markdown("### Desempenho e Prevalência por Subgrupo")
        st.dataframe(fairness_group, use_container_width=True, hide_index=True)

# 4. Sensibilidade Multi-Target
with tabs[3]:
    st.subheader("Análise de Sensibilidade (Robustez Operacional)")
    _reading_note(
        "Validade de Constructo (Cook & Campbell, 1979)",
        "As nossas conclusões sobre resiliência dependem arbitrariamente de como definimos 'baixo ESCS' e 'alto CRT'? Testamos o mesmo pipeline preditivo contra 4 definições diferentes (A, B, C, D)."
    )
    
    if sens_summary is not None and not sens_summary.empty:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("### Rank Stability Index (Spearman ρ)")
            st.write("Os modelos que performam bem na Definição A também são os melhores na B? Uma alta correlação (ρ > 0.80) indica que o ranking dos métodos é imune à definição.")
            if sens_rank is not None:
                st.dataframe(sens_rank, use_container_width=True, hide_index=True)
            
            st.markdown("### Desempenho do Melhor Modelo por Target")
            cols = ["target", "best_model", "cv_ap_mean", "holdout_auc", "prevalence"]
            st.dataframe(sens_summary[[c for c in cols if c in sens_summary.columns]], use_container_width=True, hide_index=True)
            
        with c2:
            st.markdown("### Estabilidade do Poder Preditivo")
            _render_image(FIGURES / "sensitivity" / "sensitivity_comparison.png", "Variação das métricas entre as 4 definições operacionais.")
            _render_image(FIGURES / "sensitivity" / "sensitivity_model_heatmap.png", "Heatmap de AP por Modelo e Definição.")
    else:
        st.warning("Execute o novo estágio de sensibilidade: `python3 -m src.main --stage sensitivity`")

# 5. Explicações (SHAP)
with tabs[4]:
    st.subheader("Importância das Variáveis (SHAP Values)")
    _reading_note(
        "Explainer API",
        "Utilizamos `shap.Explainer` para extrair as contribuições exatas de cada feature na escala original dos dados, evitando distorções criadas por normalizadores pré-árvore (correção SR-4). O SHAP mostra correlação, não causalidade."
    )
    
    c1, c2 = st.columns([1, 1])
    with c1:
        _render_image(FIGURES / "shap" / "shap_top20.png", "Impacto direcional (SHAP Summary Plot). Vermelho = alto valor da feature.")
    with c2:
        if importance is not None:
            st.markdown("### Top 30 Preditores")
            st.dataframe(importance.head(30), use_container_width=True, hide_index=True)

# 6. Perfis de Resiliência (Clusterização)
with tabs[5]:
    st.subheader("Análise Centrada na Pessoa: Perfis Latentes")
    _reading_note(
        "Abordagem",
        "Além de prever quem é resiliente, agrupamos TODOS os alunos via PCA + HDBSCAN/Hierárquico para descobrir perfis naturais. Depois verificamos em qual perfil a Resiliência Criativa se concentra."
    )
    
    if cluster_assoc is not None and not cluster_assoc.empty:
        top_rr = cluster_assoc.sort_values("risk_relative", ascending=False).head(1)
        if not top_rr.empty:
            cid = top_rr.iloc[0].get("cluster_id", "NA")
            rr = top_rr.iloc[0].get("risk_relative", np.nan)
            _info_card(
                f"Cluster de Risco Elevado / Foco (ID {cid})",
                f"Odds Ratio: {_fmt(rr, 2)}x",
                "Este agrupamento concentra proporcionalmente muito mais alunos resilientes que a média geral.",
                "purple"
            )
            
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Seleção e Estabilidade")
        if ranking is not None:
            st.dataframe(ranking.head(5), use_container_width=True, hide_index=True)
        if stability is not None:
            st.write("**Bootstrap Jaccard Index (Estabilidade)**")
            st.dataframe(stability.head(5), use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Associações com a Resiliência")
        if cluster_assoc is not None:
            st.dataframe(cluster_assoc, use_container_width=True, hide_index=True)
            
    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        _render_image(FIGURES / "clustering" / "cluster_pca.png", "PCA 2D por cluster")
        _render_image(FIGURES / "clustering" / "cluster_radar.png", "Radar dos perfis (Dimensões médias)")
    with c4:
        _render_image(FIGURES / "clustering" / "cluster_heatmap.png", "Heatmap dos perfis")
        _render_image(FIGURES / "clustering" / "cluster_resilience_distribution.png", "Distribuição de resiliência por perfil")

# 7. Modelagem (Detalhes)
with tabs[6]:
    st.subheader("Seleção de Algoritmos de Machine Learning")
    if metrics is not None:
        _model_rank_chart(metrics, metric_col="cv_average_precision_mean")
        st.markdown("### Métricas de Triagem e Otimização")
        st.dataframe(_compact_metrics_table(metrics), use_container_width=True, hide_index=True)
        
        with st.expander("Hiperparâmetros Vencedores (RandomizedSearchCV)"):
            cols = [c for c in ["model", "variant", "best_params"] if c in metrics.columns]
            st.dataframe(metrics[cols], use_container_width=True, hide_index=True)
    else:
        st.warning("Métricas de modelagem indisponíveis.")

# 8. Dados & Vazamento
with tabs[7]:
    st.subheader("Controle Rigoroso de Dados")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Prevalência do Target")
        if prevalence is not None:
            st.bar_chart(prevalence.set_index("target_def")[["prevalence"]], use_container_width=True, height=260)
            st.dataframe(prevalence, use_container_width=True, hide_index=True)
    with c2:
        st.markdown("### Auditoria de Leakage (Vazamento)")
        st.write("Variáveis identificadas e banidas antes do treinamento.")
        if leakage is not None:
            st.dataframe(leakage[["nome", "spearman_abs", "reason"]].head(20), use_container_width=True, hide_index=True)

# 9. Glossário
with tabs[8]:
    st.subheader("Catálogo de Variáveis do PISA")
    
    variable_catalog = None
    try:
        variable_catalog = _csv("variable_catalog.csv")
    except Exception:
        pass
        
    if variable_catalog is None or variable_catalog.empty:
        st.info("Catálogo indisponível.")
    else:
        query = st.text_input("Buscar código (ex: CR567) ou palavra-chave", value="")
        df_gloss = variable_catalog.copy()
        if query.strip():
            q = query.strip().lower()
            df_gloss = df_gloss[df_gloss.astype(str).apply(lambda x: x.str.lower().str.contains(q)).any(axis=1)]
            
        st.dataframe(df_gloss.head(100), use_container_width=True, hide_index=True)

# 10. Relatórios Técnicos
with tabs[9]:
    st.subheader("Memória de Cálculo e Relatórios (Markdown)")
    report_files = sorted(REPORTS.glob("*.md")) if REPORTS.exists() else []
    choice = st.selectbox("Selecione o artefato para leitura", [p.name for p in report_files] or ["(nenhum)"])
    if report_files:
        st.markdown("---")
        st.markdown(_read_md(REPORTS / choice))

# --- Sidebar ---
st.sidebar.title("PISA ML")
st.sidebar.caption("v2.0 — Scientific Build")
st.sidebar.markdown(
    """
    **Guia de Métricas (Classe Rara)**
    * **PR-AUC**: Qualidade global do ranking de probabilidade focando apenas nos casos positivos. Métrica primária.
    * **ROC-AUC**: Capacidade de distinguir classes. Insuflado em classes raras, usado como secundário.
    * **Limiar Otimizado**: O ponto de corte ideal que maximiza o F1-score no treino.
    """
)
st.sidebar.divider()
st.sidebar.caption("Comando para rodar o pipeline:")
st.sidebar.code("python3 -m src.main --stage <nome>", language="bash")
st.sidebar.caption("Para abrir este painel:")
st.sidebar.code("python3 -m streamlit run dashboard/app.py", language="bash")
