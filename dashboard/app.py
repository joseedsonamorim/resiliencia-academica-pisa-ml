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


def _read_md(path: Path, max_chars: int = 14000) -> str:
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
        return pd.read_csv(p)
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
    if metrics is None or metrics.empty or "cv_roc_auc_mean" not in metrics.columns:
        return None
    return metrics.sort_values(["cv_roc_auc_mean", "holdout_roc_auc"], ascending=False).iloc[0]


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
        "A": "definição mais conservadora: baixo ESCS e alto desempenho criativo por quartis.",
        "B": "definição intermediária: grupo socioeconômico baixo ampliado e desempenho criativo alto.",
        "C": "definição mais estrita: baixo ESCS e desempenho criativo no topo da distribuição.",
        "D": "definição exploratória: escore composto de criatividade acima do contexto socioeconômico.",
    }
    return explanations.get(str(target_key), "definição operacional selecionada no arquivo de configuração.")


def _info_card(title: str, value: str, note: str, tone: str = "blue") -> None:
    colors = {
        "blue": ("#1d4ed8", "#eef5ff"),
        "green": ("#047857", "#effaf4"),
        "amber": ("#b45309", "#fff7ed"),
        "rose": ("#be123c", "#fff1f2"),
        "ink": ("#334155", "#f8fafc"),
    }
    accent, bg = colors.get(tone, colors["blue"])
    st.markdown(
        f"""
        <div class="info-card" style="border-top-color:{accent}; background:{bg};">
          <div class="info-title">{html.escape(title)}</div>
          <div class="info-value">{html.escape(value)}</div>
          <div class="info-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _reading_note(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="reading-note">
          <strong>{html.escape(title)}</strong>
          <span>{html.escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _model_rank_chart(metrics: pd.DataFrame) -> None:
    if "cv_roc_auc_mean" not in metrics.columns:
        return
    rank = metrics.sort_values("cv_roc_auc_mean", ascending=False).head(10)
    chart = rank.set_index("model")[["cv_roc_auc_mean"]]
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
        "cv_roc_auc_mean",
        "cv_roc_auc_ci_low",
        "cv_roc_auc_ci_high",
        "cv_average_precision_mean",
        "cv_f1_mean",
        "holdout_roc_auc",
        "holdout_average_precision",
        "holdout_f1",
        "holdout_precision",
        "holdout_recall",
        "holdout_opt_f1",
    ]
    return metrics[[c for c in cols if c in metrics.columns]]


st.set_page_config(
    page_title="PISA Resiliência Acadêmica",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """
    <style>
    :root {
      --ink: #182230;
      --muted: #64748b;
      --line: #d9e2ef;
      --paper: #fbfcfe;
      --blue: #1d4ed8;
      --green: #047857;
      --amber: #b45309;
      --rose: #be123c;
    }
    .stApp { background: linear-gradient(180deg, #f7f9fc 0%, #ffffff 42%); color: var(--ink); }
    .main .block-container { max-width: 1360px; padding-top: 1.25rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
    h1 { font-size: clamp(2rem, 3.5vw, 3.5rem); line-height: 1.02; margin-bottom: .2rem; }
    h2 { margin-top: 1.3rem; }
    div[data-testid="stMetric"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      background: #ffffff;
      box-shadow: 0 8px 20px rgba(15, 23, 42, .05);
    }
    div[data-testid="stMetric"] label { color: var(--muted); }
    .hero {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 22px 24px;
      background:
        linear-gradient(120deg, rgba(29, 78, 216, .10), rgba(4, 120, 87, .07) 52%, rgba(180, 83, 9, .08)),
        #ffffff;
      box-shadow: 0 14px 34px rgba(15, 23, 42, .07);
      margin-bottom: 16px;
    }
    .hero-kicker {
      color: #1d4ed8;
      font-size: .82rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 7px;
    }
    .hero-body {
      max-width: 960px;
      color: #334155;
      font-size: 1.04rem;
      line-height: 1.56;
      margin-top: 8px;
    }
    .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }
    .pill {
      display:inline-flex; align-items:center; gap:6px;
      border:1px solid rgba(30,41,59,.16);
      border-radius:999px;
      padding:6px 10px;
      background: rgba(255,255,255,.72);
      color:#334155;
      font-size:.86rem;
      font-weight:650;
    }
    .info-card {
      border: 1px solid var(--line);
      border-top: 4px solid var(--blue);
      border-radius: 8px;
      padding: 14px 15px 13px;
      min-height: 142px;
      box-shadow: 0 8px 22px rgba(15, 23, 42, .045);
    }
    .info-title {
      color: #475569;
      font-size: .82rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: .06em;
      margin-bottom: 7px;
    }
    .info-value {
      color: var(--ink);
      font-size: 1.45rem;
      line-height: 1.12;
      font-weight: 820;
      margin-bottom: 8px;
    }
    .info-note {
      color: #475569;
      font-size: .93rem;
      line-height: 1.35;
    }
    .reading-note {
      display: grid;
      grid-template-columns: minmax(130px, .25fr) 1fr;
      gap: 10px;
      border-left: 4px solid #1d4ed8;
      background: #f8fbff;
      border-radius: 6px;
      padding: 11px 13px;
      margin: 10px 0 16px;
      color: #334155;
    }
    .reading-note strong { color: #1e293b; }
    .section-label {
      color:#475569;
      font-weight: 760;
      text-transform: uppercase;
      letter-spacing: .06em;
      font-size: .8rem;
      margin: 4px 0 6px;
    }
    .small-copy { color:#475569; font-size:.95rem; line-height:1.48; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
    @media (max-width: 780px) {
      .hero { padding: 18px 16px; }
      .reading-note { grid-template-columns: 1fr; }
      .info-card { min-height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
fairness = _csv("fairness_by_group.csv")
importance = _csv("shap_feature_importance.csv")
best = _best_metric(metrics)

target_key = str(meta.get("target_key", "A"))
target_stats = _target_row(prevalence, target_key)
dataset_label = project_meta.get("dataset", {}).get("csv_detected", meta.get("dataset", "NA"))
n_total = int(target_stats["n"]) if target_stats is not None and "n" in target_stats else None
n_pos = int(target_stats["n_pos"]) if target_stats is not None and "n_pos" in target_stats else None
target_prev = target_stats["prevalence"] if target_stats is not None and "prevalence" in target_stats else np.nan
best_threshold = meta.get("best_threshold", np.nan)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">PISA 2022 Brasil · análise científica reprodutível</div>
      <h1>Resiliência acadêmica e criativa em estudantes brasileiros</h1>
      <div class="hero-body">
        Este painel mostra como o estudo define estudantes resilientes, compara algoritmos,
        estima incerteza e interpreta os sinais que mais ajudam a separar os grupos.
        A leitura começa pela evidência principal e abre os detalhes metodológicos em seguida.
      </div>
      <div class="pill-row">
        <span class="pill">Dataset: {html.escape(str(dataset_label))}</span>
        <span class="pill">Target ativo: {html.escape(target_key)}</span>
        <span class="pill">Validação: CV repetida + holdout</span>
        <span class="pill">Foco: evidência, incerteza e explicabilidade</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

top_cols = st.columns(4)
with top_cols[0]:
    _info_card(
        "O que é resiliente aqui",
        f"Target {target_key}",
        _target_explanation(target_key),
        "blue",
    )
with top_cols[1]:
    _info_card(
        "Casos positivos",
        f"{n_pos:,}" if n_pos is not None else "NA",
        f"Prevalência de {_pct(target_prev)} em {n_total:,} estudantes." if n_total else "Prevalência calculada na etapa de targets.",
        "amber",
    )
with top_cols[2]:
    _info_card(
        "Melhor método",
        str(best["model"]) if best is not None else "NA",
        f"Selecionado por ROC-AUC médio em CV: {_fmt(best.get('cv_roc_auc_mean') if best is not None else np.nan)}.",
        "green",
    )
with top_cols[3]:
    _info_card(
        "Limiar recomendado",
        _fmt(best_threshold, 3),
        "Ajustado no treino para melhorar F1 em classe rara; não é uma verdade clínica.",
        "rose",
    )

tabs = st.tabs(
    [
        "Visão guiada",
        "Dados e target",
        "Métodos testados",
        "Decisão e probabilidades",
        "Explicações",
        "Perfis de Resiliência Criativa",
        "Relatórios técnicos",
    ]
)


with tabs[0]:
    st.markdown('<div class="section-label">Leitura rápida</div>', unsafe_allow_html=True)
    _reading_note(
        "Pergunta do estudo",
        "Entre estudantes com menor nível socioeconômico, quais sinais ajudam a prever desempenho criativo alto?",
    )
    _reading_note(
        "Resultado central",
        "O modelo vencedor discrimina bem os grupos, mas a classe positiva é rara; por isso recall, precisão e calibração precisam ser lidos junto com ROC-AUC.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV ROC-AUC", _fmt(best.get("cv_roc_auc_mean") if best is not None else np.nan))
    c2.metric("Holdout ROC-AUC", _fmt(best.get("holdout_roc_auc") if best is not None else np.nan))
    c3.metric("F1 otimizado", _fmt(best.get("holdout_opt_f1") if best is not None else np.nan))
    c4.metric("Features analisadas", str(len(meta.get("feature_cols", []))) if meta else "NA")

    c_left, c_right = st.columns([1.35, 1])
    with c_left:
        st.subheader("Ranking dos modelos")
        if metrics is not None:
            _model_rank_chart(metrics)
            st.caption("Barras mostram o ROC-AUC médio em validação cruzada; o desempenho no holdout fica na tabela técnica.")
        else:
            st.warning("Execute `python3 -m src.main --stage modeling` para gerar métricas.")
    with c_right:
        st.subheader("Incerteza do vencedor")
        if bootstrap is not None:
            auc_mean = _bootstrap_value(bootstrap, "roc_auc", "mean")
            auc_low = _bootstrap_value(bootstrap, "roc_auc", "ci_low")
            auc_high = _bootstrap_value(bootstrap, "roc_auc", "ci_high")
            _info_card(
                "ROC-AUC por bootstrap",
                _fmt(auc_mean),
                f"IC95%: {_fmt(auc_low)} a {_fmt(auc_high)} em reamostragens do holdout.",
                "blue",
            )
            st.dataframe(bootstrap, use_container_width=True, hide_index=True)
        else:
            st.info("Tabela de bootstrap ainda não encontrada.")

    with st.expander("Resumo objetivo do estudo"):
        st.markdown(_read_md(REPORTS / "relatorio_resumido.md", 6000))

with tabs[1]:
    st.subheader("O que está sendo analisado")
    _reading_note(
        "Unidade de análise",
        "Cada linha representa um estudante brasileiro no microdado tratado do PISA 2022.",
    )
    _reading_note(
        "Definição operacional",
        "O target combina contexto socioeconômico baixo com desempenho criativo alto; esta escolha deve ser justificada teoricamente no artigo.",
    )

    c1, c2 = st.columns([1, 1.1])
    with c1:
        st.markdown('<div class="section-label">Prevalência dos targets</div>', unsafe_allow_html=True)
        if prevalence is not None:
            prev_chart = prevalence.set_index("target_def")[["prevalence"]]
            st.bar_chart(prev_chart, use_container_width=True, height=260)
            st.dataframe(prevalence, use_container_width=True, hide_index=True)
        else:
            st.info("Tabela de targets não encontrada.")
    with c2:
        st.markdown('<div class="section-label">Controle de vazamento</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="small-copy">Variáveis diretamente derivadas do target, IDs e pesos são removidos antes da modelagem para evitar desempenho artificial.</div>',
            unsafe_allow_html=True,
        )
        if leakage is not None:
            cols = [c for c in ["nome", "severity", "score_corr_abs", "spearman_abs", "reason"] if c in leakage.columns]
            st.dataframe(leakage[cols].head(30), use_container_width=True, hide_index=True)
        else:
            st.info("Auditoria de vazamento ainda não encontrada.")

    st.subheader("Features usadas no modelo")
    if features is not None:
        st.dataframe(features.head(80), use_container_width=True, hide_index=True)
    else:
        st.info("A etapa de modelagem ainda não gerou a lista de features.")

    with st.expander("Ver relatório de target e auditoria de dados"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(_read_md(REPORTS / "target_comparison.md", 8000))
        with c2:
            st.markdown(_read_md(REPORTS / "data_audit.md", 8000))

with tabs[2]:
    st.subheader("Métodos testados e critério de escolha")
    _reading_note(
        "Como o melhor foi escolhido",
        "A seleção principal usa ROC-AUC médio em validação cruzada repetida dentro do treino; o holdout é preservado para a avaliação final.",
    )
    _reading_note(
        "Por que vários modelos",
        "Modelos lineares, árvores, boosting, SVM, KNN e Naive Bayes testam hipóteses diferentes sobre a estrutura dos dados.",
    )

    if metrics is None:
        st.warning("Execute `python3 -m src.main --stage modeling` para gerar métricas.")
    else:
        _model_rank_chart(metrics)
        st.dataframe(_compact_metrics_table(metrics), use_container_width=True, hide_index=True)
        with st.expander("Parâmetros dos modelos ajustados"):
            cols = [c for c in ["model", "variant", "best_params"] if c in metrics.columns]
            st.dataframe(metrics[cols], use_container_width=True, hide_index=True)

        failures = _csv("modeling_failures.csv")
        if failures is not None:
            with st.expander("Modelos que falharam ou foram descartados"):
                st.dataframe(failures, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Como transformar probabilidade em decisão")
    _reading_note(
        "Limiar",
        "O modelo produz uma probabilidade; o limiar decide quando chamar alguém de resiliente. Em classes raras, 0.50 costuma ser conservador demais.",
    )

    c1, c2 = st.columns([1.25, 1])
    with c1:
        if thresholds is not None:
            _threshold_curve(thresholds)
            st.caption("A curva ajuda a escolher o equilíbrio entre precisão, recall, F1 e acurácia balanceada.")
        else:
            st.info("Tabela de limiares ainda não encontrada.")
    with c2:
        if best is not None:
            _info_card(
                "Com limiar 0.50",
                f"F1 {_fmt(best.get('holdout_f1'))}",
                f"Precisão {_fmt(best.get('holdout_precision'))}; recall {_fmt(best.get('holdout_recall'))}.",
                "ink",
            )
            _info_card(
                "Com limiar otimizado",
                f"F1 {_fmt(best.get('holdout_opt_f1'))}",
                f"Limiar {_fmt(best_threshold, 3)}; melhor equilíbrio para a classe rara.",
                "green",
            )

    if thresholds is not None:
        st.dataframe(thresholds.head(25), use_container_width=True, hide_index=True)

    if predictions is not None:
        c1, c2 = st.columns([1, 1.1])
        with c1:
            st.subheader("Distribuição das probabilidades")
            st.bar_chart(_probability_bins(predictions), use_container_width=True, height=300)
        with c2:
            st.subheader("Predições no holdout")
            st.dataframe(predictions.head(120), use_container_width=True, hide_index=True)

    _render_image(FIGURES / "robustness" / "calibration_curve.png", "Curva de calibração")

with tabs[4]:
    st.subheader("Quais sinais ajudam a explicar o resultado")

    _reading_note(
        "Importância",
        "A importância indica quais variáveis mais contribuíram para separar os grupos no modelo; ela não prova causalidade.",
    )
    _reading_note(
        "Equidade",
        "A aba também mostra métricas por grupos sensíveis quando disponíveis, para detectar diferenças de prevalência ou desempenho.",
    )

    c1, c2 = st.columns([1.05, 1])
    with c1:
        _render_image(FIGURES / "shap" / "shap_top20.png", "Top 20 variáveis por importância")
        if importance is not None:
            st.dataframe(importance.head(30), use_container_width=True, hide_index=True)
    with c2:
        _render_image(
            FIGURES / "resilient_profile" / "resilient_profile_heatmap.png",
            "Perfil comparativo de estudantes resilientes",
        )
        _render_image(
            FIGURES / "resilient_profile" / "resilient_profile_radar.png",
            "Radar do perfil resiliente",
        )

    if fairness is not None:
        st.subheader("Fairness por grupo")
        st.dataframe(fairness, use_container_width=True, hide_index=True)
    else:
        st.info("A etapa de fairness ainda não gerou tabela.")

with tabs[5]:
    # Perfis de Resiliência Criativa (clusterização person-centered)
    st.subheader("Perfis de Resiliência Criativa")

    ranking = _csv("clustering_model_comparison.csv")
    stability = _csv("cluster_stability.csv")
    profiles = _csv("cluster_profiles_interpretable.csv")
    assoc = _csv("cluster_resilience_association.csv")

    st.markdown("### Ranking e escolha do melhor modelo")
    if ranking is not None and not ranking.empty:
        st.dataframe(ranking.head(20), use_container_width=True, hide_index=True)
        if "cluster_score" in ranking.columns and len(ranking) > 0:
            st.caption("A seleção usa score multicritério (qualidade + estabilidade), não apenas silhouette.")
    else:
        st.info("`outputs/tables/clustering_model_comparison.csv` não encontrado.")

    st.markdown("### Estabilidade dos clusters (bootstrap)")
    if stability is not None and not stability.empty:
        st.dataframe(stability.head(20), use_container_width=True, hide_index=True)
    else:
        st.info("`outputs/tables/cluster_stability.csv` não encontrado.")

    st.markdown("### Associações cluster × resiliência criativa")
    if assoc is not None and not assoc.empty:
        st.dataframe(assoc, use_container_width=True, hide_index=True)
    else:
        st.info("`outputs/tables/cluster_resilience_association.csv` não encontrado.")

    st.markdown("### Perfis interpretáveis")
    if profiles is not None and not profiles.empty:
        st.dataframe(profiles.head(60), use_container_width=True, hide_index=True)
    else:
        st.info("`outputs/tables/cluster_profiles_interpretable.csv` não encontrado.")

    # Imagens obrigatórias
    st.markdown("### Visualizações (clusterização)")
    _render_image(FIGURES / "clustering" / "scree_plot.png", "Scree plot (PCA)")
    _render_image(FIGURES / "clustering" / "cumulative_variance.png", "Variância acumulada (PCA)")
    _render_image(FIGURES / "clustering" / "cluster_pca.png", "PCA 2D por cluster")
    _render_image(FIGURES / "clustering" / "cluster_umap.png", "UMAP 2D por cluster")
    _render_image(FIGURES / "clustering" / "cluster_heatmap.png", "Heatmap dos perfis")
    _render_image(FIGURES / "clustering" / "cluster_radar.png", "Radar dos perfis")
    _render_image(FIGURES / "clustering" / "cluster_resilience_distribution.png", "Distribuição de resiliência por perfil")

with tabs[6]:
    st.subheader("Relatórios técnicos e rastreabilidade")
    _reading_note(
        "Rastreabilidade",
        "Esta seção preserva os relatórios completos gerados pelo pipeline para auditoria, revisão por pares e escrita do artigo.",
    )
    report_files = sorted(REPORTS.glob("*.md")) if REPORTS.exists() else []
    choice = st.selectbox("Relatório", [p.name for p in report_files] or ["(nenhum)"])
    if report_files:
        st.markdown(_read_md(REPORTS / choice))

st.sidebar.header("Como ler")
st.sidebar.markdown(
    """
**ROC-AUC**: separação entre resilientes e não resilientes.

**Average precision**: mais informativa quando a classe positiva é rara.

**Recall**: quantos resilientes reais o modelo encontra.

**Precisão**: quantos classificados como resilientes realmente são positivos.

**F1**: equilíbrio entre precisão e recall.
"""
)
st.sidebar.divider()
st.sidebar.header("Pipeline")
st.sidebar.code("python3 -m src.main --stage <nome>", language="bash")
st.sidebar.markdown(
    """
- `target_comparison`
- `leakage_audit`
- `modeling`
- `shap`
- `fairness`
- `robustness`
- `dashboard`
"""
)
st.sidebar.caption("Para abrir localmente:")
st.sidebar.code("python3 -m streamlit run dashboard/app.py", language="bash")
