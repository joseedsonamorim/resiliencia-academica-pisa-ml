# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Painel Interativo Streamlit - Design Apple-Style
Análise de Pensamento Criativo PISA 2022. Layout moderno, clean, branco e cinza claro.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import altair as alt
import numbers
from src.data_loader import load_data, LATAM_COUNTRIES
from src.ml_models import kmeans_resilientes_criativos, rf_classifier, shap_explainer, CREATIVITY_FEATURES_RF, CREATIVITY_FEATURES_KMEANS
from src.export_utils import save_model, load_model, ensure_models_dir

__rastreio_app__ = "jeas_pisa_streamlit_2026_ufrpe"

st.set_page_config(
    layout="wide",
    page_title="PISA 2022 - Pensamento Criativo",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS DESIGN - Apple Style (Branco, Cinza Claro)
# ============================================================================
st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif; }

/* Container */
.block-container { 
    padding-top: 2.5rem; 
    padding-left: 3rem; 
    padding-right: 3rem; 
    padding-bottom: 2rem; 
    background-color: #FFFFFF; 
}

/* Títulos */
.app-title { 
    font-size: 2.5rem; 
    font-weight: 700; 
    color: #000000;
    letter-spacing: -0.5px;
    margin-bottom: 0.5rem;
}

.app-subtitle { 
    color: #6E7681; 
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 1rem;
}

.app-helper { 
    color: #8B92A1; 
    font-size: 0.95rem; 
    line-height: 1.6;
    font-weight: 400;
}

/* Separadores */
.divider-line {
    border: 0;
    height: 1px;
    background: #E8EAED;
    margin: 1.5rem 0;
}

/* KPI Cards */
.kpi-card {
    border: 1px solid #E8EAED;
    border-radius: 10px;
    padding: 1.3rem;
    background: #FFFFFF;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    transition: all 0.2s ease;
}

.kpi-card:hover {
    border-color: #D1D7DE;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000;
    letter-spacing: -0.3px;
    font-weight: 600;
}

h2 { font-size: 1.8rem; margin-bottom: 1rem; }
h3 { font-size: 1.3rem; margin-bottom: 0.8rem; }

/* Buttons */
.stButton > button {
    background-color: #0A66C2;
    color: #FFFFFF;
    border: 1px solid #0A66C2;
    border-radius: 8px;
    font-weight: 500;
    letter-spacing: -0.2px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background-color: #004BA0;
    border-color: #004BA0;
    box-shadow: 0 2px 8px rgba(10, 102, 194, 0.2);
}

/* Alert Boxes */
.stSuccess { 
    background-color: #F0FDF4; 
    border: 1px solid #BBEF63; 
    border-radius: 8px;
    color: #166534;
}

.stSuccess p, .stSuccess span {
    color: #166534 !important;
}

.stInfo { 
    background-color: #F0F4F8; 
    border: 1px solid #D1DCE8; 
    border-radius: 8px;
    color: #0C4A6E;
}

.stInfo p, .stInfo span {
    color: #0C4A6E !important;
}

.stWarning { 
    background-color: #FFFBEB; 
    border: 1px solid #FCD34D; 
    border-radius: 8px;
    color: #92400E;
}

.stWarning p, .stWarning span {
    color: #92400E !important;
}

.stError { 
    background-color: #FEF2F2; 
    border: 1px solid #FECACA; 
    border-radius: 8px;
    color: #991B1B;
}

.stError p, .stError span {
    color: #991B1B !important;
}

/* Data Frames */
.dataframe {
    border: 1px solid #E8EAED;
    border-radius: 8px;
}

/* Metrics */
.stMetric {
    background-color: transparent;
    color: #000000;
}

.stMetric label {
    color: #6E7681 !important;
    font-weight: 500;
}

.stMetric .metric-value {
    color: #000000 !important;
    font-weight: 700;
}

/* Text Elements */
p, span, div, label, caption {
    color: #000000;
}

.stCaption {
    color: #6E7681 !important;
}

/* Expanders */
.stExpander {
    background-color: #FFFFFF;
    border: 1px solid #E8EAED;
    border-radius: 8px;
}

.stExpander summary {
    color: #000000;
    font-weight: 600;
}

.stExpander > div > div:last-child {
    color: #000000;
}

/* Input Elements */
.stTextInput input,
.stNumberInput input,
.stSelectbox select,
.stMultiSelect,
.stSlider {
    color: #000000 !important;
    background-color: #FFFFFF !important;
}

.stTextInput label,
.stNumberInput label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label {
    color: #000000 !important;
}

/* Tabs */
.stTabs button {
    color: #6E7681;
}

.stTabs button[aria-selected="true"] {
    color: #0A66C2;
    font-weight: 600;
}

/* Markdown & Headers */
h1, h2, h3, h4, h5, h6 {
    color: #000000 !important;
}

/* Tab content */
.tab-content {
    color: #000000;
}

/* Divider */
hr {
    background-color: #E8EAED;
}

</style>
""", unsafe_allow_html=True)

# ============================================================================
# CABEÇALHO PRINCIPAL
# ============================================================================
st.markdown('<h1 class="app-title">🧠 Pensamento Criativo PISA 2022</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="app-subtitle">Análise de Resiliência Criativa em Estudantes Latinoamericanos</p>
<p class="app-helper">
Exploração de padrões entre status socioeconômico e criatividade, identificação de arquétipos
e modelos preditivos com interpretabilidade SHAP.
</p>
""", unsafe_allow_html=True)
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

@st.cache_data(show_spinner="🔄 Carregando dados PISA 2022...")
def load_cached_data():
    return load_data()

df_all = load_cached_data()
if df_all.empty:
    st.error("❌ Dados vazios. Coloque CSV/Parquet em data/ ou use mock automático.")
    st.stop()

# ============================================================================
# SIDEBAR - NAVEGAÇÃO E CONTROLES
# ============================================================================
with st.sidebar:
    st.markdown("## 🗂️ Guia de Uso")
    with st.expander("Como começar", expanded=True):
        st.markdown("""
1. **Escolha países** → Selecione quais países incluir
2. **Explore a EDA** → Entenda os dados (Aba 1)
3. **Gere perfis** → Agrupe alunos resilientes (Aba 2)
4. **Treine modelo** → Crie preditor com SHAP (Aba 3)
5. **Exporte** → Baixe resultados em CSV (Aba 4)
        """)
    
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    
    st.markdown("## 🎯 Filtros de Dados")
    selected_countries = st.multiselect(
        "Países LATAM",
        LATAM_COUNTRIES,
        default=LATAM_COUNTRIES,
        help="Selecione quais países incluir na análise"
    )
    
    if not selected_countries:
        st.warning("Nenhum país selecionado. Usando todos.")
        selected_countries = LATAM_COUNTRIES
    
    sample_size = st.slider(
        "Amostra para gráficos",
        100, 2000, 1000,
        help="Tamanho da amostra para visualizações"
    )
    
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Executar Análises")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_kmeans = st.button(
            "🧩 K-Means",
            key="kmeans_btn",
            type="primary",
            width="stretch",
            help="Agrupa alunos resilientes em 4 perfis"
        )
    with col_btn2:
        run_rf = st.button(
            "🤖 Treinar",
            key="rf_btn",
            width="stretch",
            help="Treina modelo Random Forest + SHAP"
        )
    
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    
    st.markdown("## 📌 Sobre")
    st.markdown("**Mestrado em IA**  \nPPGIA UFRPE")
    st.caption("© 2026 Jose Edson A. Sebastiao")

# Filtrar dados
df_filtered = df_all[df_all['CNT'].isin(selected_countries)]
if df_filtered.empty:
    st.error("❌ Filtro resultou em 0 linhas. Verifique o filtro de países e se a coluna `CNT` está correta nos dados.")
    with st.expander("Diagnóstico rápido (dados carregados)"):
        st.write("Colunas:", list(df_all.columns))
        st.write("CNT únicos (top 20):", df_all['CNT'].astype(str).unique()[:20])
        st.write("Shape df_all:", df_all.shape)
        st.dataframe(df_all.head(20))
    st.stop()
if len(df_filtered) > sample_size:
    df_view = df_filtered.sample(sample_size, random_state=42)
else:
    df_view = df_filtered.copy()

resil_criativo_rate = df_filtered['Resiliente_Criativo'].mean()
baseline_latam = df_all['Resiliente_Criativo'].mean()

# Seção de KPIs
resil_criativo_rate = df_filtered['Resiliente_Criativo'].mean()
baseline_latam = df_all['Resiliente_Criativo'].mean()

st.markdown("### 📊 Visão Geral dos Dados Selecionados")

col1, col2, col3, col4 = st.columns(4, gap="small")
with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        "Estudantes",
        f"{len(df_filtered):,}",
        delta=None,
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    delta_val = f"{(resil_criativo_rate - baseline_latam)*100:+.1f}pp vs baseline"
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        "Resilientes Criativos",
        f"{resil_criativo_rate:.1%}",
        delta=delta_val
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        "Média ESCS",
        f"{df_filtered['ESCS'].mean():.2f}",
        delta=None
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        "Média PV1CREA",
        f"{df_filtered['PV1CREA'].mean():.0f}",
        delta=None
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise Exploratória de Dados (EDA)", "👥 Perfis de Resiliência Criativa", "🎯 Predição e SHAP", "📊 Exportar Relatório"])

with tab1:
    st.subheader("O que esta aba mostra")
    st.markdown(
        "- **Relação entre renda (ESCS) e criatividade (PV1CREA)**\n"
        "- **Quem são os alunos “resilientes criativos”** (baixa renda + alta criatividade dentro do país)\n"
        "- **Distribuição de PV1CREA** na amostra selecionada\n"
    )
    with st.expander("Entenda o conceito (definição usada no projeto)"):
        st.markdown(
            "- **Resiliente_Criativo = 1** quando o aluno está no **Q1 de ESCS** (mais vulnerável) e no **Q4 de PV1CREA** (mais alto) **dentro do mesmo país**.\n"
            "- O objetivo é comparar países/recortes e depois **criar perfis** (K-Means) e **predizer** (RF + SHAP) quais fatores mais pesam.\n"
        )
    
    col_left, col_right = st.columns([2,1])
    with col_left:
        scatter = (
            alt.Chart(df_view)
            .mark_circle(size=80, opacity=0.65)
            .encode(
                x=alt.X('ESCS:Q', title='Status Socioeconômico (ESCS)'),
                y=alt.Y('PV1CREA:Q', title='Pensamento Criativo (PV1CREA)'),
                color=alt.Color('Resiliente_Criativo:N', scale=alt.Scale(scheme='plasma'), title='Resil. Criativo'),
                tooltip=['CNT', 'ESCS', 'PV1CREA', 'HOMEPOS', 'ST29Q01', 'Resiliente_Criativo']
            )
            .properties(title="ESCS vs Criatividade: Identificando Resilientes Criativos", width=700, height=450)
            .interactive()
        )
        st.altair_chart(scatter, width='stretch')
    
    with col_right:
        hist_crea = (
            alt.Chart(df_view)
            .mark_bar(opacity=0.8, color='#ec4899')
            .encode(
                x=alt.X('PV1CREA:Q', bin=alt.Bin(maxbins=40), title='PV1CREA'),
                y=alt.Y('count()', title='Nº Alunos'),
                color=alt.value('#ec4899'),
                tooltip=['count()']
            )
            .properties(title="Distribuição Pensamento Criativo LATAM", height=220)
        )
        st.altair_chart(hist_crea, width='stretch')
    
    st.subheader("Como ler os gráficos")
    st.markdown(
        "- **Dispersão (ESCS × PV1CREA)**: procure pontos com **ESCS baixo** e **PV1CREA alto** (são candidatos a resiliência criativa).\n"
        "- **Cores**: indicam `Resiliente_Criativo` (0/1).\n"
        "- **Passe o mouse**: veja país e variáveis de contexto (ex.: `HOMEPOS`, `ST29Q01`, `IC004Q01`).\n"
    )

with tab2:
    st.subheader("Perfis (K-Means): como usar")
    st.markdown(
        "- Clique em **“Gerar perfis (K-Means)”** na barra lateral.\n"
        "- O K-Means roda **apenas** sobre alunos com `Resiliente_Criativo = 1`.\n"
        "- O resultado agrupa esses alunos em **arquétipos** com padrões semelhantes nas variáveis socioeconômicas e de contexto.\n"
    )
    if run_kmeans:
        try:
            df_res, kmeans_model = kmeans_resilientes_criativos(df_filtered)
            st.session_state['df_clusters'] = df_res
            st.session_state['kmeans'] = kmeans_model
            st.success("✅ Perfis K-Means gerados!")
        except Exception as e:
            st.error(f"❌ Erro K-Means: {e}")
    
    if 'df_clusters' in st.session_state:
        df_c = st.session_state['df_clusters']
        col1, col2 = st.columns(2)
        with col1:
            counts = df_c['arquétipo'].value_counts()
            bar = (
                alt.Chart(counts.reset_index())
                .mark_bar()
                .encode(x=alt.X('arquétipo:N', sort='-y'), y='count()')
                .properties(title='Distribuição Arquétipos')
            )
            st.altair_chart(bar, width='stretch')
        with col2:
            profile = df_c.groupby('arquétipo')[['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']].mean().round(2)
            st.markdown("**Média das variáveis por arquétipo** (quanto maior, mais alto em média)")
            st.dataframe(profile.T, width='stretch')
        st.markdown("**Exemplos de alunos por arquétipo**")
        st.dataframe(df_c[['CNT', 'arquétipo', 'ESCS', 'PV1CREA', 'HOMEPOS']].head(10), width='stretch')
    else:
        st.info("ℹ️ Ainda não há perfis gerados. Use o botão **“Gerar perfis (K-Means)”** na barra lateral.")

with tab3:
    st.subheader("Predição (Random Forest) e explicabilidade (SHAP)")
    st.markdown(
        "- Clique em **“Treinar predição (RF + SHAP)”** na barra lateral.\n"
        "- O modelo tenta prever `Resiliente_Criativo` usando variáveis de contexto (sem `PV1CREA`, para evitar vazamento).\n"
        "- O gráfico SHAP mostra **quais variáveis mais empurram** a predição para 0 ou 1.\n"
    )
    if run_rf:
        try:
            model, X_train_bal, X_test, report = rf_classifier(df_filtered)
            st.session_state['model'] = model
            st.session_state['X_train_bal'] = X_train_bal
            st.session_state['X_test'] = X_test
            st.session_state['report'] = report
            
            shap_vals, explainer = shap_explainer(model, X_train_bal.head(200))
            st.session_state['shap_vals'] = shap_vals
            st.session_state['explainer'] = explainer
            st.success("✅ RF treinado + SHAP!")
        except Exception as e:
            st.error(f"❌ Erro RF: {e}")
    
    if 'report' in st.session_state:
        report = st.session_state['report']
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("F1-Score (Resil.Criativo)", f"{report.get('test_f1', 0):.3f}")
        with col2: st.metric("Precisão", f"{report.get('test_precision', 0):.3f}")
        with col3: st.metric("Recall", f"{report.get('test_recall', 0):.3f}")
        
        col4, col5 = st.columns(2)
        with col4: st.metric("ROC-AUC (Teste)", f"{report.get('test_roc_auc', 0):.3f}")
        with col5: st.metric("CV F1-Score (5-fold)", f"{report.get('cv_f1_mean', 0):.3f} ± {report.get('cv_f1_std', 0):.3f}")
        
        st.divider()
        st.subheader("Detalhes por Classe (Teste)")
        if 'test' in report and isinstance(report['test'], dict):
            def _fmt_metric(val):
                if isinstance(val, numbers.Real) and not isinstance(val, bool):
                    return f"{float(val):.3f}"
                return val

            with st.expander("Ver relatório completo (JSON)", expanded=False):
                st.json({
                    k: ({sk: _fmt_metric(v) for sk, v in sv.items()} if isinstance(sv, dict) else _fmt_metric(sv))
                    for k, sv in report['test'].items()
                })
        
        if 'shap_vals' in st.session_state:
            st.subheader("Principais variáveis (SHAP)")
            shap_vals = st.session_state['shap_vals']
            X_sample = st.session_state['X_train_bal'].head(200)
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_vals, X_sample, feature_names=CREATIVITY_FEATURES_RF, show=False, max_display=10)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
    else:
        st.info("ℹ️ Ainda não há modelo treinado. Use o botão **“Treinar predição (RF + SHAP)”** na barra lateral.")

with tab4:
    st.subheader("Exportar e salvar resultados")
    st.markdown(
        "- **Baixar CSV**: disponível após treinar o modelo (aba 3).\n"
        "- **Salvar/Carregar modelo**: guarda o Random Forest para reuso sem novo treino.\n"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Baixar predições (CSV)")
        if st.button("🔽 Exportar CSV com Predições", key="export_csv"):
            if 'model' in st.session_state and 'report' in st.session_state:
                model = st.session_state['model']
                X_test = st.session_state['X_test']
                
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)[:, 1]
                
                df_export = pd.DataFrame(X_test, columns=CREATIVITY_FEATURES_RF)
                df_export['pred_Resiliente_Criativo'] = predictions
                df_export['prob_Resiliente_Criativo'] = probabilities
                
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="⬇️ Baixar CSV",
                    data=csv,
                    file_name="pisa_predictions.csv",
                    mime="text/csv"
                )
                st.success("✅ CSV pronto para download!")
            else:
                st.warning("⚠️ Treine o modelo primeiro na aba **“Predição e SHAP”**.")
    
    with col2:
        st.markdown("### 💾 Modelo em cache")
        col_save, col_load = st.columns(2)
        
        with col_save:
            if st.button("💾 Salvar Modelo RF", key="save_rf"):
                if 'model' in st.session_state:
                    try:
                        ensure_models_dir()
                        save_model(st.session_state['model'], "rf_model")
                        st.success("✅ Modelo salvo em cache!")
                    except Exception as e:
                        st.error(f"❌ Erro ao salvar: {e}")
                else:
                    st.warning("⚠️ Treine o modelo primeiro na aba **“Predição e SHAP”**.")
        
        with col_load:
            if st.button("🔄 Carregar Modelo RF", key="load_rf"):
                try:
                    ensure_models_dir()
                    model_loaded = load_model("rf_model")
                    if model_loaded:
                        st.session_state['model'] = model_loaded
                        st.success("✅ Modelo carregado de cache!")
                    else:
                        st.info("ℹ️ Nenhum modelo em cache encontrado")
                except Exception as e:
                    st.error(f"❌ Erro ao carregar: {e}")
    
    st.divider()
    st.markdown("### 📄 Resumo do treinamento")
    
    if 'report' in st.session_state:
        report = st.session_state['report']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Amostras Treino", len(st.session_state['X_train_bal']))
        with col2:
            st.metric("Total Amostras Teste", len(st.session_state['X_test']))
        with col3:
            st.metric("Features Utilizadas", len(CREATIVITY_FEATURES_RF))
        
        st.markdown("#### Métricas de Desempenho")
        metrics_dict = {
            "F1-Score (Teste)": report.get('test_f1', 0),
            "Precision (Teste)": report.get('test_precision', 0),
            "Recall (Teste)": report.get('test_recall', 0),
            "ROC-AUC (Teste)": report.get('test_roc_auc', 0),
            "F1-Score CV (média)": report.get('cv_f1_mean', 0),
            "F1-Score CV (desvio)": report.get('cv_f1_std', 0),
            "ROC-AUC CV (média)": report.get('cv_roc_auc_mean', 0),
        }
        
        for metric_name, metric_val in metrics_dict.items():
            st.write(f"**{metric_name}**: `{metric_val:.3f}`")
        
        # Resumo análise
        st.markdown("""
        #### 📋 Resumo Análise
        - **Modelo**: Random Forest (200 árvores, max_depth=10)
        - **Balanceamento**: SMOTE aplicado ao conjunto treino
        - **Validação**: Validação Cruzada 5-folds
        - **Target**: Resiliência Criativa (Q1 ESCS + Q4 PV1CREA)
        - **Features**: ESCS, HISEI, HOMEPOS, ST29Q01, IC004Q01 (sem PV1CREA para evitar data leak)
        
        **Interpretação**: 
        - F1-score ~0.6-0.7 indica bom balanceamento entre Precision e Recall
        - ROC-AUC > 0.7 sugere discriminação adequada das classes
        - CV F1 ≈ Test F1 indica baixo overfitting
        """)
    else:
        st.info("ℹ️ Treine o modelo na aba **“Predição e SHAP”** para ver o resumo aqui.")

st.markdown("---")
st.caption("🧠 PPGIA UFRPE | Fases 2-4 KDD: ML Criativo LATAM | Copyright 2026 JEAS")

