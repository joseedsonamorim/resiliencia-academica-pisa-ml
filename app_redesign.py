# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Painel Interativo Streamlit - Análise de Pensamento Criativo PISA 2022.

Design Apple-style: minimalista, clean, branco e cinza claro.
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

st.set_page_config(
    layout="wide",
    page_title="PISA 2022 - Pensamento Criativo",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CSS - Design Apple Style
# ============================================================================
st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif; }

.block-container { 
    padding-top: 2.5rem; padding-left: 3rem; padding-right: 3rem; padding-bottom: 2rem;
    background-color: #FFFFFF;
}

.app-title { 
    font-size: 2.5rem; font-weight: 700; color: #000000;
    letter-spacing: -0.5px; margin-bottom: 0.5rem;
}

.app-subtitle { 
    color: #6E7681; font-size: 1.05rem; font-weight: 400;
    margin-bottom: 1rem; letter-spacing: -0.1px;
}

.app-helper { 
    color: #8B92A1; font-size: 0.95rem; line-height: 1.6; font-weight: 400;
}

.divider-line {
    border: 0; height: 1px; background: #E8EAED; margin: 1.5rem 0;
}

.kpi-card {
    border: 1px solid #E8EAED; border-radius: 10px; padding: 1.3rem;
    background: #FFFFFF; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
    transition: all 0.2s ease;
}

.kpi-card:hover {
    border-color: #D1D7DE; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.analysis-box {
    font-size: 1rem; line-height: 1.7; background-color: #F5F6F8;
    padding: 1.5rem; border-left: 3px solid #0A66C2; border-radius: 6px; margin: 1.5rem 0;
}

h1, h2, h3, h4 {
    color: #000000; letter-spacing: -0.3px; font-weight: 600;
}

h2 { font-size: 1.8rem; margin-bottom: 1rem; }
h3 { font-size: 1.3rem; margin-bottom: 0.8rem; }

.stButton > button {
    background-color: #0A66C2; color: white; border: 1px solid #0A66C2;
    border-radius: 8px; font-weight: 500; padding: 0.6rem 1.5rem; transition: all 0.2s;
}

.stButton > button:hover {
    background-color: #004BA0; box-shadow: 0 2px 8px rgba(10, 102, 194, 0.2);
}

.stSuccess { background-color: #F0FDF4; border: 1px solid #BBEF63; border-radius: 8px; }
.stInfo { background-color: #F0F4F8; border: 1px solid #D1DCE8; border-radius: 8px; }
.stWarning { background-color: #FFFBEB; border: 1px solid #FCD34D; border-radius: 8px; }
.stError { background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; }

</style>
""", unsafe_allow_html=True)

# ============================================================================
# CABEÇALHO
# ============================================================================
st.markdown('<h1 class="app-title">🧠 Pensamento Criativo PISA 2022</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Análise de Resiliência Criativa em Estudantes Latinoamericanos</p>', unsafe_allow_html=True)
st.markdown("""
<p class="app-helper">
Exploração de padrões entre status socioeconômico e criatividade, identificação de arquétipos 
e modelos preditivos com interpretabilidade SHAP.
</p>
""", unsafe_allow_html=True)
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

# ============================================================================
# DADOS
# ============================================================================
@st.cache_data(show_spinner="Carregando dados PISA 2022...")
def load_cached_data():
    return load_data()

df_all = load_cached_data()
if df_all.empty:
    st.error("Erro ao carregar dados")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🗂️ Guia")
    with st.expander("Como começar", expanded=True):
        st.markdown("1. Escolha países\n2. Explore a EDA\n3. Gere perfis\n4. Treine modelo\n5. Exporte")
    
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    
    st.markdown("## 🎯 Filtros")
    selected_countries = st.multiselect(
        "Países LATAM", LATAM_COUNTRIES, default=LATAM_COUNTRIES,
        help="Selecione países"
    )
    
    if not selected_countries:
        selected_countries = LATAM_COUNTRIES
    
    sample_size = st.slider(
        "Amostra para gráficos", 100, 2000, 1000,
        help="Tamanho da amostra"
    )
    
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Executar")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_kmeans = st.button("🧩 K-Means", key="kmeans", type="primary", use_container_width=True)
    with col_btn2:
        run_rf = st.button("🤖 Treinar", key="rf", use_container_width=True)
    
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    st.markdown("**Mestrado IA**  \nPPGIA UFRPE\n© 2026 JEAS")

# ============================================================================
# DADOS FILTRADOS
# ============================================================================
df_filtered = df_all[df_all['CNT'].isin(selected_countries)]

if df_filtered.empty:
    st.error("Nenhum dado com os filtros selecionados")
    st.stop()

if len(df_filtered) > sample_size:
    df_view = df_filtered.sample(sample_size, random_state=42)
else:
    df_view = df_filtered.copy()

# ============================================================================
# KPIs
# ============================================================================
resil_rate = df_filtered['Resiliente_Criativo'].mean()
baseline = df_all['Resiliente_Criativo'].mean()

st.markdown("### 📊 Visão Geral")

col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Estudantes", f"{len(df_filtered):,}")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Resilientes", f"{resil_rate:.1%}", f"{(resil_rate-baseline)*100:+.1f}pp")
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Média ESCS", f"{df_filtered['ESCS'].mean():.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Média PV1CREA", f"{df_filtered['PV1CREA'].mean():.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploração de Dados",
    "👥 Perfis",
    "🎯 Modelo",
    "📥 Exportar"
])

# ============================================================================
# TAB 1: EDA
# ============================================================================
with tab1:
    st.markdown("## Exploração de Dados")
    st.markdown("Visualize a relação entre status socioeconômico e pensamento criativo.")
    
    st.markdown("### O que é Resiliência Criativa?")
    with st.expander("Definição"):
        st.markdown("Um estudante é resiliente criativo quando: Q1 ESCS + Q4 PV1CREA (no mesmo país)")
    
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        scatter = (
            alt.Chart(df_view)
            .mark_circle(size=100, opacity=0.65)
            .encode(
                x=alt.X('ESCS:Q', title='ESCS'),
                y=alt.Y('PV1CREA:Q', title='PV1CREA'),
                color=alt.Color('Resiliente_Criativo:N', scale=alt.Scale(domain=['0','1'], range=['#D1D5DB','#0A66C2']), title='Tipo'),
                tooltip=['CNT', 'ESCS', 'PV1CREA']
            )
            .properties(title="ESCS vs Criatividade", width=700, height=450)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)
    
    with col_right:
        hist = (
            alt.Chart(df_view)
            .mark_bar(color='#0A66C2', opacity=0.8)
            .encode(
                x=alt.X('PV1CREA:Q', bin=alt.Bin(maxbins=30)),
                y=alt.Y('count()')
            )
            .properties(title="Distribuição PV1CREA", height=450)
        )
        st.altair_chart(hist, use_container_width=True)

# ============================================================================
# TAB 2: PERFIS
# ============================================================================
with tab2:
    st.markdown("## Perfis K-Means")
    st.markdown("Agrupa alunos resilientes em 4 arquétipos.")
    
    if run_kmeans:
        try:
            df_res, kmeans_model = kmeans_resilientes_criativos(df_filtered)
            st.session_state['df_clusters'] = df_res
            st.session_state['kmeans'] = kmeans_model
            st.success("✅ K-Means concluído")
        except Exception as e:
            st.error(f"Erro: {e}")
    
    if 'df_clusters' in st.session_state:
        df_c = st.session_state['df_clusters']
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            counts = df_c['arquétipo'].value_counts()
            bar = (
                alt.Chart(counts.reset_index())
                .mark_bar(color='#0A66C2')
                .encode(x=alt.X('arquétipo:N', sort='-y'), y='count()')
                .properties(title="Arquétipos", height=350)
            )
            st.altair_chart(bar, use_container_width=True)
        
        with col2:
            st.markdown("#### Características")
            profile = df_c.groupby('arquétipo')[['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']].mean().round(3)
            st.dataframe(profile, use_container_width=True)
        
        st.dataframe(df_c[['CNT', 'arquétipo', 'ESCS', 'PV1CREA']].head(15), use_container_width=True, hide_index=True)
    else:
        st.info("Execute K-Means para visualizar")

# ============================================================================
# TAB 3: MODELO
# ============================================================================
with tab3:
    st.markdown("## Modelo Preditivo")
    st.markdown("Random Forest + SHAP para prever resiliência criativa.")
    
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
            st.success("✅ Modelo treinado")
        except Exception as e:
            st.error(f"Erro: {e}")
    
    if 'report' in st.session_state:
        report = st.session_state['report']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("F1", f"{report.get('test_f1', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Precisão", f"{report.get('test_precision', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("Recall", f"{report.get('test_recall', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        col4, col5 = st.columns(2)
        with col4:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("ROC-AUC", f"{report.get('test_roc_auc', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("CV F1", f"{report.get('cv_f1_mean', 0):.3f} ± {report.get('cv_f1_std', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if 'shap_vals' in st.session_state:
            st.markdown("### 🔍 SHAP - Variáveis Importantes")
            shap_vals = st.session_state['shap_vals']
            X_sample = st.session_state['X_train_bal'].head(200)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_vals, X_sample, feature_names=CREATIVITY_FEATURES_RF, show=False, max_display=10)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
    else:
        st.info("Execute o treinamento para visualizar")

# ============================================================================
# TAB 4: EXPORTAR
# ============================================================================
with tab4:
    st.markdown("## Exportar Resultados")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📥 CSV")
        if st.button("🔽 Exportar Predições", key="export", use_container_width=True):
            if 'model' in st.session_state:
                model = st.session_state['model']
                X_test = st.session_state['X_test']
                
                pred = model.predict(X_test)
                prob = model.predict_proba(X_test)[:, 1]
                
                df_exp = pd.DataFrame(X_test, columns=CREATIVITY_FEATURES_RF)
                df_exp['pred'] = pred
                df_exp['prob'] = prob
                
                csv = df_exp.to_csv(index=False)
                st.download_button("⬇️ Baixar", csv, "predictions.csv", "text/csv")
                st.success("✅ CSV pronto")
            else:
                st.warning("Treine primeiro")
    
    with col2:
        st.markdown("### 💾 Cache")
        col_s, col_l = st.columns(2)
        
        with col_s:
            if st.button("💾 Salvar", use_container_width=True):
                if 'model' in st.session_state:
                    try:
                        ensure_models_dir()
                        save_model(st.session_state['model'], "rf_model")
                        st.success("✅ Salvo")
                    except Exception as e:
                        st.error(f"Erro: {e}")
                else:
                    st.warning("Treine primeiro")
        
        with col_l:
            if st.button("🔄 Carregar", use_container_width=True):
                try:
                    ensure_models_dir()
                    loaded = load_model("rf_model")
                    if loaded:
                        st.session_state['model'] = loaded
                        st.success("✅ Carregado")
                    else:
                        st.info("Sem cache")
                except Exception as e:
                    st.error(f"Erro: {e}")

# ============================================================================
# RODAPÉ
# ============================================================================
st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
st.markdown("""
<p class="app-helper" style="text-align: center;">
🧠 PPGIA UFRPE | © 2026 Jose Edson Amorim Sebastiao
</p>
""", unsafe_allow_html=True)
