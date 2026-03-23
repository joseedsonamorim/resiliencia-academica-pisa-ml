# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

\"\"\"Módulo 3: Painel Streamlit - Fases 2-4 KDD completas.

3 Abas: EDA | Perfis Resiliência (KMeans) | Predição+SHAP (RF).
Sidebar: Título/autor + filtro CNT.
Caches otimizados, error handling.
\"\"\"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data
from src.ml_models import kmeans_resilientes, rf_classifier, shap_explainer

__rastreio_app__ = \"jeas_pisa_streamlit_2026_ufrpe\"

st.set_page_config(layout=\"wide\", page_title=\"Resiliência PISA 2022\")

st.title(\"Modelagem da Resiliência Acadêmica - PISA 2022\")
st.markdown(\"*PPGIA UFRPE - Jose Edson Amorim Sebastiao (2026)*\")

# Sidebar
with st.sidebar:
    st.header(\"Filtros\")
    cnt_options = ['BRA', 'CHL', 'COL']
    selected_cnt = st.selectbox(\"País (CNT):\", cnt_options, index=0)

@st.cache_data
def load_cached_data():
    df = load_data()
    return df[df['CNT'] == selected_cnt]

df = load_cached_data()
if df.empty:
    st.error(\"Dados vazios após filtro.\")
    st.stop()

st.sidebar.info(f\"Alunos: {len(df):,}\")

# TABS
tab1, tab2, tab3 = st.tabs([\"EDA (Fase 2)\", \"Perfis Resiliência (Fase 3)\", \"Predição + SHAP (Fase 4)\"])

with tab1:
    st.header(\"Análise Exploratória\")
    col1, col2 = st.columns(2)
    with col1:
        total = len(df)
        resil = df['Resiliente'].sum()
        st.metric(\"Total Alunos\", total)
        st.metric(\"Resilientes\", f\"{resil} ({resil/total:.1%})\")
    
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data=df, x='ESCS', y='PV1MATH', hue='Resiliente', palette='coolwarm', ax=ax)
    ax.set_xlabel(\"ESCS (Status Socioeconômico)\")
    ax.set_ylabel(\"PV1MATH (Nota Matemática)\")
    ax.set_title(\"Dispersão Resiliência: Baixo SES + Alto Desempenho\")
    st.pyplot(fig)

with tab2:
    st.header(\"Perfis de Resilientes (K-Means n=4)\")
    if st.button(\"Processar Perfis\", type=\"primary\"):
        df_res, kmeans = kmeans_resilientes(df)
        st.session_state.df_clusters = df_res
        st.session_state.kmeans = kmeans
        st.success(\"Clusters computados!\")
    
    if 'df_clusters' in st.session_state:
        df_c = st.session_state.df_clusters
        st.dataframe(df_c[['arquétipo', 'ESCS', 'HISEI', 'HOMEPOS', 'PV1MATH']].head(10))
        
        fig, ax = plt.subplots(figsize=(10,6))
        counts = df_c['arquétipo'].value_counts()
        counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(\"Contagem Alunos por Arquétipo\")
        ax.set_ylabel(\"Nº Alunos\")
        plt.xticks(rotation=45)
        st.pyplot(fig)

with tab3:
    st.header(\"Predição Resiliente + SHAP\")
    if st.button(\"Treinar Random Forest (SMOTE)\", type=\"primary\"):
        model, X_train_bal, X_test, report = rf_classifier(df)
        st.session_state.model = model
        st.session_state.X_train_bal = X_train_bal
        st.session_state.X_test = X_test
        st.session_state.report = report
        shap_vals, explainer = shap_explainer(model, X_train_bal.head(100))
        st.session_state.shap_vals = shap_vals
        st.session_state.explainer = explainer
        st.success(\"Modelo treinado! F1 Resiliente:\")
        st.metric(\"F1-Score (Resiliente=1)\", f\"{report['1']['f1-score']:.3f}\")
    
    if 'report' in st.session_state:
        st.code(st.session_state.report['1'])
        
        if 'shap_vals' in st.session_state:
            fig = st.session_state.explainer.summary_plot(
                st.session_state.shap_vals, 
                st.session_state.X_train_bal.head(100).values,
                feature_names=['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01'],
                show=False
            )
            plt.tight_layout()
            st.pyplot(fig)

st.markdown(\"---\")
st.caption(\"Completo: Fases 2-4 KDD | Copyright 2026 JEAS\")

