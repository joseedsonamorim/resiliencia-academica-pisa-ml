# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Visualizações Streamlit - Pensamento Criativo PISA 2022.

Plotly charts para EDA, perfis clustering, SHAP explicabilidade.
Atualizado para PV1CREA (Pensamento Criativo) vs legado PV1READ/MATH.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import shap
import numpy as np

def plot_proficiency_scatter(df: pd.DataFrame):
    """Scatter ESCS vs PV1CREA por país com trendline.
    
    Args:
        df: DataFrame com ESCS, PV1CREA, CNT.
    """
    fig = px.scatter(
        df, x='ESCS', y='PV1CREA', color='CNT', 
        facet_col='CNT', trendline='ols',
        hover_data=['ST29Q01', 'IC004Q01'],
        title="🎨 Pensamento Criativo (PV1CREA) vs Status Socioeconômico (ESCS)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_matrix(df: pd.DataFrame):
    """Matriz correlação de features principais.
    
    Args:
        df: DataFrame com features ESCS, HOMEPOS, PV1CREA, IC004Q01, ST29Q01.
    """
    corr_cols = ['ESCS', 'HOMEPOS', 'PV1CREA', 'IC004Q01', 'ST29Q01', 'HISEI']
    corr_cols = [col for col in corr_cols if col in df.columns]
    if len(corr_cols) > 0:
        corr = df[corr_cols].corr()
        fig = px.imshow(
            corr, text_auto=True, aspect="auto",
            title="📊 Correlações Features (Pensamento Criativo + Socio)",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, use_container_width=True)

def plot_resilience_heatmap(df: pd.DataFrame):
    """Heatmap correlações (compatibilidade)."""
    corr_cols = [col for col in ['PV1CREA', 'ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01'] 
                 if col in df.columns]
    if len(corr_cols) > 0:
        corr = df[corr_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", title="📊 Correlações")
        st.plotly_chart(fig, use_container_width=True)

def plot_performance_histogram(df: pd.DataFrame):
    """Histogramas distribuição PV1CREA por país."""
    if 'PV1CREA' not in df.columns:
        st.warning("Coluna PV1CREA não encontrada.")
        return
    
    fig = go.Figure()
    for cnt in df['CNT'].unique():
        mask = df['CNT'] == cnt
        fig.add_trace(go.Histogram(
            x=df.loc[mask, 'PV1CREA'], 
            name=cnt, 
            opacity=0.7,
            nbinsx=30
        ))
    
    fig.update_layout(
        height=500, 
        title="📈 Distribuição Pensamento Criativo (PV1CREA) por País",
        barmode='overlay',
        xaxis_title="PV1CREA",
        yaxis_title="Frequência"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_income_devices_bar(df: pd.DataFrame):
    """Bar chart HOMEPOS por quartil ESCS."""
    df_temp = df.copy()
    df_temp['ESCS_quartile'] = pd.qcut(df_temp['ESCS'], 4, labels=['Q1 Baixa', 'Q2', 'Q3', 'Q4 Alta'])
    bar_data = df_temp.groupby('ESCS_quartile')['HOMEPOS'].mean().reset_index()
    fig = px.bar(
        bar_data, x='ESCS_quartile', y='HOMEPOS', 
        title="💻 Dispositivos em Casa (HOMEPOS) por Quartil Renda",
        color='ESCS_quartile', 
        text='HOMEPOS'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title="HOMEPOS", xaxis_title="Quartil ESCS")
    st.plotly_chart(fig, use_container_width=True)

def plot_income_devices_scatter(df: pd.DataFrame):
    """Scatter ESCS vs HOMEPOS por país."""
    fig = px.scatter(
        df, x='ESCS', y='HOMEPOS', color='CNT', 
        facet_col='CNT',
        trendline='ols', 
        title="💰 Status Socioeconômico (ESCS) vs Dispositivos em Casa (HOMEPOS)"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_shap_summary(shap_values, X_test: pd.DataFrame, features: list):
    """SHAP summary plot (beeswarm)."""
    st.subheader("🔍 SHAP Global - Importância Features")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=features, show=False, plot_type="beeswarm")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro SHAP summary: {e}")

def plot_shap_waterfall(shap_values_instance, feature_names, prediction):
    """SHAP waterfall plot predição individual."""
    st.subheader("🌊 SHAP Waterfall - Explicação Predição Individual")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_instance, base_values=0, 
                data=np.zeros(len(feature_names)), 
                feature_names=feature_names
            ), 
            show=False
        )
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro SHAP waterfall: {e}")

def plot_cluster_profiles(centers: np.ndarray, archetypes: list, features: list):
    """Tabela + bar chart características médias clusters."""
    st.subheader("👥 Características Médias Clusters (Arquétipos Criativos)")
    centers_df = pd.DataFrame(centers, columns=features, index=archetypes)
    st.dataframe(centers_df.round(3), use_container_width=True)
    
    fig = px.bar(
        centers_df.reset_index().melt(id_vars='index'), 
        x='variable', y='value', color='index', 
        title="Médias Clusters por Feature", 
        barmode='group'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_scatter(X_scaled: np.ndarray, labels: np.ndarray, archetypes: list):
    """Scatter plot clusters (2D)."""
    fig = px.scatter(
        x=X_scaled[:, 0], y=X_scaled[:, 1], 
        color=labels.astype(str),
        title="🎯 Visualização Clusters (2D - Primeiros Componentes)",
        labels={'x': 'Dimensão 1', 'y': 'Dimensão 2', 'color': 'Cluster'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_shap_feature_importance(shap_values, X_test, features):
    """Bar chart top SHAP features."""
    st.subheader("⭐ Top Features - Importância SHAP")
    feature_imp = np.abs(shap_values).mean(0)
    imp_df = pd.DataFrame({'feature': features, 'imp': feature_imp}).sort_values('imp', ascending=True)
    
    fig = px.bar(
        imp_df.tail(8), x='imp', y='feature', 
        orientation='h', 
        title="Top 8 Features por Importância SHAP (valor absoluto)",
        color='feature'
    )
    st.plotly_chart(fig, use_container_width=True)

import matplotlib.pyplot as plt


