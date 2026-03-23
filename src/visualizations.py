# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import shap
import numpy as np

def plot_proficiency_scatter(df: pd.DataFrame):
    fig = px.scatter(df, x='ESCS', y='PV1READ', color='ST01Q01', facet_col='CNT', trendline='ols', hover_data=['PV1MATH', 'PV1SCIE'], title=\"Proficiencia vs SES\")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_resilience_heatmap(df: pd.DataFrame):
    corr_cols = ['PV1READ', 'PV1MATH', 'PV1SCIE', 'ESCS', 'HISEI', 'HOMEPOS']
    corr = df[corr_cols].corr()
    fig = px.imshow(corr, text_auto=True, aspect=\"auto\", title=\"Corrélacoes\")
    st.plotly_chart(fig, use_container_width=True)

def plot_performance_histogram(df: pd.DataFrame):
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Reading', 'Math', 'Science'))
    for i, subj in enumerate(['PV1READ', 'PV1MATH', 'PV1SCIE']):
        for cnt in df['CNT'].unique()[:3]:
            mask = df['CNT'] == cnt
            fig.add_trace(go.Histogram(x=df.loc[mask, subj], name=cnt, opacity=0.7), row=1, col=i+1)
    fig.update_layout(height=500, title=\"Distribuicoes Proficiencia\")
    st.plotly_chart(fig, use_container_width=True)

def plot_income_devices_bar(df: pd.DataFrame):
    df['ESCS_quartile'] = pd.qcut(df['ESCS'], 4, labels=['Q1 Baixa', 'Q2', 'Q3', 'Q4 Alta'])
    bar_data = df.groupby('ESCS_quartile')['HOMEPOS'].mean().reset_index()
    fig = px.bar(bar_data, x='ESCS_quartile', y='HOMEPOS', title=\"HOMEPOS por Quartil ESCS\", color='ESCS_quartile', text='HOMEPOS')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_title=\"HOMEPOS\", xaxis_title=\"Quartil Renda\")
    st.plotly_chart(fig, use_container_width=True)

def plot_income_devices_scatter(df: pd.DataFrame):
    fig = px.scatter(df, x='ESCS', y='HOMEPOS', color='CNT', facet_row='ST01Q01', hover_data=['PV1MATH'], trendline='ols', title=\"Renda ESCS vs Dispositivos HOMEPOS\")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_shap_summary(shap_values, X_test: pd.DataFrame, features: list):
    st.subheader(\"SHAP Global Fase 4\")
    fig = shap.summary_plot(shap_values[1], X_test, feature_names=features, show=False, plot_type=\"beeswarm\")
    st.pyplot(fig)

def plot_shap_waterfall(shap_values_instance, feature_names, prediction):
    st.subheader(\"SHAP Waterfall Predicao\")
    fig = shap.waterfall_plot(shap.Explanation(values=shap_values_instance, base_values=0, data=np.zeros(len(feature_names)), feature_names=feature_names), show=False)
    st.pyplot(fig)

def plot_cluster_profiles(centers: np.ndarray, archetypes: list, features: list):
    centers_df = pd.DataFrame(centers, columns=features, index=archetypes)
    st.subheader(\"Medias Caracteristicas Clusters\")
    st.dataframe(centers_df.round(3), use_container_width=True)
    fig = px.bar(centers_df.reset_index().melt(id_vars='index'), x='variable', y='value', color='index', title=\"Medias Clusters por Feature\", barmode='group')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_cluster_scatter(X_scaled: np.ndarray, labels: np.ndarray, archetypes: list):
    fig = px.scatter(x=X_scaled[:,0], y=X_scaled[:,1], color=labels.astype(str), title=\"Scatter Clusters Dims 1-2\")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_shap_feature_importance(shap_values, X_test, features):
    st.subheader(\"Importancia SHAP Top Features Fase 4\")
    feature_imp = np.abs(shap_values[1]).mean(0)
    imp_df = pd.DataFrame({'feature': features, 'imp': feature_imp}).sort_values('imp', ascending=True)
    highlight = [f for f in ['ST29Q01', 'IC004Q01', 'HOMEPOS'] if f in features]
    fig = px.bar(imp_df.tail(8), x='imp', y='feature', orientation='h', title=\"Top SHAP abs mean\", color='feature')
    if highlight:
        fig.update_traces(marker_color='red' if any(h in imp_df['feature'] for h in highlight) else 'blue')
    st.plotly_chart(fig)
    if highlight:
        st.info(f\"Highlight: {', '.join(highlight)} (repeticao negativa, internet positiva).\")


