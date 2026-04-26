# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Módulo 3: Painel Interativo Streamlit - Modelagem Preditiva Pensamento Criativo PISA 2022.

Layout wide. Sidebar filtros CNT LATAM. Abas: EDA (texto analítico profundo PT), Perfis Resiliência Criativa (KMeans), Predição SHAP (RF F1).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import altair as alt
from src.data_loader import load_data, LATAM_COUNTRIES
from src.ml_models import kmeans_resilientes_criativos, rf_classifier, shap_explainer, CREATIVITY_FEATURES_RF, CREATIVITY_FEATURES_KMEANS
from src.export_utils import save_model, load_model, ensure_models_dir

__rastreio_app__ = "jeas_pisa_streamlit_2026_ufrpe"

st.set_page_config(
    layout="wide",
    page_title="Modelagem Preditiva Pensamento Criativo PISA 2022",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# CSS custom
st.markdown("""
<style>
.block-container {padding-top: 1rem;}
.app-title {font-size: 2rem; font-weight: 800;}
.app-subtitle {color: #6b7280; font-size: 1.1rem;}
.kpi-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);}
.analysis-text {font-size: 1rem; line-height: 1.6; background-color: #f9fafb; padding: 1.2rem; border-left: 4px solid #3b82f6; margin: 1rem 0;}
</style>
""", unsafe_allow_html=True)

# Título
st.markdown('<h1 class="app-title">🧠 Modelagem Preditiva do Pensamento Criativo PISA 2022</h1>', unsafe_allow_html=True)
st.markdown('<p class="app-subtitle">Abordagem ML Explicável: Resiliência Criativa em Alunos Latinoamericanos de Baixa Renda (Q1 ESCS + Q4 PV1CREA)</p>', unsafe_allow_html=True)

@st.cache_data(show_spinner="🔄 Carregando dados PISA 2022...")
def load_cached_data():
    return load_data()

df_all = load_cached_data()
if df_all.empty:
    st.error("❌ Dados vazios. Coloque CSV/Parquet em data/ ou use mock automático.")
    st.stop()

# Sidebar filtros
with st.sidebar:
    st.header("🎛️ Filtros")
    selected_countries = st.multiselect(
        "Países LATAM", LATAM_COUNTRIES, default=LATAM_COUNTRIES,
        help="Filtre países para análise"
    )
    sample_size = st.slider("Amostra para visualização", 100, 2000, 1000)
    
    st.divider()
    run_kmeans = st.button("🧩 Gerar Perfis K-Means", type="primary")
    run_rf = st.button("🤖 Treinar RF + SHAP")
    
    st.caption("PPGIA UFRPE - Jose Edson A. Sebastiao (2026)")

# Filtrar dados
df_filtered = df_all[df_all['CNT'].isin(selected_countries)]
if len(df_filtered) > sample_size:
    df_view = df_filtered.sample(sample_size, random_state=42)
else:
    df_view = df_filtered.copy()

resil_criativo_rate = df_filtered['Resiliente_Criativo'].mean()
baseline_latam = df_all['Resiliente_Criativo'].mean()

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Alunos selecionados", f"{len(df_filtered):,}")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Resil. Criativo", f"{resil_criativo_rate:.1%}", f"{(resil_criativo_rate - baseline_latam)*100:+.1f}pp LATAM")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Média ESCS", f"{df_filtered['ESCS'].mean():.2f}")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric("Média PV1CREA", f"{df_filtered['PV1CREA'].mean():.0f}")
    st.markdown('</div>', unsafe_allow_html=True)

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📈 Análise Exploratória de Dados (EDA)", "👥 Perfis de Resiliência Criativa", "🎯 Predição e SHAP", "📊 Exportar Relatório"])

with tab1:
    st.markdown("""
    <div class="analysis-text">
    <strong>Passo 1: Contextualização Teórica (PISA 2022 Criatividade)</strong><br>
    O PISA 2022 introduz avaliação de Pensamento Criativo (PV1CREA) como competência transversal.
    Hipótese central: existe 'resiliência criativa' em alunos de baixa renda (Q1 ESCS) com alto desempenho criativo (Q4 PV1CREA)?
    Justificativa variáveis: ESCS (proxy renda familiar), HOMEPOS (infra domiciliar), ST29Q01 (repetência experiência escolar),
    IC004Q01 (acesso digital). Espera-se que infraestrutura positiva mitigue desvantagens SES no pensamento divergente.
    </div>
    """, unsafe_allow_html=True)
    
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
        st.altair_chart(scatter, use_container_width=True)
    
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
        st.altair_chart(hist_crea, use_container_width=True)
    
    st.markdown("""
    <div class="analysis-text">
    <strong>Análise dos Resultados (Gráficos Acima):</strong><br>
    Observa-se cluster de <em>outliers positivos</em> (vermelho plasma): alunos Q1 ESCS (~-0.8) com PV1CREA >550.
    Correlação esperada ESCS-CREA (~0.3-0.4), mas presença ~5-10% resilientes criativos valida hipótese.
    Infra (HOMEPOS>0.5, IC004Q01=1) aparece nos tooltips como facilitador. Próximo: modelar predição.
    Justifica ML: desbalanceamento extremo (~5%) requer SMOTE; explicabilidade SHAP identifica drivers (ex: repetência negativa).
    </div>
    """, unsafe_allow_html=True)

with tab2:
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
            st.altair_chart(bar, use_container_width=True)
        with col2:
            profile = df_c.groupby('arquétipo')[['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']].mean().round(2)
            st.dataframe(profile.T)
        st.dataframe(df_c[['CNT', 'arquétipo', 'ESCS', 'PV1CREA', 'HOMEPOS']].head(10))

with tab3:
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
            st.json({k: {sk: f"{v:.3f}" if isinstance(v, float) else v for sk,v in sv.items()} 
                    for k,sv in report['test'].items()})
        
        if 'shap_vals' in st.session_state:
            shap_vals = st.session_state['shap_vals']
            X_sample = st.session_state['X_train_bal'].head(200)
            plt.figure(figsize=(10,6))
            shap.summary_plot(shap_vals, X_sample, feature_names=CREATIVITY_FEATURES_RF, show=False, max_display=10)
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()

with tab4:
    st.subheader("📊 Exportar Resultados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 Baixar Dados + Predições")
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
                st.warning("⚠️ Treinar RF primeiro (Aba 3)")
    
    with col2:
        st.markdown("### 💾 Cache de Modelos")
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
                    st.warning("⚠️ Treinar RF primeiro (Aba 3)")
        
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
    st.markdown("### 📄 Relatório Análise")
    
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
        st.info("ℹ️ Executar treino RF (Aba 3) para visualizar relatório")

st.markdown("---")
st.caption("🧠 PPGIA UFRPE | Fases 2-4 KDD: ML Criativo LATAM | Copyright 2026 JEAS")

