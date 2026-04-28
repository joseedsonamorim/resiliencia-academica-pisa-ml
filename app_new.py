# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Módulo 3: Painel Interativo Streamlit - Design Moderno Apple-Style.

Layout limpo e minimalista com paleta branco/cinza claro. Interfaces intuitivas
e bem documentadas para exploração de dados de resiliência criativa PISA 2022.
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
# CSS DESIGN - Apple Style: Branco, Cinza Claro, Clean
# ============================================================================
st.markdown("""
<style>
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif; }

/* Container base */
.block-container { 
    padding-top: 2.5rem; 
    padding-left: 3rem; 
    padding-right: 3rem; 
    padding-bottom: 2rem; 
    background-color: #FFFFFF; 
}

/* Tipografia principal */
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

/* Divider */
.divider-thin {
    border: 0;
    height: 1px;
    background: #E8EAED;
    margin: 1.5rem 0;
}

/* Cards KPI - Estilo Apple */
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

/* Secções de análise */
.analysis-box {
    font-size: 1rem;
    line-height: 1.7;
    background-color: #F5F6F8;
    padding: 1.5rem;
    border-left: 3px solid #0A66C2;
    border-radius: 6px;
    margin: 1.5rem 0;
}

.analysis-box strong {
    color: #000000;
    font-weight: 600;
}

/* Tipografia dos headings */
h1, h2, h3, h4, h5, h6 {
    color: #000000;
    letter-spacing: -0.3px;
    font-weight: 600;
}

h2 { font-size: 1.8rem; margin-bottom: 1rem; }
h3 { font-size: 1.3rem; margin-bottom: 0.8rem; }

/* Expandables */
.streamlit-expanderHeader {
    background-color: #F5F6F8;
    border: 1px solid #E8EAED;
    border-radius: 8px;
}

/* Botões */
.stButton > button {
    background-color: #0A66C2;
    color: #FFFFFF;
    border: 1px solid #0A66C2;
    border-radius: 8px;
    font-weight: 500;
    letter-spacing: -0.2px;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s ease;
    height: auto;
}

.stButton > button:hover {
    background-color: #004BA0;
    border-color: #004BA0;
    box-shadow: 0 2px 8px rgba(10, 102, 194, 0.2);
}

/* Alert boxes */
.stSuccess { background-color: #F0FDF4; border: 1px solid #BBEF63; border-radius: 8px; }
.stInfo { background-color: #F0F4F8; border: 1px solid #D1DCE8; border-radius: 8px; }
.stWarning { background-color: #FFFBEB; border: 1px solid #FCD34D; border-radius: 8px; }
.stError { background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; }

/* Data frames */
.dataframe {
    border: 1px solid #E8EAED;
    border-radius: 8px;
}

/* Captions */
.streamlit-caption {
    color: #8B92A1;
    font-size: 0.9rem;
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

st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================
@st.cache_data(show_spinner="Carregando dados PISA 2022...")
def load_cached_data():
    return load_data()

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================
df_all = load_cached_data()
if df_all.empty:
    st.error("❌ Erro ao carregar dados. Verifique se o arquivo existe em data/")
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
    
    st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)
    
    st.markdown("## 🎯 Filtros de Dados")
    selected_countries = st.multiselect(
        "Países LATAM",
        LATAM_COUNTRIES,
        default=LATAM_COUNTRIES,
        help="Selecione quais países incluir na análise"
    )
    
    if not selected_countries:
        st.warning("Nenhum país selecionado. Usando todos os países.")
        selected_countries = LATAM_COUNTRIES
    
    sample_size = st.slider(
        "Amostra para gráficos",
        100, 2000, 1000,
        help="Tamanho da amostra para visualizações (não afeta treinamento)"
    )
    
    st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)
    
    st.markdown("## ⚙️ Executar Análises")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_kmeans = st.button(
            "🧩 K-Means",
            key="kmeans_btn",
            type="primary",
            use_container_width=True,
            help="Agrupa alunos resilientes em 4 perfis"
        )
    with col_btn2:
        run_rf = st.button(
            "🤖 Treinar",
            key="rf_btn",
            use_container_width=True,
            help="Treina modelo Random Forest + SHAP"
        )
    
    st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)
    
    st.markdown("## 📌 Sobre")
    st.markdown("**Mestrado em IA**  \nPPGIA UFRPE")
    st.caption("© 2026 Jose Edson Amorim Sebastiao")

# ============================================================================
# FILTRO DE DADOS
# ============================================================================
df_filtered = df_all[df_all['CNT'].isin(selected_countries)]

if df_filtered.empty:
    st.error("Filtro resultou em 0 registros. Verifique a seleção de países.")
    st.stop()

if len(df_filtered) > sample_size:
    df_view = df_filtered.sample(sample_size, random_state=42)
else:
    df_view = df_filtered.copy()

# ============================================================================
# KPIs - VISÃO GERAL
# ============================================================================
resil_criativo_rate = df_filtered['Resiliente_Criativo'].mean()
baseline_latam = df_all['Resiliente_Criativo'].mean()

st.markdown("### 📊 Visão Geral")

col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        label="Estudantes",
        value=f"{len(df_filtered):,}",
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    delta_text = f"{(resil_criativo_rate - baseline_latam)*100:+.1f}pp vs baseline"
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        label="Resilientes Criativos",
        value=f"{resil_criativo_rate:.1%}",
        delta=delta_text
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        label="Média ESCS",
        value=f"{df_filtered['ESCS'].mean():.2f}",
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_because_html=True)

with col4:
    st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
    st.metric(
        label="Média PV1CREA",
        value=f"{df_filtered['PV1CREA'].mean():.0f}",
        label_visibility="visible"
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)

# ============================================================================
# TABS - CONTEÚDO PRINCIPAL
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Exploração de Dados",
    "👥 Perfis Estudantes",
    "🎯 Modelo Preditivo",
    "📥 Exportar Resultados"
])

# ============================================================================
# TAB 1: EXPLORAÇÃO DE DADOS
# ============================================================================
with tab1:
    st.markdown("## Exploração de Dados (EDA)")
    st.markdown("""
Visualize a relação entre status socioeconômico (ESCS) e pensamento criativo (PV1CREA).
Identifique estudantes com resiliência criativa - aqueles que demonstram alta criatividade
apesar de desvantagens econômicas.
    """)
    
    st.markdown("### 🎓 O que é Resiliência Criativa?")
    with st.expander("Definição", expanded=False):
        st.markdown("""
Um estudante é considerado **resiliente criativo** quando:

- Está no **Q1 de ESCS** (25% mais vulneráveis economicamente) no seu país
- **E** no **Q4 de PV1CREA** (25% mais criativos) no seu país

Esta abordagem identifica superadores: estudantes que, apesar de desvantagens, demonstram
alto pensamento criativo e inovação.
        """)
    
    col_left, col_right = st.columns([2, 1], gap="large")
    
    with col_left:
        scatter = (
            alt.Chart(df_view)
            .mark_circle(size=100, opacity=0.65)
            .encode(
                x=alt.X(
                    'ESCS:Q',
                    title='Status Socioeconômico (ESCS)',
                    scale=alt.Scale(domain=[-3, 3])
                ),
                y=alt.Y(
                    'PV1CREA:Q',
                    title='Pensamento Criativo (PV1CREA)',
                    scale=alt.Scale(domain=[300, 700])
                ),
                color=alt.Color(
                    'Resiliente_Criativo:N',
                    scale=alt.Scale(
                        domain=['0', '1'],
                        range=['#D1D5DB', '#0A66C2']
                    ),
                    title='Tipo'
                ),
                tooltip=['CNT:N', 'ESCS:Q', 'PV1CREA:Q', 'HOMEPOS:Q', 'ST29Q01:N']
            )
            .properties(
                title="Socioeconômico vs Criatividade",
                width=700,
                height=450
            )
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)
    
    with col_right:
        hist_crea = (
            alt.Chart(df_view)
            .mark_bar(opacity=0.8, color='#0A66C2')
            .encode(
                x=alt.X('PV1CREA:Q', bin=alt.Bin(maxbins=30), title='PV1CREA'),
                y=alt.Y('count()', title='Frequência'),
                color=alt.value('#0A66C2'),
                tooltip=['count()']
            )
            .properties(title="Distribuição de Criatividade", height=450)
        )
        st.altair_chart(hist_crea, use_container_width=True)
    
    st.markdown("### 💡 Interpretação")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("""
**Pontos Azuis** = Resilientes Criativos
- Canto inferior-direito
- ESCS baixo + PV1CREA alto
- Superadores
        """)
    
    with col_b:
        st.markdown("""
**Pontos Cinzas** = Outros Estudantes
- Padrões diversos
- Passe o mouse para detalhes
- Contexto completo
        """)

# ============================================================================
# TAB 2: PERFIS (K-MEANS)
# ============================================================================
with tab2:
    st.markdown("## Perfis de Resiliência Criativa (K-Means)")
    st.markdown("""
O algoritmo K-Means agrupa estudantes resilientes criativos em **4 arquétipos** com base em
similaridades nas variáveis de contexto socioeconômico.
    """)
    
    st.markdown("### 📋 Como usar")
    st.markdown("""
1. Clique em **K-Means** na barra lateral
2. Aguarde o processamento e agrupamento
3. Visualize distribuição e características de cada perfil
    """)
    
    if run_kmeans:
        try:
            df_res, kmeans_model = kmeans_resilientes_criativos(df_filtered)
            st.session_state['df_clusters'] = df_res
            st.session_state['kmeans'] = kmeans_model
            st.success("✅ K-Means concluído! Veja os resultados abaixo.")
        except Exception as e:
            st.error(f"Erro ao executar K-Means: {e}")
    
    if 'df_clusters' in st.session_state:
        df_c = st.session_state['df_clusters']
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            counts = df_c['arquétipo'].value_counts()
            bar = (
                alt.Chart(counts.reset_index())
                .mark_bar(color='#0A66C2')
                .encode(
                    x=alt.X('arquétipo:N', sort='-y', title='Arquétipo'),
                    y=alt.Y('count()', title='Quantidade'),
                    color=alt.value('#0A66C2'),
                    tooltip=['count()']
                )
                .properties(title="Distribuição de Arquétipos", height=350)
            )
            st.altair_chart(bar, use_container_width=True)
        
        with col2:
            st.markdown("#### Características Médias")
            profile = df_c.groupby('arquétipo')[['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']].mean().round(3)
            st.dataframe(profile, use_container_width=True, hide_index=False)
        
        st.markdown("#### Exemplos de Estudantes")
        st.dataframe(
            df_c[['CNT', 'arquétipo', 'ESCS', 'PV1CREA', 'HOMEPOS']].head(15),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ℹ️ Execute K-Means para visualizar os perfis aqui.")

# ============================================================================
# TAB 3: MODELO PREDITIVO (RF + SHAP)
# ============================================================================
with tab3:
    st.markdown("## Modelo Preditivo (Random Forest + SHAP)")
    st.markdown("""
Treine um classificador Random Forest para prever resiliência criativa usando variáveis
de contexto. O SHAP explica quais variáveis mais influenciam as predições.
    """)
    
    st.markdown("### 📋 Como usar")
    st.markdown("""
1. Clique em **Treinar** na barra lateral
2. Aguarde o treinamento com validação cruzada
3. Visualize métricas, SHAP e resultados
    """)
    
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
            st.success("✅ Modelo treinado! Veja os resultados abaixo.")
        except Exception as e:
            st.error(f"Erro ao treinar modelo: {e}")
    
    if 'report' in st.session_state:
        report = st.session_state['report']
        
        st.markdown("### 📈 Desempenho do Modelo")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric("F1-Score (Teste)", f"{report.get('test_f1', 0):.3f}")
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
            st.metric("ROC-AUC (Teste)", f"{report.get('test_roc_auc', 0):.3f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col5:
            st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
            st.metric(
                "CV F1-Score",
                f"{report.get('cv_f1_mean', 0):.3f}",
                delta=f"±{report.get('cv_f1_std', 0):.3f}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if 'shap_vals' in st.session_state:
            st.markdown("### 🔍 Variáveis Mais Importantes (SHAP)")
            shap_vals = st.session_state['shap_vals']
            X_sample = st.session_state['X_train_bal'].head(200)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_vals, X_sample,
                feature_names=CREATIVITY_FEATURES_RF,
                show=False, max_display=10
            )
            plt.tight_layout()
            st.pyplot(plt.gcf())
            plt.close()
        
        with st.expander("Relatório Detalhado", expanded=False):
            st.markdown("#### Métricas por Classe")
            if 'test' in report and isinstance(report['test'], dict):
                def _fmt(v):
                    if isinstance(v, (int, float)):
                        return f"{float(v):.3f}"
                    return str(v)
                st.json({k: {sk: _fmt(v) for sk, v in sv.items()} if isinstance(sv, dict) else _fmt(sv) for k, sv in report['test'].items()})
    else:
        st.info("ℹ️ Execute o treinamento para visualizar os resultados.")

# ============================================================================
# TAB 4: EXPORTAR
# ============================================================================
with tab4:
    st.markdown("## Exportar Resultados")
    st.markdown("Baixe predições em CSV e salve/carregue modelos do cache.")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("### 📥 Predições em CSV")
        if st.button("🔽 Exportar Predições", key="export_csv", use_container_width=True):
            if 'model' in st.session_state:
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
                st.success("✅ CSV gerado!")
            else:
                st.warning("Treine o modelo primeiro.")
    
    with col2:
        st.markdown("### 💾 Cache de Modelos")
        col_s, col_l = st.columns(2)
        
        with col_s:
            if st.button("💾 Salvar", key="save_rf", use_container_width=True):
                if 'model' in st.session_state:
                    try:
                        ensure_models_dir()
                        save_model(st.session_state['model'], "rf_model")
                        st.success("✅ Salvo!")
                    except Exception as e:
                        st.error(f"Erro: {e}")
                else:
                    st.warning("Treine o modelo primeiro.")
        
        with col_l:
            if st.button("🔄 Carregar", key="load_rf", use_container_width=True):
                try:
                    ensure_models_dir()
                    model_loaded = load_model("rf_model")
                    if model_loaded:
                        st.session_state['model'] = model_loaded
                        st.success("✅ Carregado!")
                    else:
                        st.info("Nenhum modelo em cache.")
                except Exception as e:
                    st.error(f"Erro: {e}")
    
    st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)
    
    if 'report' in st.session_state:
        st.markdown("### 📊 Resumo do Treinamento")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Amostras (Treino)", len(st.session_state['X_train_bal']))
        with col2:
            st.metric("Amostras (Teste)", len(st.session_state['X_test']))
        with col3:
            st.metric("Features", len(CREATIVITY_FEATURES_RF))
        
        report = st.session_state['report']
        st.markdown("#### Métricas Finais")
        metrics_dict = {
            "F1-Score (Teste)": report.get('test_f1', 0),
            "Precisão (Teste)": report.get('test_precision', 0),
            "Recall (Teste)": report.get('test_recall', 0),
            "ROC-AUC (Teste)": report.get('test_roc_auc', 0),
            "F1-Score CV": report.get('cv_f1_mean', 0),
            "ROC-AUC CV": report.get('cv_roc_auc_mean', 0),
        }
        for name, val in metrics_dict.items():
            st.write(f"**{name}**: `{val:.3f}`")

# ============================================================================
# RODAPÉ
# ============================================================================
st.markdown('<hr class="divider-thin">', unsafe_allow_html=True)
st.markdown("""
<p class="app-helper" style="text-align: center; margin-top: 2rem;">
🧠 PPGIA UFRPE | Fases 2-4 KDD | © 2026 Jose Edson Amorim Sebastiao
</p>
""", unsafe_allow_html=True)
