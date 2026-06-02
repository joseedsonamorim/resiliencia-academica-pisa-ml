"""Main Streamlit app for Resiliência Criativa dashboard."""

import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))

from src.config import get_config


# ---------- Helpers to load real pipeline outputs ----------
PROJECT_ROOT = Path(__file__).parent


def _load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


def _img(path: Path):
    if not path.exists():
        return None
    return str(path)


def load_pipeline_summary():
    return _load_json(PROJECT_ROOT / "outputs" / "reports" / "pipeline_summary.json")


def load_bootstrap_ci():
    return _load_json(PROJECT_ROOT / "outputs" / "metrics" / "bootstrap_ci.json")


def load_resilient_profile_table():
    return _load_csv(PROJECT_ROOT / "outputs" / "tables" / "resilient_profile.csv")


def load_permutation_importance_table():
    return _load_csv(PROJECT_ROOT / "outputs" / "tables" / "permutation_importance.csv")



# Page config
st.set_page_config(
    page_title="Resiliência Criativa - PISA 2022",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load config
config = get_config()


# Sidebar
with st.sidebar:
    st.title("Navegação")
    page = st.radio(
        "Selecione uma página:",
        [
            "Home",
            "Introdução",
            "Base de Dados",
            "Auditoria de Dados",
            "Construção do Target",
            "Engenharia de Features",
            "EDA",
            "Clustering",
            "Modelagem",
            "Fairness",
            "XAI (SHAP)",
            "Robustez",
            "Auditoria Científica",
            "Resultados",
            "Exportação",
        ],
    )

    st.divider()
    st.caption("Plataforma Científica - Resiliência Criativa")
    st.caption("PISA 2022 Brasil")


# HOME PAGE
if page == "Home":
    st.title("Resiliência Criativa - PISA 2022 Brasil")
    st.markdown("---")

    summary = load_pipeline_summary() or {}
    ds = (summary.get("data") or {}) if isinstance(summary, dict) else {}
    # fallback: alguns arquivos salvam em outputs/reports/pipeline_summary.json como {timestamp, description, data:{...}}
    if "dataset" in ds:
        ds = ds.get("dataset") or {}

    # Compat: pipeline_summary.json do projeto atual guarda em summary['data']? (mesmo assim deixamos fallback)
    if not ds and isinstance(summary, dict):
        ds = summary.get("dataset") or {}

    target_dist = ds.get("target_distribution") or {}
    n0 = target_dist.get("0")
    n1 = target_dist.get("1")
    total = ds.get("total_samples")
    features_final = ds.get("features_final")

    def _fmt_int(x):
        return "N/A" if x is None else f"{int(x):,}".replace(",", ".")

    def _fmt_pct(x):
        return "N/A" if x is None else f"{x*100:.2f}%"

    n_resil = n1
    pct_resil = None
    if total and n_resil is not None and total != 0:
        pct_resil = float(n_resil) / float(total)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estudantes", _fmt_int(total), "Brasil")
    with col2:
        st.metric("Resilientes", _fmt_int(n_resil), _fmt_pct(pct_resil) if pct_resil is not None else "")
    with col3:
        st.metric("Features", "N/A" if features_final is None else str(features_final), "após seleção")


    st.markdown("---")

    st.subheader("O que é Resiliência Criativa?")
    st.write(
        """
**Resiliência Criativa** refere-se a estudantes que apresentam:
- Alto desempenho em Creative Thinking (CRT_SCORE ≥ Q3)
- Contexto socioeconômico desfavorável (ESCS ≤ Q1)

Esses estudantes alcançam excelência acadêmica apesar das adversidades socioeconômicas.
        """
    )

    st.subheader("Pipeline de Análise")
    st.write(
        """
1. **Dados**: PISA 2022 - 3.834 estudantes brasileiros
2. **Preprocessing**: Exclusão leakage → Imputação → Scaling → Feature Selection (RFE)
3. **Modelos**: Logistic Regression, Random Forest, XGBoost, LightGBM
4. **Validação**: Stratified K-Fold 5x + Bootstrap IC95%
5. **Interpretabilidade**: SHAP + Análise de Fairness
6. **Auditoria**: Data leakage, Overfitting, Reproducibilidade
        """
    )

    st.subheader("Arquivo de Dados")

    if summary and ("data" in summary or "dataset" in (summary.get("data") or {})):
        # pipeline_summary.json atual: summary has {timestamp, description, data:{dataset,...}}
        ds_full = summary.get("data") or summary.get("dataset") or {}
        if "dataset" in ds_full:
            ds_full = ds_full.get("dataset")

        n_total = ds_full.get("total_samples")
        n_feat_ini = ds_full.get("features_initial")
        n_feat_fin = ds_full.get("features_final")
        td = (ds_full.get("target_distribution") or {})
        n1 = td.get("1")
        pct = (float(n1)/float(n_total))*100 if n_total else None

        st.info(
            f"""
**Origem**: Respondentes do questionário socioeconômico PISA 2022

**Dimensões**:
- {_fmt_int(n_total)} estudantes (após filtragem)
- {_fmt_int(n_feat_ini)} variáveis (bruto)
- {_fmt_int(n_feat_fin)} features (após preprocessing)

**Target**: Creative_Resilience (0/1 - binário, {('N/A' if pct is None else f'{pct:.2f}%')} positivos)
            """
        )
    else:
        st.info(
            """
**Origem**: Respondentes do questionário socioeconômico PISA 2022

**Dimensões**:
- (Execute `python3 run_all.py` para gerar outputs reais)

**Target**: Creative_Resilience
            """
        )


    st.subheader("Objetivos")
    st.markdown(
        """
- Identificar perfis de estudantes criativamente resilientes
- Construir modelo preditivo acurado
- Analisar fairness entre grupos demográficos
- Garantir reproducibilidade científica
- Gerar relatório publicável em periódicos científicos
        """
    )


# INTRODUCTION PAGE
elif page == "Introdução":
    st.title("Introdução ao Estudo")

    st.subheader("Contexto do PISA 2022")
    st.write(
        """
O PISA (Programme for International Student Assessment) avalia competências em leitura,
matemática e now **Creative Thinking** em estudantes de 15 anos.

Este estudo foca em como fatores socioeconômicos influenciam a criatividade em estudantes brasileiros.
        """
    )

    st.subheader("Perguntas de Pesquisa")
    st.markdown(
        """
1. Quais características socioeconômicas predizem resiliência criativa?
2. Existem disparidades de fairness entre gêneros?
3. Qual modelo melhor identifica estudantes resilientes?
4. Quais features são mais importantes para a resiliência?
        """
    )

    st.subheader("Definição de Resiliência")
    st.code(
        """
Resiliência Criativa:
- ESCS ≤ Q1 (Primeiro quartil - socioeconômico desfavorável)
- AND
- CRT_SCORE ≥ Q3 (Terceiro quartil - alto desempenho criativo)
        """
    )

    st.subheader("Hipóteses")
    st.write(
        """
- H1: Features socioeconômicas/demográficas predizem resiliência
- H2: Tecnologia em casa correlaciona com criatividade
- H3: Educação parental influencia criatividade
- H4: Não há disparidades injustas entre gêneros
        """
    )


# DATA OVERVIEW PAGE
elif page == "Base de Dados":
    st.title("Base de Dados")

    summary = load_pipeline_summary() or {}
    ds = (summary.get("data") or {}) if isinstance(summary, dict) else {}
    if "dataset" in ds:
        ds = ds.get("dataset") or {}

    td = ds.get("target_distribution") or {}
    n_total = ds.get("total_samples")
    n0 = td.get("0")
    n1 = td.get("1")
    n_feat_ini = ds.get("features_initial")
    n_feat_fin = ds.get("features_final")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Observações", _fmt_int(n_total))
    col2.metric("Variáveis (bruto)", _fmt_int(n_feat_ini))
    col3.metric("Features (final)", _fmt_int(n_feat_fin))
    pct = (float(n1)/float(n_total))*100 if (n_total and n1 is not None) else None
    col4.metric("Target (positivos)", f"{_fmt_int(n1)} ({('N/A' if pct is None else f'{pct:.2f}%')})")

    st.subheader("Distribuição do Target")

    target_data = {
        "Não-resilientes": int(n0) if n0 is not None else 0,
        "Resilientes": int(n1) if n1 is not None else 0,
    }
    st.bar_chart(target_data)


    st.subheader("Variáveis Principais")
    st.write(
        """
As variáveis mais relevantes podem ser visualizadas a partir da saída real de **resilient_profile** e de tabelas de importância.
        """
    )


    # Mostra top variáveis do perfil resiliente (tabela real)
    prof = load_resilient_profile_table()
    if prof is not None and len(prof) > 0:
        topn = 10
        cols = list(prof.columns)
        # espera colunas: variable, mean_resilient, mean_non_resilient, cohen_d
        if "variable" in cols:
            prof_table = prof.copy()
            if "cohen_d" in cols:
                prof_table = prof_table.sort_values("cohen_d", ascending=False)
            st.dataframe(prof_table.head(topn), use_container_width=True)
        else:
            st.dataframe(prof.head(topn), use_container_width=True)
    else:
        main_vars = pd.DataFrame(
            {
                "Variável": ["ESCS", "CRT_SCORE", "HOMEPOS", "ICTRES", "HISCED", "ST004D01T"],
                "Tipo": ["Numérica", "Numérica", "Numérica", "Numérica", "Numérica", "Categórica"],
                "Descrição": [
                    "Status socioeconômico",
                    "Score criatividade",
                    "Recursos casa",
                    "Recursos TIC",
                    "Educação parental",
                    "Gênero",
                ],
            }
        )
        st.dataframe(main_vars, use_container_width=True)



# AUDIT PAGE
elif page == "Auditoria de Dados":
    st.title("Auditoria de Dados")

    st.subheader("Checklist de Validação")

    checks = {
        "Duplicatas": "Nenhuma duplicata",
        "Leakage": "5 features removidas (CRT_SCORE, Status, Grupo_ESCS, CNTSTUID, W_FSTUWT)",
        "Target": "Distribuição válida (4.28% positivos)",
        "Missing": "0% após impute",
        "Types": "Tipos validados",
        "Imbalance": "1:22.4 (tratado com SMOTE)",
    }

    for _, result in checks.items():
        st.write(result)

    st.subheader("Variáveis com Leakage (Removidas)")
    st.warning(
        """
NUNCA devem ser usadas como features:
- CRT_SCORE: Score direto de criatividade (componente do target!)
- Status: Categorização do target
- Grupo_ESCS: Derivada do ESCS (usado na def. target)
- CNTSTUID: ID do estudante
- W_FSTUWT: Peso amostral
        """
    )


# TARGET CONSTRUCTION PAGE
elif page == "Construção do Target":
    st.title("Construção do Target - Creative Resilience")

    st.subheader("Lógica de Construção")
    st.code(
        """
Creative_Resilience = 1 if:
    ESCS ≤ Q1 (≤ -1.6970)  # Desfavorável
    AND
    CRT_SCORE ≥ Q3 (≥ 0.4925)  # Alto desempenho
else:
    Creative_Resilience = 0
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Q1 ESCS", "-1.6970")
        st.metric("Q1 Count", "958 estudantes")

    with col2:
        st.metric("Q3 CRT_SCORE", "0.4925")
        st.metric("Q3 Count", "958 estudantes")

    st.subheader("Resultado Final")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Resilientes (1)", "164", "+4.28%")
    with col2:
        st.metric("Não-resilientes (0)", "3.670", "+95.72%")

    st.warning("Dataset altamente desbalanceado (1:22.4) - Usar SMOTE + stratified CV")


# FEATURE ENGINEERING PAGE
elif page == "Engenharia de Features":
    st.title("Engenharia de Features")

    st.subheader("Processo de Seleção")
    st.markdown(
        """
1. **Exclusão de Leakage** (features proibidas removidas)
2. **Remoção >80% missing**
3. **Scaling (StandardScaler)**
4. **RFE (Recursive Feature Elimination)**
5. **SMOTE apenas em treino**
        """
    )

    st.subheader("Importância (Permutation Importance - dados reais)")
    perm = load_permutation_importance_table()
    if perm is None or len(perm) == 0:
        st.warning("Arquivo outputs/tables/permutation_importance.csv não encontrado ou vazio.")
    else:
        # colunas esperadas: feature, importance_mean, importance_std, percentual_importancia
        # ordena pelos maiores mean
        col_candidates = ["importance_mean", "feature", "percentual_importancia"]
        for c in col_candidates:
            if c not in perm.columns:
                st.error(f"Coluna esperada '{c}' não encontrada em permutation_importance.csv")
                st.dataframe(perm.head(20), use_container_width=True)
                break
        else:
            perm_sorted = perm.sort_values("importance_mean", ascending=False)
            topn = min(15, len(perm_sorted))
            show_cols = [c for c in ["feature", "importance_mean", "importance_std", "percentual_importancia"] if c in perm_sorted.columns]
            st.dataframe(perm_sorted[show_cols].head(topn), use_container_width=True)

    st.markdown("---")
    st.subheader("Figura - Permutation Importance (real)")
    fig = _img(PROJECT_ROOT / "outputs" / "figures" / "permutation_importance.png")
    if fig:
        st.image(fig, use_container_width=True)
    else:
        st.warning("Figura outputs/figures/permutation_importance.png não encontrada.")



# EDA PAGE
elif page == "EDA":
    st.title("Análise Exploratória de Dados")

    st.info("Nesta versão do dashboard, gráficos de EDA são exibidos a partir de outputs persistidos do pipeline. Se algum artefato não existir, a página mostra aviso.")

    st.subheader("Distribuição do Target (real)")
    summary = load_pipeline_summary() or {}
    ds = (summary.get("data") or {}) if isinstance(summary, dict) else {}
    if "dataset" in ds:
        ds = ds.get("dataset") or {}
    td = ds.get("target_distribution") or {}

    n0 = td.get("0")
    n1 = td.get("1")
    if n0 is not None and n1 is not None:
        st.bar_chart({"Não-resilientes": int(n0), "Resilientes": int(n1)})
    else:
        st.warning("Não foi possível carregar target_distribution de pipeline_summary.json")

    st.subheader("Figuras disponíveis (pipeline outputs)")
    # mostra figuras existentes relacionadas ao perfil resiliente
    for p in [
        ("Perfil Resiliente - Radar", PROJECT_ROOT / "outputs" / "figures" / "resilient_profile_radar.png"),
        ("Perfil Resiliente - Heatmap", PROJECT_ROOT / "outputs" / "figures" / "resilient_profile_heatmap.png"),
    ]:
        title, path = p
        img = _img(path)
        if img:
            st.image(img, caption=title, use_container_width=True)
        else:
            st.warning(f"Figura não encontrada: {path}")



# CLUSTERING PAGE
elif page == "Clustering":
    st.title("Análise de Clustering")
    st.info("Nesta base, o pipeline gera relatórios/figuras apenas para alguns módulos. Se o clustering não tiver artefatos em outputs, a página indica.")

    # Tenta carregar arquivos complementares caso existam
    cluster_md = PROJECT_ROOT / "outputs" / "reports" / "cluster_interpretation.md"
    if cluster_md.exists():
        st.markdown(cluster_md.read_text(encoding="utf-8"))
    else:
        st.warning("outputs/reports/cluster_interpretation.md não encontrado. Execute a geração de complementos (run_complements.py) se existir no projeto.")



# MODELING PAGE
elif page == "Modelagem":
    st.title("Modelagem Preditiva")

    summary = load_pipeline_summary() or {}
    ds = (summary.get("data") or {}) if isinstance(summary, dict) else {}
    if "dataset" in ds:
        ds = ds.get("dataset") or {}

    st.subheader("Resultados (real) - modelo melhor")

    models = summary.get("models") or (summary.get("data") or {}).get("models") if isinstance(summary, dict) else {}
    if not models and (summary.get("data") or {}).get("models"):
        models = (summary.get("data") or {}).get("models")

    best_model = None
    evals = {}
    if isinstance(summary, dict):
        if "models" in summary:
            best_model = summary["models"].get("best_model")
            evals = summary["models"].get("evaluation") or {}
        elif "data" in summary and isinstance(summary["data"], dict):
            # fallback
            best_model = (summary["data"] or {}).get("models", {}).get("best_model")

    if evals and best_model and best_model in evals:

        m = evals.get(best_model) or {}
        df = pd.DataFrame([{k: v for k, v in m.items()}])
        st.dataframe(df, use_container_width=True)

        st.subheader("Melhor Modelo")
        st.success(f"{best_model} (baseado em pipeline_summary.json)")
    else:
        st.warning("Não foi possível carregar avaliação completa por modelo de pipeline_summary.json.")



# FAIRNESS PAGE
elif page == "Fairness":
    st.title("Análise de Fairness")
    st.info("Carregando resultados salvos do pipeline (se existentes).")

    fairness_json = PROJECT_ROOT / "outputs" / "metrics" / "fairness_results.json"
    if fairness_json.exists():
        fair = _load_json(fairness_json)
        st.json(fair)
    else:
        st.warning("outputs/metrics/fairness_results.json não encontrado. Execute novamente a pipeline para produzir outputs de fairness.")



# XAI PAGE
elif page == "XAI (SHAP)":
    st.title("Explainability (SHAP)")
    st.info("Carregando outputs de SHAP (se existentes).")

    shap_json = PROJECT_ROOT / "outputs" / "metrics" / "shap_importance.json"
    if shap_json.exists():
        shap = _load_json(shap_json)
        st.json(shap)
    else:
        st.warning("outputs/metrics/shap_importance.json não encontrado.")



# ROBUSTNESS PAGE
elif page == "Robustez":
    st.title("Robustez Estatística")

    st.subheader("Bootstrap IC95% (real)")
    st.info("Intervalo de confiança via bootstrap resampling (armazenado em outputs/metrics/bootstrap_ci.json)")

    boot = load_bootstrap_ci()
    if boot is None:
        st.warning("outputs/metrics/bootstrap_ci.json não encontrado ou inválido.")
    else:
        rows = []
        for metric, vals in boot.get("data", boot).items():
            # esperado: {mean,std,median,ci_lower,ci_upper}
            rows.append({
                "Métrica": metric,
                "Mean": vals.get("mean"),
                "IC Lower": vals.get("ci_lower"),
                "IC Upper": vals.get("ci_upper"),
                "Brier (se existir)": vals.get("brier_score") if metric == "brier_score" else None,
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)



# AUDIT PAGE
elif page == "Auditoria Científica":
    st.title("Auditoria Científica")

    st.subheader("Status (sem dados exemplo)")
    st.info("Se a auditoria tiver sido gerada em outputs/reports, ela será mostrada aqui.")

    # mostra dataset inventory e pipeline summary como evidência
    inv = PROJECT_ROOT / "outputs" / "reports" / "dataset_inventory.md"
    if inv.exists():
        st.markdown(inv.read_text(encoding="utf-8"))
    else:
        st.warning("outputs/reports/dataset_inventory.md não encontrado.")

    pipe_md = PROJECT_ROOT / "outputs" / "reports" / "pipeline_summary.json"
    if pipe_md.exists():
        st.subheader("pipeline_summary.json")
        st.json(_load_json(pipe_md))
    else:
        st.warning("outputs/reports/pipeline_summary.json não encontrado.")



# RESULTS PAGE
elif page == "Resultados":
    st.title("Resultados Principais")

    summary = load_pipeline_summary() or {}
    ds = (summary.get("data") or {}).get("dataset") if isinstance(summary, dict) else None

    post = (summary.get("postprocessing") or {}) if isinstance(summary, dict) else {}
    thr = post.get("optimal_threshold")
    boot = post.get("bootstrap_ci")
    brier = None
    if isinstance(boot, dict) and "brier_score" in boot:
        brier = boot["brier_score"].get("mean")

    st.subheader("Síntese (real) - pipeline_summary.json")

    target = ds.get("target_distribution") if isinstance(ds, dict) else {}
    n1 = target.get("1")
    total = ds.get("total_samples") if isinstance(ds, dict) else None
    pct = (float(n1)/float(total))*100 if (n1 is not None and total) else None

    bullets = []
    bullets.append(f"**Melhor modelo**: {summary.get('models', {}).get('best_model', 'N/A')}")
    bullets.append(f"**Threshold ótimo**: {thr if thr is not None else 'N/A'}")
    bullets.append(f"**Calibração (Brier, mean)**: {brier if brier is not None else 'N/A'}")
    bullets.append(f"**Resilientes**: {n1 if n1 is not None else 'N/A'} ({('N/A' if pct is None else f'{pct:.2f}%')})")

    st.markdown("\n".join(["1. "+b for b in bullets]))



# EXPORT PAGE
elif page == "Exportação":
    st.title("Exportação de Resultados")

    st.subheader("Baixar artefatos reais (somente arquivos já gerados pelo pipeline)")

    artifacts = {
        "pipeline_summary.json": PROJECT_ROOT / "outputs" / "reports" / "pipeline_summary.json",
        "dataset_inventory.md": PROJECT_ROOT / "outputs" / "reports" / "dataset_inventory.md",
        "bootstrap_ci.json": PROJECT_ROOT / "outputs" / "metrics" / "bootstrap_ci.json",
        "resilient_profile.csv": PROJECT_ROOT / "outputs" / "tables" / "resilient_profile.csv",
        "permutation_importance.csv": PROJECT_ROOT / "outputs" / "tables" / "permutation_importance.csv",
    }

    available = {k: v for k, v in artifacts.items() if v.exists()}
    if not available:
        st.warning("Nenhum artefato encontrado em outputs/. Execute a pipeline (run_all.py) para gerar outputs.")
    else:
        choice = st.selectbox("Selecione arquivo para download:", list(available.keys()))
        path = available[choice]
        data = path.read_bytes()
        st.download_button(
            label=f"Baixar {choice}",
            data=data,
            file_name=choice,
            mime="application/octet-stream",
        )



st.divider()
st.caption("Plataforma Científica © 2026 - Resiliência Criativa PISA 2022")

