# Script de Slides — Análise e Resultados (PISA 2022 | Resiliência Criativa)

> **Uso**: copie/cole cada seção em seu software de apresentação (PowerPoint/Google Slides/Marp/Pitch) como *um slide por bloco*. 
> **Obs.**: este texto foi preparado para ser autoexplicativo e para guiar fala durante a apresentação. Sempre que você quiser incluir uma figura, use o caminho do arquivo indicado.

---

## Slide 1 — Título
**Preditores da Resiliência Criativa em Contextos de Vulnerabilidade Socioeconômica**

- Pipeline reproduzível de Learning Analytics + EDM
- Machine Learning + XAI (SHAP)
- Auditoria de Fairness e Robustez

**Dados**: PISA 2022 (Brasil) — microdado `pisa_brasil_estudo_limpo.csv`

---

## Slide 2 — Problema e Motivação
- Desigualdade socioeconômica afeta oportunidades e resultados educacionais.
- O PISA 2022 introduz **Pensamento Criativo**, permitindo estudar criatividade em larga escala.
- **Ideia central**: há estudantes com **criatividade elevada mesmo em vulnerabilidade**.
- Objetivo: operacionalizar essa condição como **Resiliência Criativa** e estudar padrões preditivos explicáveis.

---

## Slide 3 — Construto e Operacionalização (Target)
- Resiliência Criativa é operacionalizada por combinação entre:
  - **Vulnerabilidade socioeconômica**: `ESCS`
  - **Criatividade**: `CRT_SCORE`
- Foram avaliadas definições alternativas de target: **A/B/C/D**.
- **Target principal (A)**: estudantes com **ESCS ≤ Q1** e **CRT_SCORE ≥ Q3**.
- **Prevalência** do Target A: **0,0427752** (164/3834) — classe rara.

> Figura opcional (se quiser mostrar prevalência por definição): use `outputs/tables/targets_definitions_prevalence.csv` (não é figura).

---

## Slide 4 — Pipeline (Visão Geral)
1. Auditoria inicial de dados (duplicatas/missingness)
2. Gate de data leakage (proxy-based gate)
3. Construção e comparação de targets A/B/C/D
4. Análise exploratória (EDA)
5. Perfilização person-centered (clusterização)
6. Modelagem supervisionada para Target A
7. XAI (SHAP)
8. Fairness por grupos sensíveis
9. Robustez (bootstrap/calibração)

---

## Slide 5 — Dados: Origem e Escopo
- Dataset: `pisa_brasil_estudo_limpo.csv`
- Linhas: **3834**; colunas: **1275**
- Principais variáveis:
  - `CRT_SCORE` (criatividade)
  - `ESCS` e proxy/derivados (vulnerabilidade)
  - Recursos tecnológicos: `ICTRES`
  - Recursos domésticos: `HOMEPOS`
  - Indicador socioeducacional proxy: `HISCED`
  - Pesos: `W_FSTUWT`

---

## Slide 6 — Limpeza e Auditoria Inicial
**O que foi checado**
- Duplicatas: **0 linhas duplicadas**
- Missingness: alta em diversas variáveis `CR*` (itens específicos)

**Como isso impacta o pipeline**
- Requer seleção criteriosa de features e cuidado com instabilidade.

**Figura recomendada**
- Qualidade/dados: `outputs/figures/data_quality.png`
- Missingness: `outputs/figures/missing_values.png`

---

## Slide 7 — Controle de Data Leakage (Gate)
- Proxy de target usada para gate: **`CRT_SCORE`**.
- A auditoria produz lista de candidatos de *leakage*.
- Alguns candidatos aparecem com severidade alta (ex.: nomes/variáveis com relação direta ou proxy do target).

**Ponto-chave para falar**
- A meta é evitar que o modelo aprenda **o desfecho “por construção”** em vez de aprender padrões educacionais.

---

## Slide 8 — Construção e Comparação dos Targets
- Definições avaliadas: A/B/C/D (quartis/percentis e combinação composta).
- Estatísticas relevantes:
  - Target A: prevalência **0,0427752**
  - Target B/C/D: prevalências distintas (classe pode variar em raridade)

**Mensagem interpretativa**
- “Resiliência” depende da operacionalização: isso afeta o que é detectável e explicável.

---

## Slide 9 — EDA: Padrões Iniciais
- Missingness e distribuições dos itens `CR*` variam substancialmente.
- Comparação resilient vs não resilient sugere diferenças em variáveis ligadas a:
  - tecnologia/acesso (`ICTRES` e proxies)
  - recursos domésticos (`HOMEPOS`)
  - indicadores de criatividade (`CR*`)

**Figuras recomendadas**
- `outputs/figures/eda/hist_ESCS.png`
- `outputs/figures/eda/hist_HOMEPOS.png`

---

## Slide 10 — Perfilização Latente (Clusterização)
- Abordagem person-centered para explorar heterogeneidade.
- Redução dimensional (PCA): **58 componentes** (do artefato de clustering).
- Algoritmos testados: múltiplos (inclui variantes não supervisionadas).
- Solução final: **agglomerative_ward com k=2**.

**Figura recomendada (estabilidade/seleção)**
- Scree plot: `outputs/figures/clustering/scree_plot.png`
- Variância acumulada: `outputs/figures/clustering/cumulative_variance.png`

---

## Slide 11 — Clusterization: Perfis e Interpretação
- Existem **2 perfis** (clusters): 0 e 1.
- Cluster 1 tem prevalência maior de Resiliência Criativa (Target A).

**Figuras recomendadas**
- Silhouette/estrutura: `outputs/figures/clusterer/clusterer_silhouette.png` *(se usar; ver pasta)*
- Distribuição/associação com resiliência:
  - `outputs/figures/clustering/cluster_resilience_distribution.png`
- Radar do perfil:
  - `outputs/figures/clustering/cluster_radar.png`

---

## Slide 12 — Resiliência Criativa: Ranking de Diferenças
- Comparação resilient vs não resilient indica variáveis com maior diferença de médias.
- Top diferenças são tipicamente dominadas por variáveis `CR*` (itens de criatividade), além de contextos específicos.

**Figura recomendada**
- Radar/heatmap dos perfis resilientes:
  - `outputs/figures/resilient_profile/resilient_profile_heatmap.png`
  - `outputs/figures/resilient_profile/resilient_profile_radar.png`

---

## Slide 13 — Modelos de Machine Learning (Target A)
- Tarefas: classificação binária (Target A — classe rara).
- Algoritmos avaliados: regressão logística e métodos baseados em árvores.
- Métrica principal (comparação): **ROC-AUC**.

> Mensagem para falar: classe rara afeta F1 e sensibilidade ao limiar.

---

## Slide 14 — Desempenho: Modelo Vencedor
- Melhor modelo: **`xgboost_tuned`**.
- Métricas:
  - ROC-AUC (CV) = **0,9236**
  - ROC-AUC (holdout) = **0,9313**
- F1:
  - F1 = **0,2051** (limiar 0.50)
  - F1 = **0,3548** (limiar otimizado 0.110)

**Figura opcional**
- Se existir gráfico de calibração, usar (ver robustness): `outputs/figures/robustness/calibration_curve.png`

---

## Slide 15 — XAI: Como o Modelo Explica
- Técnica: SHAP (`shap_tree`)
- Objetivo: identificar variáveis com maior contribuição média na predição.

**Top variáveis por SHAP (mensagem principal)**
- `HISCED` (mais relevante)
- `ICTRES`
- `HOMEPOS`
- várias variáveis `CR*` associadas aos padrões de criatividade/engajamento

**Figura recomendada**
- Top 20 SHAP: `outputs/figures/shap/shap_top20.png`

---

## Slide 16 — Categorias de Fatores (para interpretação)
- Agrupamento interpretativo dos fatores (baseado nas famílias de variáveis do projeto):
  - **Socioeconômicos/socioeducacionais**: p.ex. `HISCED`, `ESCS` proxies
  - **Tecnológicos**: p.ex. `ICTRES`
  - **Familiares/domésticos**: p.ex. `HOMEPOS`
  - **Criatividade/itens CR***: variáveis `CR*`

> Observação: se você quiser categorizar estritamente, use `outputs/reports/variable_catalog.md` como fonte.

---

## Slide 17 — Fairness: Auditoria de Equidade
- A auditoria considera variáveis sensíveis (por configuração/artefatos).
- Métricas reportadas incluem recall/erro por grupo.
- Ponto crítico: alguns grupos podem ter **prevalência positiva 0**, gerando **ROC-AUC = nan** para esses recortes.

**Mensagem para falar**
- Fairness em classe rara exige leitura cuidadosa das definições e métricas.

---

## Slide 18 — Robustez e Calibração
- Robustez: bootstrap e análise de estabilidade.
- Calibração: gráfico de calibração probabilística.

**Métricas reportadas (robustez do melhor modelo)**
- Brier score = **0,0228**
- roc_auc bootstrap mean = **0,973985**
- CI = **[0,966172, 0,980896]**

**Figura recomendada**
- `outputs/figures/robustness/calibration_curve.png`

---

## Slide 19 — Integração dos Resultados (Resumo Científico)
- Target A define o construto como criatividade alta sob vulnerabilidade.
- Clusterização sugere heterogeneidade: existem subgrupos com diferentes taxas relativas.
- XAI aponta um eixo socioeducacional dominante e eixos complementares tecnológicos e domésticos.
- Fairness e robustez reduzem risco de conclusões instáveis.

---

## Slide 20 — Conclusões e Implicações
- Resiliência Criativa é um fenômeno **raro** (classe rara impacta métricas).
- Padrões aprendidos são consistentes com mecanismos educacionais multivariados:
  - condições socioeducacionais (proxy via `HISCED`)
  - acesso/uso tecnológico (`ICTRES`)
  - recursos domésticos (`HOMEPOS`)
  - variáveis de criatividade (`CR*`)

---

## Slide 21 — Limitações (para fechar com rigor)
- Dados observacionais: sem causalidade.
- Risco de proxies no alvo (leakage gate reduz, mas não elimina todos riscos).
- Classe rara e subgrupos com prevalência 0 impactam fairness.
- Missingness alta em variáveis `CR*` pode afetar interpretabilidade.

---

## Slide 22 — Trabalhos Futuros
- Replicar em outras janelas/ciclos do PISA.
- Explorar causalidade/estratégias quase-experimentais (quando possível).
- Métodos adicionais de equidade para lidar com prevalência 0.
- Expandir explicações para nível de estudante e perfis.

---

## Slide 23 — Artefatos Reprodutíveis
- Relatórios: `outputs/reports/*` (clusterer, modeling, shap, fairness, robustness)
- Tabelas: `outputs/tables/*`
- Dashboard (se quiser demo):
  - imagens do dashboard:
    - `outputs/figures/dashboard/dashboard_desktop.png`
    - `outputs/figures/dashboard/dashboard_mobile_pw.png`

---

## Slide 24 — Perguntas / Discussão
- Quais componentes do ambiente parecem “permitir” criatividade sob vulnerabilidade?
- Como traduzir XAI em intervenção pedagógica?
- Como operacionalizar fairness quando o target é extremamente raro?

