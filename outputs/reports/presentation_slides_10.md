# Apresentação (10 slides) — Análise e Resultados (PISA 2022 | Resiliência Criativa)

> **Formato**: cada seção abaixo é um slide.
> **Importante**: ao longo do texto, variáveis aparecem em formato `NOME_ORIGINAL` (mantidas no idioma do dataset) e entre colchetes aparece a **tradução/descrição** (glossário). Isso garante que qualquer pessoa entenda sem precisar do dicionário do PISA.

---

## Slide 1 — Contexto (PISA 2022 e criatividade)
- O PISA 2022 inclui o domínio **Pensamento Criativo** para medir processos de geração/avaliação/melhoria de ideias.
- A pesquisa operacionaliza um fenômeno educacional: **Resiliência Criativa** = **criatividade alta** em **condição socioeconômica vulnerável**.

---

## Slide 2 — Dados (o que foi usado no pipeline)
- Fonte: microdado processado `pisa_brasil_estudo_limpo.csv`.
- Amostra final: **3834** estudantes.
- Variáveis centrais (sempre com glossário):
  - `ESCS` [Índice socioeconômico, status econômico/social/cultural]
  - `CRT_SCORE` [Escore de Pensamento Criativo (criatividade)]
  - `ICTRES` [Recursos/tecnologia educacional em casa]
  - `HOMEPOS` [Recursos domésticos/ambiente de suporte]
  - `HISCED` [Proxy socioeducacional (nível educacional do contexto)]

---

## Slide 3 — Limpeza e qualidade dos dados
- Auditoria inicial:
  - Duplicatas (linhas inteiras): **0**.
  - Missingness: elevada em diversas variáveis `CR*` [itens/indicadores de criatividade].
- Implicação para resultados:
  - seleção de features e estabilidade das explicações (XAI) exigem cautela.

**Figuras (recomendadas)**
- `outputs/figures/data_quality.png`
- `outputs/figures/missing_values.png`

---

## Slide 4 — Target (Resiliência Criativa) e classe rara
- Definições testadas: **A/B/C/D** (quartis/percentis e combinação composta).
- Definição principal: **Target A**
  - `ESCS ≤ Q1` e `CRT_SCORE ≥ Q3`
- Prevalência do Target A:
  - **0,0427752** (164/3834) → **classe rara**.

---

## Slide 5 — Perfilização latente por clusterização (heterogeneidade)
- Abordagem person-centered: busca **perfis latentes** via clusterização.
- PCA para compactar o espaço:
  - **58 componentes** retidas.
- Melhor solução:
  - **agglomerative_ward** com **k=2**.
- Associação ao Target A:
  - Cluster 1: **risk_relative ≈ 1,367** e **odds_ratio ≈ 1,389**.

**Figuras (recomendadas)**
- `outputs/figures/clustering/scree_plot.png`
- `outputs/figures/clustering/cumulative_variance.png`
- `outputs/figures/clustering/cluster_radar.png`

---

## Slide 6 — Modelagem de Machine Learning (Target A)
- Tarefa: classificação binária para prever o Target A (classe rara).
- Modelo vencedor (no artefato de modelagem): **`xgboost_tuned`**.
- Métrica principal: **ROC-AUC**.

---

## Slide 7 — Desempenho e limiar (o que os números mostram)
- `xgboost_tuned`:
  - ROC-AUC (CV) = **0,9236**
  - ROC-AUC (holdout) = **0,9313**
- Classe rara → F1 é sensível ao limiar:
  - F1 = **0,2051** (limiar 0.50)
  - F1 = **0,3548** (limiar otimizado **0,110**)

---

## Slide 8 — XAI (SHAP): quais variáveis importam mais?
- Explicação global por SHAP (`shap_tree`).
- Top eixos/variáveis (do artefato):
  - `HISCED` [proxy socioeducacional] (maior relevância)
  - `ICTRES` [recursos tecnológicos/educacionais em casa]
  - `HOMEPOS` [recursos domésticos/ambiente de suporte]
  - Variáveis `CR*` [itens/indicadores de criatividade/engajamento em tarefas]

**Figura (recomendada)**
- `outputs/figures/shap/shap_top20.png`

---

## Slide 9 — Fairness e Robustez (credibilidade dos resultados)
- Fairness:
  - fairness por grupos sensíveis (variáveis sensíveis reportadas no projeto incluem `ST004D01T`, `Grupo_ESCS`, `HISCED`).
  - há subpopulações com **prevalência positiva 0** → métricas como **ROC-AUC = nan** nesses recortes.
- Robustez:
  - Brier score = **0,0228**
  - roc_auc bootstrap mean = **0,973985**
  - CI = **[0,966172, 0,980896]**

**Figura (recomendada)**
- `outputs/figures/robustness/calibration_curve.png`

---

## Slide 10 — Conclusões (o que os resultados sustentam)
- O Target A (classe rara) permite investigar **Resiliência Criativa** como interseção entre vulnerabilidade (`ESCS`) e criatividade (`CRT_SCORE`).
- Há **heterogeneidade**: clusterização com `k=2` revela subgrupo com associação mais forte ao Target A.
- O modelo **aprende sinal** preditivo (ROC-AUC alto), mas a métrica de classe rara (F1) depende criticamente do limiar.
- A explicabilidade (SHAP) aponta um eixo socioeducacional dominante (`HISCED`), com complementos tecnológicos (`ICTRES`), domésticos (`HOMEPOS`) e indicadores `CR*`.
- Fairness/robustez reforçam a credibilidade, porém limitam inferências quando subgrupos são muito raros (ROC-AUC `nan`).

---

# Glossário rápido (variáveis do projeto)
> Use este bloco como “glossário” opcional no final da apresentação.

- `ESCS` — Índice socioeconômico (proxy de vulnerabilidade)
- `CRT_SCORE` — Escore de Pensamento Criativo (proxy de criatividade)
- `HISCED` — Proxy socioeducacional (nível educacional/conhecimento contextual)
- `ICTRES` — Recursos/tecnologia educacional em casa
- `HOMEPOS` — Recursos domésticos/ambiente de suporte
- `CR*` — Itens/indicadores do domínio de criatividade e/ou variáveis derivadas do comportamento/engajamento nas tarefas
- `ST004D01T` — Variável sensível usada na auditoria de fairness (mantida como nome original; detalhes completos em `outputs/reports/fairness_report.md`)
- `Grupo_ESCS` — Agrupamento/recorte relacionado ao ESCS usado na auditoria de fairness

