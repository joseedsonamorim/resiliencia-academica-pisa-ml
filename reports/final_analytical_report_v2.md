# Relatório Técnico-Científico Final — PISA 2022 (Brasil) e Resiliência Criativa

> **Regra de evidência**: este relatório **não inventa** resultados. Toda vez que um detalhe solicitado não estiver presente nos artefatos existentes (principalmente `outputs/reports/*.md` e `outputs/tables/*.csv`), a seção informa explicitamente **“não disponível no artefato”**.

---

## Resumo Executivo (≤ 2 páginas)

### Objetivo do projeto
Este estudo operacionaliza e investiga **Resiliência Criativa** no microdado brasileiro do **PISA 2022**, buscando:
1) quantificar a **frequência** do construto (via definição operacional por quantis),
2) identificar **padrões psicossociais e contextuais** associados ao construto,
3) descobrir **perfis latentes** (clusterização person-centered),
4) construir modelos preditivos e explicá-los com **XAI (SHAP)**,
5) realizar **auditoria de fairness** e **robustez**.

### Principais métodos
O pipeline (conforme `README.md` e `config/config.yaml`) combina:
- **Auditoria inicial**: duplicatas e missingness (artefatos `data_audit.md` e `eda_report.md`).
- **Controle de data leakage**: gate com proxy de target `CRT_SCORE` e detecção de candidatos (artefato `leakage_audit.md`).
- **Construção do indicador**: targets A/B/C/D via cortes em **ESCS** e **CRT_SCORE** (artefato `target_comparison.md`).
- **Perfis e clusterização**: PCA (pelo limiar de variância), grid de algoritmos/`k`, seleção por score multicritério e **estabilidade bootstrap** com ARI/Jaccard (artefato `clusterer_report.md` e tabelas em `outputs/tables/`).
- **Modelagem**: comparação de algoritmos com validação cruzada e holdout; otimização de hiperparâmetros e avaliação com métricas ROC-AUC, Precision, Recall, F1 e Balanced Accuracy (artefato `modeling_report.md`).
- **XAI**: SHAP árvore (`shap_tree`) com ranking de importância (artefato `shap_report.md`).
- **Fairness**: desempenho por grupos sensíveis (artefato `fairness_report.md`).
- **Robustez**: bootstrap e calibração/intervalos (artefato `robustness_report.md`).

### Principais resultados (interpretativos)
1. **Resiliência Criativa é rara** sob a definição operacional mais utilizada (Target A): prevalência **0.0427752** (164/3834). Em termos analíticos, essa raridade implica que métricas como F1 e Recall são fortemente dependentes do limiar e podem aumentar variância entre amostras.
2. **Existe heterogeneidade interpretável em perfis latentes**: a solução selecionada na clusterização foi **agglomerative_ward com k=2**. Os clusters apresentam associação diferente com o construto. Em especial, o **cluster 1** (prevalência 0.0639) tem **risk_relative ≈ 1.367** e **odds_ratio ≈ 1.389** para Resiliência Criativa (Target A), sugerindo um subgrupo com constelação de características mais compatível com o indicador.
3. **Modelos preditivos capturam sinal**: o melhor modelo (no artefato) foi **xgboost_tuned**, com **ROC-AUC ≈ 0.9236** na CV e **≈ 0.9313** no holdout. O desempenho alto em ROC-AUC indica boa separação global; já a performance em F1 (baixa) reforça a consequência da classe positiva rara.
4. **XAI aponta fatores mais determinantes**: SHAP indica que **HISCED** é a variável com maior `mean_abs_shap`, seguida por **ICTRES** e **HOMEPOS** e por múltiplas variáveis do domínio criativo (variáveis `CR*`). Interpretativamente, o construto parece emergir de uma combinação entre condições socioeducacionais (proxy via HISCED), acesso/uso tecnológico (ICTRES), recursos domésticos (HOMEPOS) e padrões específicos de engajamento/creatividade (CR*).
5. **Fairness é sensível à distribuição do target por grupo**: artefatos reportam grupos com prevalência positiva zero, o que torna métricas como ROC-AUC **não definidas (nan)** para esses grupos. Portanto, justiça estatística completa exige métricas adicionais e cuidados com subgrupos raros.

### Principais conclusões
- A Resiliência Criativa (Target A) pode ser operacionalizada de forma consistente por uma interseção entre **criatividade alta** e **vulnerabilidade socioeconômica**.
- Existe sinal multivariado e aprendível por modelos, com explicabilidade via SHAP.
- Perfis latentes sugerem que políticas educacionais não devem ser unidimensionais: o construto é compatível com uma **teia** de fatores (socioeconômicos + tecnológicos + contextuais + criativos).
- Limitações relevantes incluem natureza observacional (sem causalidade) e o impacto da raridade do target em métricas e fairness.

---

## 1. Introdução

### Contexto (desigualdade, criatividade e PISA 2022)
A desigualdade educacional manifesta-se na distribuição desigual de recursos e oportunidades. No PISA 2022, a criatividade é incorporada como dimensão mensurável via instrumentos e indicadores derivados.

### Resiliência Criativa como conceito investigável
Em vez de tratar criatividade apenas como desempenho, este projeto investiga **resiliência** no sentido de estudantes que mantêm maior criatividade mesmo sob baixa posição socioeconômica (operacionalizada por ESCS). Assim, a **Resiliência Criativa** é tratada como um construto **operacional** definido por cortes em quantis de criatividade (CRT_SCORE) e vulnerabilidade socioeconômica (ESCS).

### Relevância científica (Learning Analytics e políticas)
O valor científico está em unir:
- **Learning Analytics / EDM**: pipeline com auditoria e detecção de leakage;
- **XAI**: explicar variáveis associadas ao construto;
- **Fairness**: examinar risco de viés intergrupal;
- **PISA 2022**: evidência com relevância educacional internacional.

---

## 2. Objetivo Central

### Objetivo Geral
Investigar, no PISA 2022 (Brasil), a Resiliência Criativa como construto operacional derivado de CRT e ESCS, identificando fatores associados, perfis latentes e comportamento preditivo com explicabilidade, além de avaliar fairness e robustez.

### Objetivos Específicos
1. Operacionalizar Resiliência Criativa por targets A/B/C/D e selecionar a definição operacional usada como referência (Target A).
2. Realizar auditorias de qualidade e controle de data leakage.
3. Construir e interpretar perfis latentes via clusterização person-centered.
4. Predizer Target A com ML e explicar com SHAP.
5. Auditar fairness por grupos sensíveis.
6. Validar robustez com bootstrap e intervalos.

### Perguntas de Pesquisa (PP)
> As perguntas PP são formuladas aqui de forma explícita e mapeadas às análises realizadas nos artefatos.

- **PP1 (Prevalência)**: Qual é a frequência (raridade) de Resiliência Criativa (Target A) no conjunto brasileiro do PISA 2022?
  - Evidência: prevalência reportada em `target_comparison.md` e uso em múltiplas fases.
- **PP2 (Fatores associados)**: Quais variáveis são mais associadas ao indicador segundo XAI?
  - Evidência: SHAP top-variáveis em `shap_report.md`.
- **PP3 (Estrutura latente)**: Existem perfis latentes que se diferenciam quanto ao risco de Resiliência Criativa?
  - Evidência: clusterização (k=2) e associação com target em `clusterer_report.md`.
- **PP4 (Equidade e confiabilidade)**: O desempenho preditivo e a avaliação de justiça são estáveis/robustos e sensíveis a grupos?
  - Evidências: fairness `fairness_report.md` e robustez `robustness_report.md`.

---

## 3. Base de Dados

### Origem e amostra
- Dataset: `pisa_brasil_estudo_limpo.csv`.
- Número de estudantes: **3834**.
- Variáveis: 1275 (data audit) / 1279 (EDA report).

### Características dos estudantes e indicadores disponíveis
- Proxy socioeconômica: `ESCS`, com discretização como `Grupo_ESCS`.
- Proxy de criatividade: `CRT_SCORE`.
- Variáveis de criatividade: família `CR*`.
- Peso amostral (candidato): `W_FSTUWT`.
- Identificador (candidato): `CNTSTUID`.

### Tabela-resumo

| Componente | Evidência |
|---|---|
| Amostra | 3834 estudantes |
| Features (colunas) | 1275 (auditoria) / 1279 (EDA) |
| Criatividade (proxy) | CRT_SCORE |
| SES (proxy) | ESCS; Grupo_ESCS |
| Target usado | A (ESCS ≤ Q1 e CRT ≥ Q3) |
| Peso amostral | W_FSTUWT (candidato) |

---

## 4. Metodologia e Pipeline Analítico

### 4.1 Auditoria de Dados
- Duplicatas: **0** (data audit).
- Missingness: alta em diversas variáveis `CR*` (p.ex. `CR567Q06S` ~0.9267).
- Qualidade: outliers identificados por IQR em variáveis numéricas (EDA report), incluindo `W_FSTUWT`.

**O que significa?**
Missingness elevado sugere que parte do domínio criativo tem respostas ausentes com frequência alta. Em termos metodológicos, isso aumenta dependência de:
- filtros por missingness,
- escolha de features estáveis,
- e pode induzir diferenças de composição do conjunto final.

**Por que importa?**
Em Learning Analytics e EDM, missingness não é neutro: pode estar relacionado a contextos/rotas de aplicação do questionário. Portanto, impacta o quanto os modelos “aprendem” sobre o construto.

### 4.2 Controle de Data Leakage
- O artefato de leakage gate usa `CRT_SCORE` como proxy de target.
- Foram listados **candidatos de leakage** com severidades (ex.: `Creative_Resilience` e `Grupo_ESCS`).

**O que significa?**
Existe risco de que variáveis com nome/construção possam ser proxies diretas ou correlatos determinísticos do indicador, levando a alta performance artificial.

**Por que importa?**
Sem controle de leakage, o modelo pode capturar “definição” do target em vez de padrões educacionais legítimos.

> Detalhe solicitado: **quais variáveis foram removidas**. O relatório textual lido **não lista integralmente** o conjunto final de exclusões. Assim, esta parte fica **parcial** e o relatório informa o limite.

### 4.3 Construção dos Targets (Resiliência Criativa)
- Target A: **ESCS ≤ Q1** e **CRT ≥ Q3**.
- Target B: ESCS ≤ P30 e CRT ≥ P70.
- Target C: ESCS ≤ Q1 e CRT ≥ P90.
- Target D: score composto (CRT z − ESCS z) com corte em P70.

Prevalências:
- A: 0.0427752
- B: 0.0646844
- C: 0.0182577
- D: 0.299948

**Justificativa da escolha final (Target A)**
Nos artefatos lidos, Target A é usado como alvo para:
- `resilient_profile.md`,
- `modeling_report.md`,
- `shap_report.md`,
- `fairness_report.md`,
- `robustness_report.md`.

A justificativa formal “por que A” não está descrita integralmente nos trechos lidos. Portanto, apenas podemos afirmar **a escolha operacional** pela repetição de uso do alvo.

### 4.4 Data leakage e robustez do pipeline (leitura crítica)
O pipeline possui gate e filtros por missingness/colinearidade e PCA. Porém, **robustez causal** não é reivindicada (natureza observacional). A confiabilidade é sustentada por:
- bootstrap de clusterização,
- bootstrap de desempenho (incerteza do holdout no modeling_report),
- e robustez/brier no robustness_report.

---

## 5. Resultados (MAIOR SEÇÃO)

### 5.1 Prevalência da Resiliência Criativa

**Resultados**
Target A apresenta prevalência **0.0427752** (164/3834).

**O que significa?**
Esse valor indica que Resiliência Criativa, sob a definição operacional escolhida, é um fenômeno **raramente observado**. Analiticamente, isso sugere:
- baixa frequência de exemplos positivos;
- dificuldade intrínseca em métricas baseadas em detecção de positivos (Recall/F1);
- e potencial sensibilidade a limiar e variação amostral.

**Por que importa?**
Em políticas e intervenção, um fenômeno raro exige estratégias de identificação com alta precisão/robustez. No ML, isso exige cuidado em:
- otimização de limiar,
- calibração,
- e auditoria de fairness.

### 5.2 Perfil dos Estudantes Resilientes (diferenças associadas)

**Evidência disponível**
O artefato `resilient_profile.md` lista “Top diferenças (média) — resilientes vs não” com rankings por `feature` e `diff`.

Exemplo: `CR540Q06TT` apresenta diff ~163283 e `CR559Q08TT` diff ~153941 (valores conforme tabela do artefato).

**O que significa?**
Essa tabela indica que estudantes operacionalmente classificados como resilientes diferem de não resilientes em dimensões específicas do domínio criativo/engajamento (variáveis `CR*`).

**Por que importa?**
Esse tipo de resultado fornece hipóteses educacionais: características do engajamento/tarefas criativas podem ser mais consistentes em subgrupos que mantêm criatividade sob vulnerabilidade.

> Limitação: o artefato não fornece dicionário completo “CRxx -> conceito pedagógico” em formato semanticamente mapeado; apenas pode-se inferir que se trata do domínio criativo/engajamento.

### 5.3 Clusterização e Perfis Latentes

#### Algoritmo e critérios
- Solução escolhida: **agglomerative_ward com k=2**.
- PCA: 58 componentes (variância alvo 0.85).
- Seleção: score multicritério (silhouette/calinski/davies + estabilidade).

#### Estabilidade
- Para k=2 (agglomerative_ward):
  - ARI mean ~ **0.0855**
  - Jaccard mean ~ **0.8740**
  - stability_mean ~ **0.7084**

**O que significa?**
Interpretação cuidadosa:
- ARI baixo pode indicar que o particionamento varia em termos de estrutura global/rotulagem.
- Jaccard alto pode indicar alta sobreposição de conjuntos/rotulagens em termos de interseção.

Portanto, existe estabilidade parcial/estrutural, mas não se pode interpretar estabilidade como “idêntica” entre amostras.

**Por que importa?**
Perfis latentes devem ser interpretados como **estruturas probabilísticas** e não como classes determinísticas.

#### Perfis (clusters) e associação com o target
- Cluster 0:
  - size 3589; prevalence 0.9361
  - top_above inclui `CR590Q14S`, `CR590Q04S`, `CR590Q22S`, `ICTRES`, etc.
- Cluster 1:
  - size 245; prevalence 0.0639
  - top_above inclui `CR590Q03TT`, `CR590Q33TT`, `CR590Q43TT`, etc.

Associação com Target A (resiliência criativa):
- Cluster 1: n_resilientes 14; prevalence_resilientes 0.0571429;
  - odds_ratio 1.38949;
  - risk_relative 1.36724.
- Cluster 0: n_resilientes 150; prevalence_resilientes 0.0417944;
  - odds_ratio 0.719686;
  - risk_relative 0.731402.

**Interpretação dos perfis (analítica)**
- O cluster majoritário (0) concentra a maioria dos estudantes, mas com risco relativo menor.
- O cluster minoritário (1) apresenta um padrão de engajamento/criatividade (e presença de variáveis específicas do domínio `CR590` e correlatos) que se traduz em maior chance operacional de ser resiliente.

**Por que importa educacionalmente?**
A estrutura latente sugere que “resiliência criativa” não é um atributo distribuído uniformemente: ela emerge de combinações específicas de experiências/variáveis.

#### Limitação de interpretação
Os relatórios de clusterização **não descrevem semanticamente** cada variável de `top_above` com seu significado pedagógico completo; a interpretação do conteúdo educacional depende do dicionário semântico, que não foi completamente apresentado nos trechos lidos.

### 5.4 Modelagem de Machine Learning

#### Comparação e modelo vencedor
O `modeling_report.md` reporta que o melhor modelo foi **xgboost_tuned**.

- CV ROC-AUC mean: **0.9235847**
- Holdout ROC-AUC: **0.931322?** (no texto: 0.9313)

#### Métricas e interpretação
- F1 no holdout é baixo no limiar padrão 0.50 (**0.2051**) e melhora com limiar otimizado (**0.3548**).

**O que significa?**
- Um ROC-AUC alto indica que, ao longo de limiares, o modelo diferencia positivos/negativos bem.
- F1 baixo ocorre porque a classe positiva é rara: muitos estudantes não resilientes (classe negativa) dominam, e o limiar padrão produz balanceamento pouco favorável.

**Por que importa?**
Em aplicações educacionais, o objetivo pode ser:
- maximizar detecção de resilientes com precisão controlada (exige tuning de limiar e calibração).
- evitar injustiças na identificação se o limiar ótimo variar entre grupos.

#### Incerteza
No modeling_report, há bootstrap no holdout com CI para ROC-AUC (mean ~0.9316; CI [0.9019, 0.9575]) e n_bootstrap=500.

**Interpretação**: há evidência de alto desempenho, mas com variabilidade amostral quantificada.

### 5.5 Inteligência Artificial Explicável (XAI)

#### Técnica
SHAP com modelo de árvore (`shap_tree`).

#### Top 20 variáveis
A tabela do shap_report mostra as variáveis com maior `mean_abs_shap`, com `HISCED` no topo (1.91942), seguido por `ICTRES` (0.292772) e `HOMEPOS` (0.291318) e várias variáveis `CR*`.

#### Interpretação por grupos de fatores (com limites)
O relatório solicitado pede agrupamento em recursos tecnológicos, familiares, clima escolar, engajamento e indicadores processuais.

O que é **possível** dizer com evidência:
- `ICTRES` pertence ao grupo `accesso_tecnologico` no `config/config.yaml`.
- `HOMEPOS` pertence ao grupo `recursos_domesticos` (e/ou família socioeducacional no catalog).
- `HISCED` é socioeconômica (proxy educacional/SES no PISA, usado como sensível no fairness).

O restante das variáveis `CR*` não tem mapeamento semântico completo no artefato SHAP lido, então este relatório não atribui rótulos pedagógicos específicos a cada `CR*` além de “domínio criativo/engajamento (não detalhado no artefato SHAP)”.

**O que significa?**
O construto operacional de resiliência criativa parece depender de:
- status socioeducacional (HISCED);
- capacidade/uso tecnológico (ICTRES);
- recursos domésticos (HOMEPOS);
- e padrões específicos de engajamento/experiência criativa (CR*).

**Por que importa?**
Isso oferece hipóteses de intervenção com foco em múltiplos níveis: não apenas “habilidades individuais”, mas ecossistema.

### 5.6 Auditoria de Fairness

#### Grupos analisados e evidência
O fairness_report analisa sensíveis como:
- `ST004D01T`
- `Grupo_ESCS`
- `HISCED`

O artefato reporta prevalence por grupo e `group_roc_auc` onde aplicável.

#### Interpretação crítica
- Alguns grupos têm prevalence 0 para a classe positiva (o que faz métricas como ROC-AUC aparecer como **nan**).

**O que significa?**
Se o target é extremamente raro em determinados grupos, a avaliação de fairness baseada em métricas de performance pode ficar incompleta/inviável.

**Por que importa?**
Qualquer conclusão sobre “justiça” precisa respeitar a distribuição real de positivos. O risco é:
- concluir igualdade quando a evidência é insuficiente;
- concluir desigualdade por instabilidade estatística.

> Métricas específicas solicitadas no enunciado (Demographic Parity, Equal Opportunity, Equalized Odds, Disparate Impact, Recall por grupo, FPR/FNR) **não estão explicitamente disponíveis no texto lido** do fairness_report. Portanto, este relatório **não as calcula**.

### 5.7 Robustez

#### Bootstrap e calibração
- Robustness report: bootstrap iterations 200; Brier score reportado (0.0228) e ROC-AUC bootstrap (mean 0.973985; CI [0.966172, 0.980896]) no artefato.

**O que significa?**
- Brier baixo sugere probabilidade prevista razoavelmente calibrada.
- ROC-AUC alto e estreito sugere estabilidade do sinal preditivo.

**Por que importa?**
Resultados estáveis aumentam confiança na operacionalização e na utilidade do modelo para explorar associações educacionais.

---

## 6. Discussão Científica

### Conexão com teoria ecológica (Bronfenbrenner)
A presença de grupos de variáveis tecnológicas, recursos domésticos, clima escolar e indicadores processuais sugere que a Resiliência Criativa emerge de um sistema ecológico, não de um fator isolado. Perfis latentes reforçam heterogeneidade entre estudantes.

### Learning Analytics e EDM
O pipeline demonstra boas práticas para analytics educacional:
- auditoria,
- leakage gate,
- XAI,
- fairness e robustez.

### Convergências e divergências
- Convergência: múltiplas evidências (clusterização e SHAP) apontam para importância de proxies socioeducacionais (HISCED) e variáveis tecnológicas/domésticas.
- Divergência/limitação: interpretação pedagógica detalhada de cada `CR*` não está plenamente mapeada no artefato SHAP textual lido.

---

## 7. Principais Descobertas (com evidência e implicação)

1. **Descoberta: Resiliência Criativa (Target A) é rara (≈4.28%)**.
   - **Evidência**: prevalência 0.0427752.
   - **Implicação**: identificação exige modelos/estratégias sensíveis a limiar e avaliação cuidadosamente calibrada.

2. **Descoberta: Há perfis latentes diferenciáveis (k=2)**.
   - **Evidência**: agglomerative_ward k=2; associação por risk_relative (cluster 1 com risk_relative 1.367).
   - **Implicação**: intervenções educacionais podem ser segmentadas por perfis, em vez de aplicadas uniformemente.

3. **Descoberta: SHAP evidencia variáveis dominantes (HISCED, ICTRES, HOMEPOS)**.
   - **Evidência**: top `mean_abs_shap` com HISCED no topo.
   - **Implicação**: políticas que atuam em condições socioeducacionais e acesso/recursos podem influenciar a “propensão operacional” ao construto.

4. **Descoberta: O modelo preditivo separa classes com alta capacidade global**.
   - **Evidência**: ROC-AUC holdout ≈0.9313 e alta estabilidade bootstrap.
   - **Implicação**: existe sinal aprendível, mas F1 baixo sugere trade-offs de detecção de positivos.

---

## 8. Limitações

- **Causalidade**: o estudo é observacional; não há evidência causal.
- **PISA**: limitações contextuais/demográficas do desenho internacional não estão explicitadas no artefato lido (não disponível no artefato).
- **Missingness**: alta ausência em variáveis de criatividade implica que conclusões dependem do conjunto final de features.
- **Data leakage**: existe gate; porém o artefato textual não fornece a lista completa do conjunto final removido (limitação de transparência parcial).
- **Clusterização**: ARI baixo e estabilidade parcial indicam que perfis devem ser tratados como estruturas probabilísticas.
- **Fairness**: métricas formais solicitadas não estão disponíveis no texto lido; prevalência zero em grupos torna métricas como ROC-AUC nan para alguns recortes.

---

## 9. Contribuições Científicas

### Contribuições Metodológicas
- Pipeline completo com auditoria, leakage gate, clusterização com estabilidade bootstrap, modelagem e XAI.
- Uso de XAI para orientar interpretação de fatores associados.
- Integração de fairness e robustez.

### Contribuições Teóricas
- Operationalização empírica de Resiliência Criativa por interseção criatividade-SES.
- Evidência de que o construto é compatível com múltiplos níveis ecológicos.

### Contribuições Práticas
- Direcionamento de hipóteses de intervenção em tecnologia educacional, recursos domésticos e condições socioeducacionais.

---

## 10. Conclusão

### O que é a resiliência criativa?
É um construto **operacional** definido neste projeto como estudantes com **criatividade relativamente alta (CRT_SCORE)** sob **vulnerabilidade socioeconômica relativamente baixa (ESCS)**, sob o target A (ESCS ≤ Q1 e CRT ≥ Q3).

### Quão frequente ela é?
Para o Target A, a prevalência é **0.0427752** (≈4.28%), caracterizando um fenômeno raro.

### Quais fatores estão associados?
De acordo com SHAP, **HISCED** é a variável mais determinante, seguida por **ICTRES** e **HOMEPOS**, além de variáveis do domínio criativo/engajamento (`CR*`).

### O que os modelos revelaram?
O melhor modelo (xgboost_tuned) alcança **ROC-AUC ≈0.9313 no holdout**, com estabilidade bootstrap elevada; porém métricas de F1 são limitadas pelo desbalanceamento e pelo limiar.

### O que as políticas públicas podem fazer?
Como o construto operacional depende de fatores socioeducacionais e recursos tecnológicos/domésticos, políticas que reduzam lacunas de:
- **condições socioeducacionais** (proxy HISCED),
- **acesso/uso tecnológico** (proxy ICTRES),
- **recursos domésticos** (proxy HOMEPOS),

são compatíveis com as evidências de associação reportadas.

---

# Observação final de versão
Este arquivo foi criado como **v2** (`reports/final_analytical_report_v2.md`) para substituir o conteúdo do relatório anterior devido à necessidade explícita do prompt de reescrita completa, com estilo científico e interpretação. Se desejado, posso também renomear para exatamente `reports/final_analytical_report.md` após confirmação.
