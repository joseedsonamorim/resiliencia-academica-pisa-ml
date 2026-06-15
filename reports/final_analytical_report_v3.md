# Relatório Técnico-Científico Final (v3) — PISA 2022 (Brasil) e Resiliência Criativa

> **Regra de evidência**: nenhuma afirmação numérica é inventada. Quando um detalhe solicitado não estiver explicitamente presente nos artefatos existentes (especialmente `outputs/reports/*.md` e tabelas em `outputs/tables/*.csv`), a seção declara **“não disponível no artefato”**.

---

## Resumo Executivo (≤ 2 páginas)

### Objetivo do projeto
Investigar, no microdado brasileiro do **PISA 2022**, a **Resiliência Criativa** como construto operacional definido pela interseção entre **criatividade (CRT_SCORE)** e **vulnerabilidade socioeconômica (ESCS)**, com ênfase em:
- prevalência do construto;
- identificação de **mecanismos multivariados** associados;
- descoberta de **perfis latentes**;
- previsão com **Machine Learning** e **explicabilidade (SHAP)**;
- avaliação de **fairness** e **robustez**.

### Principais métodos
O pipeline descrito nos artefatos combina auditoria, prevenção de leakage, operacionalização do indicador e modelagem:
1) auditoria de dados (duplicatas e missingness),
2) gate de data leakage por proxy (`CRT_SCORE`),
3) construção de targets A/B/C/D por quantis de ESCS e CRT,
4) perfilização person-centered com clusterização (PCA + busca por algoritmos/`k` + estabilidade bootstrap com ARI/Jaccard),
5) modelagem preditiva para Target A (holdout estratificado + CV repetida e ranking por ROC-AUC e métricas derivadas),
6) explicação do melhor modelo via SHAP (`shap_tree`),
7) fairness por variáveis sensíveis reportadas,
8) robustez por bootstrap e calibração.

### Principais resultados (integrados e interpretativos)
1. **Resiliência Criativa (Target A) é rara**: prevalência **0.0427752** (164/3834). A raridade implica que métricas dependentes do limiar (Recall/F1) são mais instáveis e exigem calibração cuidadosa.
2. **Existe estrutura latente consistente com risco diferencial**: a clusterização selecionou **agglomerative_ward k=2** e mostrou que o **cluster 1** (245 estudantes; 6.39% da amostra) apresenta **risk_relative ≈ 1.367** e **odds_ratio ≈ 1.389** para Resiliência Criativa (Target A). Assim, os dados suportam que “resiliência” operacional não se distribui aleatoriamente, mas emerge em subgrupos com padrões específicos.
3. **ML aprende sinal preditivo**: o modelo vencedor (**xgboost_tuned**) alcançou **ROC-AUC ≈ 0.9236 (CV)** e **≈ 0.9313 (holdout)**. O alto ROC-AUC sugere capacidade de discriminação global; o F1 baixo (com limiar 0.50) evidencia o trade-off típico de classes raras.
4. **XAI aponta um eixo socioeducacional dominante (HISCED)** e dois eixos complementares: **ICTRES** e **HOMEPOS**, além de múltiplas variáveis do domínio criativo (`CR*`). Em conjunto com a clusterização, isso sugere um mecanismo educacional coerente com uma visão ecológica: condições socioeducacionais e recursos contextuais podem modular como estudantes expressam criatividade sob vulnerabilidade.
5. **Fairness é condicionada pela distribuição do target em grupos**: o fairness_report apresenta grupos com **prevalência positiva zero**, levando a AUCs como **nan** para esses recortes. Logo, justiça formal completa requer métricas adicionais não disponíveis explicitamente no artefato lido.

### Conclusões principais
- A Resiliência Criativa pode ser operacionalizada empiricamente no PISA 2022 com previsibilidade e explicabilidade.
- Os achados convergem para a ideia de **mecanismos multivariados** envolvendo **socioeconomia/proxy educacional**, **recursos tecnológicos/domésticos** e **padrões de engajamento/criatividade**.
- A validade deve ser interpretada à luz do risco de variáveis explicativas atuarem como proxy de SES e do impacto de classes raras em fairness.

---

## 1. Introdução

### 1.1 Desigualdade educacional e criatividade no PISA 2022
A desigualdade educacional é um fenômeno estrutural que se manifesta em oportunidades, recursos e condições de aprendizagem. O PISA 2022 incorpora dimensões associadas à criatividade, permitindo investigações sobre como estudantes expressam pensamento criativo sob diferentes contextos.

### 1.2 Resiliência Criativa como construto operacional
Neste projeto, Resiliência Criativa é operacionalizada por um procedimento estatístico-quantílico (artefato `target_comparison.md`):
- Target A combina **ESCS baixo** (≤ Q1) com **CRT alto** (≥ Q3).

**O que isso faz conceitualmente?**
Ao definir resiliência como desempenho/expressão de criatividade em condições socioeconômicas vulneráveis, o indicador desloca o foco de “quem é criativo” para “quem mantém criatividade sob vulnerabilidade”.

**Por que isso importa para Learning Analytics/EDM?**
Porque cria um alvo (target) com significado educacional, permitindo explorar associações multivariadas, segmentar subpopulações e explicar modelos.

---

## 2. Objetivo Central

### 2.1 Objetivo Geral
Identificar e explicar mecanismos associados à Resiliência Criativa no PISA 2022 (Brasil), combinando evidências de clusterização (estrutura latente), explicabilidade (SHAP), desempenho preditivo (ML) e avaliação de fairness e robustez.

### 2.2 Objetivos Específicos
- Operacionalizar Resiliência Criativa por targets A/B/C/D e utilizar Target A como referência.
- Auditar dados e mitigar data leakage.
- Detectar perfis latentes via clusterização person-centered.
- Prever Target A com ML e interpretar via SHAP.
- Avaliar fairness e robustez.

### 2.3 Perguntas de Pesquisa (PP)
- **PP1 (prevalência)**: qual a frequência de Resiliência Criativa (Target A)?
- **PP2 (mecanismos)**: quais variáveis (interpretadas via SHAP) estão mais associadas ao indicador e como se organizam em mecanismos multivariados?
- **PP3 (estrutura latente)**: quais perfis latentes diferenciam risco de Resiliência Criativa e como se interpretam pedagogicamente?
- **PP4 (equidade e confiabilidade)**: como fairness e robustez sustentam (ou limitam) a validade dos achados?

---

## 3. Base de Dados

### 3.1 Fonte e amostra
- Dataset: `pisa_brasil_estudo_limpo.csv`.
- Estudantes: **3834** (artefatos `data_audit.md` e `eda_report.md`).

### 3.2 Variáveis e indicadores disponíveis
- Criatividade: `CRT_SCORE` e variáveis `CR*`.
- SES: `ESCS` e discretização `Grupo_ESCS`.
- Sensíveis analisados em fairness: `ST004D01T`, `Grupo_ESCS`, `HISCED`.
- Peso amostral (candidato): `W_FSTUWT`.

### 3.3 Tabela-resumo

| Elemento | Evidência |
|---|---|
| n estudantes | 3834 |
| Criatividade | CRT_SCORE + CR* |
| SES | ESCS + Grupo_ESCS |
| Target usado para explicação e fairness | Target A |
| Prevalência Target A | 0.0427752 |

---

## 4. Metodologia e Pipeline Analítico

### 4.1 Auditoria de Dados
- Duplicatas: **0**.
- Missingness: muito elevada em múltiplas variáveis CR (p.ex., ~0.9267 em `CR567Q06S`).

**O que isso significa?**
Missingness elevada afeta composição das features elegíveis e pode introduzir viés de observação.

**Por que isso importa?**
Em fairness e XAI, a disponibilidade de respostas pode refletir diferenças reais de contexto, mas também limita inferência.

### 4.2 Controle de Data Leakage
- O artefato `leakage_audit.md` identifica candidatos de leakage usando proxy `CRT_SCORE` e critérios (ex.: `corr_abs>=0.70`) e severidade.

**Limitação de transparência**
Quais variáveis foram removidas no conjunto final não está explicitamente listado no texto lido (não disponível no artefato para lista completa).

### 4.3 Construção dos Targets
- Target A: ESCS ≤ Q1 e CRT ≥ Q3 (artefato `target_comparison.md`).
- Também foram definidos Targets B, C e D; prevalências reportadas.

### 4.4 Clusterização person-centered e estabilidade
- Solução escolhida: `agglomerative_ward` com `k=2`.
- PCA: 58 componentes (alvo 0.85).
- Seleção por score multicritério (pesos documentados em config) e estabilidade bootstrap (ARI/Jaccard).

### 4.5 Modelagem preditiva e XAI
- Melhor modelo: `xgboost_tuned`.
- Métricas: ROC-AUC, Precision, Recall, F1, Balanced Accuracy (conforme `modeling_report.md`).
- XAI: SHAP árvore (`shap_tree`) com ranking por `mean_abs_shap` (conforme `shap_report.md`).

### 4.6 Fairness e robustez
- Fairness report: desempenho por grupos sensíveis e ocorrência de grupos com AUC nan devido a prevalence 0.
- Robustez: bootstrap (n=200) com ROC-AUC e Brier score reportados.

---

## 5. Resultados Integrados (Narrativa Interpretativa)

### 5.1 Prevalência: por que Resiliência Criativa é difícil de medir e modelar
**Evidência**: Target A tem prevalência **0.0427752**.

**O que este resultado significa?**
A raridade implica que, mesmo com bom ROC-AUC, a identificação prática de “resilientes” pode ser sensível a limiar e definição operacional. Assim, o pipeline não avalia apenas capacidade de discriminação, mas também a confiabilidade dessa capacidade quando a classe positiva é pouco frequente.

**Por que este resultado importa?**
Em intervenção educacional, classificar estudantes raros exige alta estabilidade e equidade; do contrário, decisões podem ser instáveis ou injustas.

### 5.2 Mecanismos multivariados: convergência entre ML, SHAP e clusterização
Ao integrar os artefatos de clusterização (`clusterer_report.md`), XAI (`shap_report.md`) e modelagem (`modeling_report.md`), emerge um mecanismo plausível nos dados:

1. **Eixo socioeducacional dominante**: SHAP aponta `HISCED` como a maior importância.
2. **Eixo de recursos e mediação**: `ICTRES` e `HOMEPOS` aparecem entre os maiores SHAP values.
3. **Eixo criativo/processual**: muitas variáveis do domínio `CR*` aparecem tanto nas top diferenças de resilientes (`resilient_profile.md`) quanto no `top_above`/`top_below` dos clusters.

**O que isso significa?**
Nos dados, a Resiliência Criativa operacional não é explicada por um único fator, mas por uma combinação:
- condições socioeducacionais e de status educacional (proxy via HISCED),
- recursos e acesso (tecnológico e doméstico),
- e padrões de engajamento/expressão criativa capturados por variáveis `CR*`.

**Por que isso importa?**
Porque contraria a visão reducionista de “resiliência” como atributo individual e favorece uma interpretação mais ecológica: estudantes podem expressar criatividade sob vulnerabilidade quando mediadores contextuais (recursos e apoio) estão presentes.

### 5.3 Perfis latentes: caracterização detalhada dos clusters encontrados
A clusterização selecionou dois perfis (k=2). O relatório textual fornece `cluster_name`, tamanho, prevalência, e listas de variáveis dominantes (`top_above`, `top_below`).

#### Perfil Latente 1 — Cluster 0 (Alta — Engajamento Criativo)
- **Tamanho**: 3589 estudantes.
- **Prevalência**: 0.936098.
- **top_above** (10 principais listadas no artefato):
  - `CR590Q14S; CR590Q04S; CR590Q22S; ICTRES; CR590Q23S; CR590Q05S; CR590Q03S; CR590Q40TT; CR590Q36TT; CR590Q26TT`
- **top_below** (10 principais listadas):
  - `CR590Q03TT; CR590Q33TT; CR590Q43TT; CR590Q14TT; CR590Q38TT; CR590Q20TT; CR590Q27TT; CR590Q23TT; CR590Q47TT; CR590Q22TT`

**Interpretação pedagógica (com limites de evidência semântica)**
- O cluster 0 é o perfil dominante e se caracteriza por um conjunto de variáveis com códigos `CR590Q*` associados (no contexto do projeto) a tarefas/criatividade, além de `ICTRES`.
- Como o artefato não traz o dicionário semântico completo de cada `CR590Qxx*`, a interpretação pedagógica precisa ser descrita como “padrões específicos de respostas em tarefas de engajamento criativo”, em vez de afirmar conteúdos cognitivos específicos.

#### Perfil Latente 2 — Cluster 1 (Alta — Engajamento Criativo)
- **Tamanho**: 245 estudantes.
- **Prevalência**: 0.063902.
- **top_above**:
  - `CR590Q03TT; CR590Q33TT; CR590Q43TT; CR590Q14TT; CR590Q38TT; CR590Q20TT; CR590Q27TT; CR590Q23TT; CR590Q47TT; CR590Q22TT`
- **top_below**:
  - `CR590Q14S; CR590Q04S; CR590Q22S; ICTRES; CR590Q23S; CR590Q05S; CR590Q03S; CR590Q40TT; CR590Q36TT; CR590Q26TT`

**Interpretação pedagógica**
- O cluster 1 é minoritário, mas concentra maior risco para Resiliência Criativa (Target A).
- Pedagogicamente, o resultado sugere que existem subpadrões em “engajamento criativo” (por códigos `TT` vs `S` e combinações específicas de `CR590`), os quais, sob a definição do target, se traduzem em maior probabilidade operacional de resiliência.

### 5.4 Como fairness afeta (e limita) a validade das conclusões
**Evidência**: fairness_report analisa sensíveis (`ST004D01T`, `Grupo_ESCS`, `HISCED`) e mostra que alguns grupos têm prevalence positiva zero, resultando em ROC-AUC como **nan**.

**O que este resultado significa?**
Quando a classe positiva não ocorre em um grupo, métricas de desempenho baseadas em separação (como AUC) tornam-se inviáveis.

**Por que isso importa?**
Porque impõe limites ao diagnóstico de equidade. Mesmo que o modelo seja globalmente bom, a avaliação de justiça fica incompleta para grupos raros.

### 5.5 Como ML e XAI estruturam a explicação única dos mecanismos
**Evidência ML**: xgboost_tuned com ROC-AUC alto.

**Evidência XAI**: SHAP indica HISCED dominante; ICTRES e HOMEPOS relevantes; múltiplas variáveis CR relevantes.

**Evidência Cluster**: clusters com composição diferente de variáveis CR590 e ICTRES apresentam risco diferencial para Target A.

**Síntese em uma explicação única**
Os dados sustentam um mecanismo integrado:
1) o status educacional/socioeconômico (proxy HISCED) cria uma base contextual,
2) recursos e mediações (ICTRES e HOMEPOS) podem atuar como facilitadores para transformar engajamento em criatividade observável,
3) padrões específicos de respostas em tarefas de criatividade/engajamento (variáveis CR* e CR590) diferenciam subgrupos,
4) juntos, esses elementos aumentam a probabilidade operacional de Target A.

---

## 5.6 Risco crítico: HISCED como proxy de ESCS (validade e impactos)

### 5.6.1 Por que este risco é real no projeto
- ESCS é a variável componente do target A (ESCS ≤ Q1).
- `HISCED` aparece como sensível em fairness e como maior importância em SHAP.

Assim, é plausível que `HISCED` carregue informações muito próximas (ou correlacionadas) de ESCS ou de componentes educacionais que se relacionam diretamente com vulnerabilidade socioeconômica.

### 5.6.2 O que este resultado significa?
Se `HISCED` atua como proxy de `ESCS`, então parte da “explicação” do modelo pode estar capturando o mesmo gradiente socioeconômico que define o target, e não um mecanismo independente ligado à resiliência criativa em si.

### 5.6.3 Por que isso importa para validade?
- A validade interna do indicador fica condicionada: o modelo pode aprender “diferenças socioeconômicas” refletidas indiretamente no target, mesmo após leakage gate.
- Para validade externa e para interpretação educacional, isso exige cautela: políticas deduzidas a partir de HISCED podem estar atuando sobre a vulnerabilidade socioeconômica geral (o que pode ser legítimo), mas não necessariamente sobre “resiliência criativa” como mecanismo específico.

### 5.6.4 Evidência disponível e limitações
O artefato lido **não contém** uma análise formal de correlação entre HISCED e ESCS, nem uma verificação explícita de proxy (como “mutual information ratio” para esse par) no texto lido. Logo, a afirmação sobre proxy é tratada como **risco interpretativo fundamentado** pela estrutura do target e pela importância em SHAP.

**Onde isso está documentado?**
- O papel de ESCS no target está em `target_comparison.md`.
- A importância de HISCED está em `shap_report.md`.

---

## 6. Discussão (nível artigo científico)

### 6.1 Resiliência acadêmica e resiliência criativa: continuidade teórica e distinção empírica
A evidência operacional do projeto sugere que Resiliência Criativa pode ser entendida como uma extensão do raciocínio de resiliência acadêmica: estudantes em contextos desfavoráveis podem manter expressões de competências (aqui, criatividade). O que diferencia este estudo é que o construto operacionaliza criatividade e vulnerabilidade simultaneamente.

**O que este resultado significa?**
A prevalência baixa indica que a “resiliência criativa” é um padrão de exceção estatística, compatível com a ideia de que manter criatividade sob vulnerabilidade requer condições particulares.

**Por que importa?**
Porque sugere que intervenções precisam ser sensíveis ao risco: identificar e apoiar subgrupos que compõem o cluster de maior risco relativo.

### 6.2 Pensamento criativo: mecanismos de engajamento e expressão
Os clusters compartilham o rótulo “Alta — Engajamento Criativo”, mas diferem nas variáveis dominantes (CR590Qxx com codificações distintas e ICTRES/ausências). Isso implica que “alto engajamento” não é homogêneo: existem submodos de engajamento que, empiricamente, se relacionam com o target.

### 6.3 Learning Analytics e EDM: pipeline como evidência de confiabilidade
O projeto incorpora boas práticas:
- auditoria e missingness,
- gate de leakage,
- estabilidade via bootstrap para clusterização,
- robustez e calibração.

**O que isso significa?**
A confiabilidade do sinal melhora quando a estrutura é testada em reamostragens e quando há mitigação de vazamento.

### 6.4 Explainable AI: SHAP como ferramenta interpretativa e como risco de “proxy learning”
O SHAP identifica HISCED como mais importante. Sob a perspectiva de validade, isso pode ser interpretado de duas formas:
1) HISCED representa um mediador real de condições que habilitam criatividade sob vulnerabilidade;
2) HISCED atua como proxy de ESCS e, por consequência, a explicação pode refletir principalmente o gradiente socioeconômico que define o target.

O projeto exige, portanto, uma leitura em dois níveis: interpretativo (mecanismos contextuais) e metodológico (risco de proxy learning).

### 6.5 Convergências e divergências
- Convergência: clusterização (risco diferencial) + SHAP (variáveis dominantes) + ML (alto ROC-AUC) sustentam uma narrativa integrada.
- Divergência/limitação: fairness tem métricas possivelmente incompletas devido a prevalence zero em subgrupos (não disponível no artefato lido para métricas formais adicionais).

---

## 7. Conclusões

### O que é a resiliência criativa?
É operacionalmente definida como estudantes com **criatividade alta (CRT_SCORE)** apesar de **vulnerabilidade socioeconômica (ESCS baixo)**, no Target A (ESCS ≤ Q1 e CRT ≥ Q3).

### Quão frequente ela é?
No PISA Brasil processado do projeto, prevalência **0.0427752**.

### Quais fatores estão associados?
SHAP aponta **HISCED** como dominante, seguido por **ICTRES** e **HOMEPOS**, além de múltiplas variáveis do domínio criativo (CR*). A clusterização reforça subpadrões em `CR590Q*` e participação/ausência de `ICTRES` nos perfis.

### O que os modelos revelaram?
O melhor modelo (xgboost_tuned) apresenta alto desempenho global (ROC-AUC ~0.9236 CV e ~0.9313 holdout). A heterogeneidade por clusters indica que a resiliência criativa se concentra em subgrupos.

### O que as políticas públicas podem fazer?
Os achados são compatíveis com políticas em múltiplos níveis:
- reduzir lacunas socioeducacionais (capturadas por proxies como HISCED/ESCS);
- aumentar acesso e uso efetivo de tecnologia educacional (ICTRES);
- fortalecer recursos domésticos e ambientes de apoio (HOMEPOS);
- desenhar intervenções pedagógicas que atendam subpadrões de engajamento criativo observáveis em variáveis `CR*`.

### Nota final sobre evidência e validade
A interpretação dos mecanismos deve considerar o risco de proxy: HISCED pode atuar como proxy de ESCS, potencialmente alinhando-se ao componente que define o target. Isso não invalida o estudo, mas exige cautela na causalidade interpretativa e na extrapolação para desenho de políticas.

---

# Arquivo gerado
Este relatório foi salvo em `reports/final_analytical_report_v3.md`.

(Se desejado, posso substituir também `reports/final_analytical_report.md` pela versão v3 após confirmação.)

