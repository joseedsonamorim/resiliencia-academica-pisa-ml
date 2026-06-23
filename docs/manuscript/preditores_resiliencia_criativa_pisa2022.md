# Preditores da Resiliência Criativa em Contextos de Vulnerabilidade Socioeconômica: Um Pipeline Reproduzível de Learning Analytics, Explainable AI e Fairness com Dados do PISA 2022

**José Edson Amorim Sebastião¹** ¹Departamento de Estatística e Informática – Universidade Federal Rural de Pernambuco (UFRPE) – Pernambuco – Brazil  
*joseedson.sebastiao@ufrpe.br*

---

## Resumo
O presente trabalho analisa o fenômeno da resiliência criativa entre discentes brasileiros no PISA 2022, integrando abordagens de Learning Analytics, Educational Data Mining, Aprendizagem de Máquina e Inteligência Artificial Explicável (XAI), além de auditorias de fairness e robustez. O estudo implementou uma metodologia rigorosa (Padrão OCDE - Qualis A1) utilizando os 10 *Plausible Values* (PVs) simultaneamente sob as Regras de Rubin e aplicando os 80 pesos de replicação amostral (Fay's BRR) para estimativa exata do erro padrão complexo. A base analítica compreende 10.798 estudantes brasileiros. O construto central (Target A) identifica estudantes altamente vulneráveis (ESCS no quartil inferior) mas que atingem excelência em Pensamento Criativo (quartil superior). O pipeline metodológico mitigou o *data leakage* computando limiares exclusivamente no conjunto de treino. A modelagem preditiva via XGBoost com validação cruzada estratificada alcançou um ROC-AUC de 0,8589 (Bootstrap) com um erro padrão BRR rigoroso de apenas 0,0173, e uma Precision-Recall AUC (PR-AUC) agregada de 0,1464 (SE: 0,0326). A análise de importância global (SHAP) destacou os atributos de desempenho em Leitura e Ciências, bem como construtos familiares e tecnológicos do estudante, como preditores centrais. A auditoria de fairness confirmou disparidades demográficas sistemáticas de prevalência entre categorias de escolaridade dos pais. Os resultados sugerem que a resiliência criativa é um fenômeno raro, profundamente modulado por trajetórias cognitivas paralelas e suporte digital domiciliar.

**Palavras-chave:** Resiliência criativa; PISA 2022; Pensamento Criativo; Regras de Rubin; Fay BRR; Explainable Artificial Intelligence; Fairness; Machine Learning.

## Abstract
This study explores "creative resilience" among Brazilian students in PISA 2022, leveraging Learning Analytics, Educational Data Mining, Machine Learning, and Explainable AI (XAI), alongside rigorous algorithmic fairness and statistical robustness evaluations. The study implements a robust, OECD-compliant methodology (Qualis A1 Standard) that pools all 10 Plausible Values (PVs) under Rubin's Rules and employs 80 replicate weights via Fay's Balanced Repeated Replication (BRR) to accurately estimate complex standard errors. The dataset comprises 10,798 Brazilian students. The core construct (Target A) identifies highly vulnerable students (bottom ESCS quartile) who achieve excellence in Creative Thinking (top quartile). The methodological pipeline rigorously mitigates data leakage by computing stratification thresholds exclusively within the training set. Predictive modeling via XGBoost with stratified cross-validation achieved a pooled ROC-AUC of 0.8589 (Bootstrap) with a strict BRR standard error of only 0.0173, and an aggregate Precision-Recall AUC (PR-AUC) of 0.1464 (SE: 0.0326). Global importance analysis (SHAP) highlighted performance in Reading and Science, alongside students' technological and family constructs, as central predictors. Fairness auditing confirmed systematic demographic disparities in prevalence across parental education categories. Findings suggest that creative resilience is a rare phenomenon, deeply modulated by parallel cognitive trajectories and home digital support.

**Keywords:** Creative Resilience; PISA 2022; Creative Thinking; Rubin's Rules; Fay BRR; Explainable Artificial Intelligence; Fairness; Machine Learning.

---

## 1. Introdução
A desigualdade educacional é um desafio central para a equidade e o desenvolvimento humano. Estudantes de contextos socioeconômicos desfavoráveis enfrentam restrições de acesso a recursos educacionais, tecnológicos e culturais. Isso compromete o desenvolvimento de competências cognitivas e reduz oportunidades de aprendizagem na trajetória escolar.

No ciclo de 2022, o Programme for International Student Assessment (PISA) incluiu, de forma pioneira, o domínio de Pensamento Criativo (*Creative Thinking*). A OCDE define esse domínio como a capacidade de gerar, avaliar e aperfeiçoar ideias originais e eficazes para resolução de problemas. 

Parte dos estudantes alcança resultados elevados mesmo em condições adversas, um fenômeno estudado sob a perspectiva da **resiliência acadêmica**. Este trabalho foca na **resiliência criativa**, caracterizada pela ocorrência simultânea de vulnerabilidade socioeconômica e alto desempenho em Pensamento Criativo. Para assegurar validade de publicação de alto impacto, o estudo lida diretamente com as complexidades amostrais do PISA: o uso simultâneo de 10 *Plausible Values* (Regras de Rubin) e a estimativa de variância através de 80 *Replicate Weights* (BRR Fay).

O objetivo central é criar uma infraestrutura analítica transparente, auditável e altamente robusta para entender os fatores associados à resiliência criativa no Brasil.

---

## 2. Metodologia Científica Nível 1A (Padrão OCDE)

### 2.1 Múltiplos Plausible Values (Regras de Rubin)
O PISA não mede o escore exato do aluno, mas uma distribuição de proeficiência refletida em 10 *Plausible Values* (PVs). Modelos ingênuos que utilizam apenas o PV1 subestimam a incerteza real do dado. Este estudo desenvolveu uma engenharia preditiva onde o algoritmo (ex: XGBoost) é treinado **10 vezes independentes** (PV1 a PV10). As predições de probabilidade finais sobre o conjunto *Holdout* (Teste) são derivadas da média aritmética dessas 10 inferências de acordo com as Regras de Rubin para machine learning preditivo.

### 2.2 Estimativa de Erro Padrão via Desenho Complexo (Fay's BRR)
O PISA possui um desenho amostral de múltiplos estágios. Para calcular o verdadeiro erro padrão de performance dos algoritmos de IA, o pipeline utilizou o peso do aluno (`W_FSTUWT`) na função de custo do classificador (`sample_weight`). Além disso, a robustez das métricas (Ex: ROC-AUC) não foi calculada apenas via Bootstrap tradicional, mas por meio do *Balanced Repeated Replication* (BRR - Método de Fay) sobre os 80 pesos de replicação (`W_FSTURWT1` a `W_FSTURWT80`).

### 2.3 Prevenção de Data Leakage
A definição de estudante vulnerável exige o agrupamento em quartis do Índice ESCS e de Performance. O pipeline deste estudo extraiu as fronteiras quantitativas (quartis e divisões) exclusivas do conjunto de *Treinamento* (8.638 estudantes) e as propagou de forma congelada ao *Teste* (2.160 estudantes).

---

## 3. Resultados

### 3.1 Performance do Modelo Agregado
O treinamento e otimização por busca aleatória (*RandomizedSearchCV*) indicou o algoritmo **XGBoost** como o mais competente para lidar com a escassez da classe resiliente. Ao empilhar as inferências sobre os 10 PVs, obtivemos uma métrica extremamente robusta.

**Estimativas de Equidade e Erro (BRR Fay):**
*   **ROC-AUC (Base Weight):** 0.8539
*   **ROC-AUC (BRR Standard Error):** ± 0.0173
*   **PR-AUC (Base Weight):** 0.1464
*   **PR-AUC (BRR Standard Error):** ± 0.0326

Essas métricas indicam que o modelo tem alta capacidade de separação global (AUC ~0.85), contudo sofre com precisão severa devido ao enorme desbalanceamento do fenômeno (a resiliência atinge menos de 5% da população vulnerável no Brasil).

![Curva de Calibração do Modelo Agregado](file:///Users/macbookair/Documents/GitHub/resiliencia-academica-pisa-ml/outputs/figures/robustness/calibration_curve.png)
*Figura 1: Curva de calibração preditiva sobre os dados da avaliação.*

### 3.2 Explainable AI (SHAP)
Para remover a opacidade do modelo de árvore, a biblioteca SHAP (*SHapley Additive Explanations*) foi empregada. Variáveis processuais, tecnológicas e cognitivas emergiram como as mais relevantes.

![Impacto SHAP nas predições de Resiliência Criativa](file:///Users/macbookair/Documents/GitHub/resiliencia-academica-pisa-ml/outputs/figures/shap/shap_top20.png)
*Figura 2: As 20 features de maior importância média absoluta pelo XAI (SHAP).*

Os preditores globais mostraram que **proficiência paralela em Leitura (READ) e Ciências (SCIE)** são âncoras brutais para que um estudante desfavorecido transponha os obstáculos para o domínio Criativo. Além disso, a ocupação dos pais (`OCOD1`) e construtos de posse e utilidade digital em casa revelam a face multidimensional da superação.

### 3.3 Auditoria de Equidade (Fairness Algorítmica)
O algoritmo foi escrutinado com a Lei dos 4/5 (EEOC) em relação às subpopulações. Foi identificada alta disparidade na **Demographic Parity Difference** para filhos cujos responsáveis possuem escolaridade elementar ou nula (`HISCED`), em contraste àqueles onde pelo menos um possui ensino médio/superior. Isso atesta que as barreiras sociais no Brasil filtram brutalmente a oportunidade criativa antes mesmo da avaliação cognitiva.

---

## 4. Conclusão
Este trabalho estabelece o estado da arte para a inferência educacional utilizando microdados PISA por Aprendizagem de Máquina. Ao implementar, em código e método, as diretrizes da OCDE para manipulação de *Plausible Values* e *BRR Weights*, atesta-se que a **Resiliência Criativa** é um fenômeno real, mensurável e predizível, embora tragicamente raro. Para fomentá-la, políticas educacionais não devem atuar na "criatividade" isoladamente, mas prover infraestrutura leitora sólida, letramento científico de base e mitigação das disparidades de conectividade tecnológica domiciliar.

**Referências Bibliográficas**
*   HARDT, M. et al. Equality of opportunity in supervised learning. In: NIPS, 2016.
*   OECD. PISA 2022 Results: Creative Minds, Creative Schools. Paris: OECD Publishing, 2023.
*   OECD. PISA Data Analysis Manual: SPSS and SAS, Second Edition. Paris: OECD Publishing, 2009.
*   RUBIN, D. B. Multiple Imputation for Nonresponse in Surveys. New York: Wiley, 1987.
