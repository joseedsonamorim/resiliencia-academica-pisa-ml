# Relatório resumido

## Objetivo
Identificar estudantes resilientes no PISA Brasil e comparar métodos de aprendizado de máquina para selecionar o modelo com melhor evidência preditiva sob validação rigorosa.

## Dados e target
Foram analisados 3,834 estudantes do arquivo `pisa_brasil_estudo_limpo.csv`. O target ativo foi **A**, com 164 casos positivos (4.28% da amostra). As features passaram por filtros de missingness, baixa variância, pesos/IDs e variáveis com risco de vazamento.

## Método
Foram comparados 14 candidatos/variantes de modelos, incluindo regressão logística, SVM, KNN, Naive Bayes, Random Forest, Extra Trees, boosting e modelos opcionais instalados (XGBoost/LightGBM/CatBoost quando disponíveis). A seleção usou ROC-AUC médio em CV repetida (5 folds x 3 repetições) no treino; o holdout estratificado foi reservado para avaliação final. O melhor modelo teve incerteza estimada por bootstrap.

## Resultado principal
O melhor modelo foi **xgboost_tuned**, com ROC-AUC médio de CV=0.9236 e ROC-AUC no holdout=0.9313. Com limiar padrão 0.50, o F1 no holdout foi 0.2051; com limiar otimizado em validação interna (0.110), o F1 subiu para 0.3548. O bootstrap do holdout estimou ROC-AUC médio 0.9316 (IC95% 0.9019-0.9575).

## Leitura científica
O desempenho discriminativo é alto, mas a classe resiliente é rara; por isso, precisão, recall, average precision, calibração, fairness e estabilidade entre targets devem acompanhar o ROC-AUC. Para submissão em periódico de alto impacto, recomenda-se explicitar a definição teórica de resiliência, o desenho amostral do PISA, os pesos, o controle de vazamento e análises de sensibilidade entre targets.
