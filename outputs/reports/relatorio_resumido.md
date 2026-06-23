# Relatório resumido

## Objetivo
Identificar estudantes resilientes no PISA Brasil e comparar métodos de aprendizado de máquina para selecionar o modelo com melhor evidência preditiva sob validação rigorosa.

## Dados e target
Foram analisados 3,834 estudantes do arquivo `pisa_brasil_estudo_limpo.csv`. O target ativo foi **A**, com 168 casos positivos (4.38% da amostra). As features passaram por filtros de missingness, baixa variância, pesos/IDs e variáveis com risco de vazamento.

## Método
Foram comparados 13 candidatos/variantes de modelos, incluindo regressão logística, SVM, KNN, Naive Bayes, Random Forest, Extra Trees, boosting e modelos opcionais instalados (XGBoost/LightGBM/CatBoost quando disponíveis). A seleção usou **Average Precision (PR-AUC)** médio em CV repetida (5 folds x 3 repetições) no treino (SR-2); o holdout estratificado foi reservado para avaliação final. O target foi recomputado com limiares exclusivos do treino (CC-1). O modelo final foi calibrado (SR-1).

## Resultado principal
O melhor modelo foi **gaussian_nb**, com Average Precision (PR-AUC) médio de CV=0.2013 e ROC-AUC no holdout=0.8801. Com limiar padrão 0.50, o F1 no holdout foi 0.2500; com limiar otimizado em validação interna (0.195), o F1 subiu para 0.2675. O bootstrap do holdout estimou ROC-AUC médio 0.8796 (IC95% 0.8478-0.9079).

## Leitura científica
O desempenho discriminativo é alto, mas a classe resiliente é rara; por isso, precisão, recall, average precision, calibração, fairness e estabilidade entre targets devem acompanhar o ROC-AUC. Para submissão em periódico de alto impacto, recomenda-se explicitar a definição teórica de resiliência, o desenho amostral do PISA, os pesos, o controle de vazamento e análises de sensibilidade entre targets.
