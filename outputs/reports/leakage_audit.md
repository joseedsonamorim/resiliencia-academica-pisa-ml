# Leakage Audit (gate inicial)

- Proxy de target usada: **CRT_SCORE**
- Dataset: pisa_brasil_estudo_limpo.csv

## Top leakage candidates (top 50)
| nome                |   score_corr_abs |   corr_sign |   spearman_abs |   n_unique |   missing_ratio | reason                                    | severity   |
|:--------------------|-----------------:|------------:|---------------:|-----------:|----------------:|:------------------------------------------|:-----------|
| CR561Q06S           |         0.863759 |    0.863759 |       0.833162 |          5 |        0.921492 | corr_abs>=0.70                            | MÉDIO      |
| Creative_Resilience |         0.290975 |    0.290975 |       0.28238  |          2 |        0        | name contains creative/resilience         | CRÍTICO    |
| Grupo_ESCS          |         0.200777 |    0.200777 |       0.220522 |          4 |        0        | name contains known target-proxy families | CRÍTICO    |