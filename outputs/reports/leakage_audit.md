# Leakage Audit (gate inicial)

- Proxy de target usada: **CRT_SCORE**
- Dataset: pisa_brasil_estudo_limpo.csv

## Top leakage candidates (top 50)
```csv
nome,score_corr_abs,corr_sign,spearman_abs,n_unique,missing_ratio,reason,severity
CR561Q06S,0.8637585182155149,0.8637585182155149,0.8331622263031047,5,0.9214919144496609,corr_abs>=0.70,MÉDIO
Creative_Resilience,0.29097471544437004,0.29097471544437004,0.28237978346832493,2,0.0,name contains creative/resilience,CRÍTICO
Grupo_ESCS,0.2007772554820604,0.2007772554820604,0.22052246820191862,4,0.0,name contains known target-proxy families,CRÍTICO
```