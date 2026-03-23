# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.
# Resiliencia Academica PISA ML - Funcionamento Completo

## Funcionamento do Projeto
1. **Carregamento e Limpeza (Fase 2)**: `src/data_loader.py` carrega PISA 2022 CSV/Parquet, filtra LATAM, remove alunos sem PV1MATH, imputa ESCS/HISEI/HOMEPOS por pais.
2. **Variavel Alvo**: `src/utils.py` - `resilient_quartiles` = ESCS <= Q1 (baixa renda) E PV1MATH >= Q4 (altas notas).
3. **EDA e Visualizacoes**: `src/visualizations.py` - scatter SES vs prof, bar/scatter renda (ESCS) vs dispositivos (HOMEPOS).
4. **Clustering (Fase 3)**: K-Means n=4 em resilientes, perfis archetypes (recursos, escola, motivado, desvantagens).
5. **Classificacao RF (Fase 4)**: `src/ml_models.py` - Random Forest com SMOTE balance + class_weight, SHAP importance (repeticao ST29Q01, internet IC004Q01/HOMEPOS).
6. **Dashboard Streamlit**: `app.py` - Tabs EDA/ML/Clusters, botoes train RF/KMeans, SHAP plots, sidebar docs.
7. **Documentacao**: `/docs/` - teorico.md, eda.md, kdd.md (processo KDD completo).

## Como Executar
```
pip install -r requirements.txt
streamlit run app.py
```

## Dados
- Demo LATAM para teste.
- Real: Baixe PISA 2022 student CSV para `data/pisa_2022_full.csv`.


---
Copyright (c) 2026 Jose Edson Amorim Sebastiao.

