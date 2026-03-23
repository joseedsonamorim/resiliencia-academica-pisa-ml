# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

# Processo KDD Fase 4

1. Understanding Domain: Resiliencia PISA PP1.
2. Target Dataset: pisa_latam.parquet.
3. Cleaning: Drop math NaN, impute.
4. Reduction: Features ESCS HISEI HOMEPOS ST29 IC004.
5. Mining Task: Classification RF.
6. Algorithm: RF SMOTE balanced.
7. Mining: Pipeline fit.
8. Evaluation: F1 resil 0.7 demo.
9. Interpretation: SHAP HISEI top, repeticao neg, internet pos.

Rotina: Streamlit cached, joblib.

