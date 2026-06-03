# Modeling (Fase 8)

- Dataset: `pisa_brasil_estudo_limpo.csv`
- Target: **A**
- Features: 51
- Melhor modelo (ROC-AUC holdout): **random_forest** (0.9269)

| model               | target   |   accuracy |   precision |   recall |       f1 |   roc_auc |   cv_roc_auc_mean |   cv_f1_mean |
|:--------------------|:---------|-----------:|------------:|---------:|---------:|----------:|------------------:|-------------:|
| random_forest       | A        |   0.946545 |    0.357143 | 0.30303  | 0.327869 |  0.926947 |          0.904951 |     0.271405 |
| logistic_regression | A        |   0.840939 |    0.175182 | 0.727273 | 0.282353 |  0.894228 |          0.901618 |     0.301943 |
