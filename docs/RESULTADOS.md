# Resultados já gerados

Última execução bem-sucedida do pipeline (fases 1–12). Arquivos em `outputs/` e `models/`.

## Principais achados

### Targets (prevalência)
| Target | Prevalência | Regra resumida |
|--------|-------------|----------------|
| A | ~4,3% | ESCS baixo (Q1) + CRT alto (Q3) |
| B | ~6,5% | ESCS ≤ P30 + CRT ≥ P70 |
| C | ~1,8% | Mais restritivo (CRT ≥ P90) |
| D | ~30% | Score composto z(CRT)−z(ESCS) ≥ P70 |

### Clusterização (k=4)
Melhor **silhouette** → aglomerativo hierárquico (~0,29).

### Modelagem (target A)
| Modelo | ROC-AUC (holdout) |
|--------|-------------------|
| Random Forest | ~0,93 |
| Regressão logística | ~0,89 |

Modelo salvo: `models/best_model.joblib`

### Dataset
- **3.834** estudantes, **1.275** variáveis
- Arquivo: `data/pisa_brasil_estudo_limpo.csv`

## Onde encontrar cada artefato

| Fase | Relatório | Tabelas / figuras |
|------|-----------|-------------------|
| Auditoria | `outputs/reports/data_audit.md` | `outputs/figures/missing_values.png` |
| Targets | `outputs/reports/target_comparison.md` | `outputs/tables/targets_definitions*.csv` |
| Leakage | `outputs/reports/leakage_audit.md` | `outputs/tables/leakage_candidates.csv` |
| EDA | `outputs/reports/eda_report.md` | `outputs/tables/eda_numeric_describe.csv` |
| Perfil | `outputs/reports/resilient_profile.md` | `outputs/figures/resilient_profile/` |
| Clusters | `outputs/reports/clusterer_report.md` | `outputs/tables/clusterer_*.csv` |
| Modelos | `outputs/reports/modeling_report.md` | `outputs/tables/modeling_metrics.csv` |
| SHAP | `outputs/reports/shap_report.md` | `outputs/figures/shap/shap_top20.png` |
| Fairness | `outputs/reports/fairness_report.md` | `outputs/tables/fairness_by_group.csv` |
| Robustez | `outputs/reports/robustness_report.md` | `outputs/tables/robustness_*.csv` |

Para rever tudo visualmente: `./scripts/run_dashboard.sh`
