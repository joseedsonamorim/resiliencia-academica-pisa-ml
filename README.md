# PISA 2022 — Resiliência Criativa (Brasil)

Pipeline de análise do microdado PISA 2022: auditoria, targets A/B/C/D, modelagem e dashboard.

## Estrutura

```
├── config/config.yaml      # parâmetros do pipeline
├── project/metadata.json   # metadados (dataset detectado, versão)
├── data/                   # CSV de entrada (pisa_brasil_estudo_limpo.csv)
├── src/                    # código Python por fase
├── scripts/                # run_stage.sh, run_all.sh, setup_venv.sh
├── outputs/
│   ├── reports/            # relatórios .md
│   ├── tables/             # tabelas .csv
│   └── figures/            # gráficos .png
├── models/                 # modelos treinados (.joblib)
└── dashboard/              # app Streamlit
```

Documentação detalhada: [docs/ESTRUTURA.md](docs/ESTRUTURA.md)

## Início rápido

```bash
./venv_setup.sh
source .venv/bin/activate
./scripts/run_stage.sh data_audit
./scripts/run_all.sh          # todas as fases (demorado)
./scripts/run_dashboard.sh    # interface Streamlit
```

Use sempre `python3` (não `python`).

## Fases do pipeline

| Stage | Descrição |
|-------|-----------|
| `data_audit` | Qualidade e missingness |
| `data_dictionary` | Dicionário de variáveis |
| `variable_discovery` | Catálogo por família |
| `leakage_audit` | Gate de vazamento |
| `target_comparison` | Targets A/B/C/D |
| `eda` | Análise exploratória |
| `resilient_profile` | Perfil resilientes vs não |
| `clusterer` | KMeans / hierárquico / GMM |
| `modeling` | LR, Random Forest (+ boosting opcional) |
| `shap` | Importância de features |
| `fairness` | Métricas por grupo sensível |
| `robustness` | Bootstrap e calibração |
| `dashboard` | Manifest + Streamlit |

## Resultados

Após executar o pipeline, consulte `outputs/reports/` e `outputs/tables/`. O modelo treinado fica em `models/best_model.joblib`.
