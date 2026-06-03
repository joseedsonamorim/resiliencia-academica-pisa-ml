# Estrutura do projeto

## Pastas

| Pasta | Conteúdo |
|-------|----------|
| `config/` | `config.yaml` — seeds, paths, targets, fairness |
| `project/` | `metadata.json` — rastreio do dataset e versão do pipeline |
| `data/` | Dataset CSV (não versionado no git) |
| `src/` | Módulos do pipeline e `src/utils/` |
| `scripts/` | Shell scripts para rodar fases |
| `outputs/reports/` | Relatórios Markdown |
| `outputs/tables/` | CSVs (métricas, targets, leakage, etc.) |
| `outputs/figures/` | Figuras por fase |
| `models/` | `best_model.joblib`, `modeling_meta.json` |
| `dashboard/` | `app.py` (Streamlit) |
| `docs/` | Documentação e `TODO.md` |
| `tests/` | Testes (a implementar) |

## Código principal

- `src/main.py` — CLI (`python3 -m src.main --stage <nome>`)
- `src/pipeline_stages.py` — despacho das fases
- `src/utils/paths.py` — caminhos absolutos a partir da raiz do repo

## Arquivos removidos na reorganização

- CSV/RDS duplicados na raiz do projeto
- Stubs vazios (`*_pipeline_stub.py`)
- Scripts `run_*.sh` espalhados na raiz (unificados em `scripts/`)
- `venv_add_deps.sh` (dependências em `requirements.txt`)
