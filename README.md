# PISA 2022 — Resiliência Criativa (Brasil)

Pipeline acadêmico de Machine Learning para detecção e análise de **Resiliência Criativa** em estudantes brasileiros utilizando os microdados oficiais do PISA 2022.

Este projeto implementa metodologias analíticas robustas padrão OCDE (Qualis A1), operando simultaneamente sobre os 10 *Plausible Values* (Regras de Rubin) e estimando o erro padrão através do método *Balanced Repeated Replication* (BRR Fay) utilizando 80 pesos replicados.

## Estrutura Científica de Diretórios

```
├── config/                 # Arquivos de configuração yaml
├── dashboard/              # Código fonte do app interativo Streamlit
├── data/                   # [Ignorado no Git] Dados locais do PISA
│   ├── raw/                # Arquivos originais (.sav, .csv cru)
│   └── processed/          # Datasets filtrados após limpeza
├── docs/                   # Documentação e referências
│   ├── manuscript/         # Versões do artigo científico (.md, .docx)
│   └── references/         # Documentos oficiais, questionários, frameworks
├── models/                 # [Joblibs ignorados] Modelos empacotados
├── outputs/                # Resultados das fases do pipeline
│   ├── figures/            # Gráficos e visualizações (SHAP, Curvas)
│   ├── reports/            # Artefatos textuais em formato .md
│   └── tables/             # Tabelas estruturadas com métricas (CSV)
├── scripts/                # Automação do ambiente (bash)
├── src/                    # Código modular das fases do pipeline Python
└── tests/                  # Scripts de testes unitários automatizados
```

## Setup do Ambiente

O projeto requer Python 3.9+ e utiliza um ambiente virtual.

```bash
# 1. Torne os scripts executáveis
chmod +x scripts/*.sh

# 2. Inicialize o Virtual Environment e instale dependências
./scripts/setup_venv.sh

# 3. Ative o ambiente
source .venv/bin/activate
```

## Uso

O orquestrador `main.py` roda as fases isoladas do projeto ou o pipeline completo. Ele lida com todas as etapas: depuis da auditoria dos dados, extração de métricas de Equidade, SHAP values, até inferência combinada com Regras de Rubin.

```bash
# Executar a fase de auditoria
PYTHONPATH=. python3 src/main.py --stage data_audit

# Executar todas as fases consecutivamente (pode ser demorado)
./scripts/run_all.sh

# Levantar a interface web visual (Streamlit)
./scripts/run_dashboard.sh
```

## Metodologia & Reproduzibilidade

* **Combate a Data Leakage:** Fronteiras quantitativas (quartis ESCS e Performance Criativa) aprendidas estritamente no conjunto de Treino.
* **Rubin's Rules Pooling:** Os algoritmos preditivos (XGBoost) são instanciados e treinados independentemente sobre PV1 a PV10, tendo suas *probabilities* agrupadas.
* **Fay BRR (Replicate Weights):** Erros padrões computados repetindo as métricas nos 80 conjuntos de replicação propostos pela OCDE, assegurando a confiabilidade das generalizações.
* **Explainable AI:** Diagnósticos de explicabilidade calculados via aproximações `shap`.

---
*Desenvolvido como pesquisa acadêmica focada na mitigação das disparidades em Machine Learning Educacional.*
