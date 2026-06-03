# TODO — PROJETO PISA 2022 Resiliência Criativa (PISA 2022 Brasil)

## Planejado (seguindo o plano aprovado + ajustes metodológicos)

- [x] Criar estrutura de pastas do projeto (config/, project/, src/, tests/, dashboard/, outputs/, models/, docs/, scripts/).
- [ ] Criar `src/` com utilitários base: seed global, detecção automática do dataset `data/`, logging, config, metadata.
- [x] Fase 1: `src/data_audit.py`
- [x] Fase 2: `src/variable_discovery.py`
- [x] Fase 3: `src/leakage_detector.py`
  - [ ] Gate obrigatório (nenhuma modelagem antes desta etapa)
  - [ ] Gerar `outputs/reports/leakage_audit.md`
  - [ ] Gerar `outputs/tables/leakage_candidates.csv`
- [x] Fase 4: `src/target_builder.py`
  - [ ] Definições A/B/C/D
  - [ ] Comparar prevalência/balanceamento/estabilidade
  - [ ] Gerar `outputs/reports/target_comparison.md`
- [x] Fase 5–12: EDA, perfil, clusterer, modeling, SHAP, fairness, robustness, dashboard básico
- [x] Data dictionary: `outputs/reports/data_dictionary.md`
- [ ] Weighted analysis: criar rotinas que comparam com pesos `W_FSTUWT`
- [ ] Reprodutibilidade: config.yaml, metadata.json, seed, checksums
- [ ] Testes: suíte completa em `tests/`
- [ ] Artefatos finais: README.md, PROJECT_SUMMARY.md, DOCUMENTATION.md, ROADMAP.md

