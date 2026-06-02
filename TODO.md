# TODO — Auditoria Científica Avançada (Melhorias Complementares)

> Regra: **não alterar arquitetura/pipeline/models/dashboard existentes**. Apenas adicionar **módulos novos** e **páginas novas**.

## Etapa 0 — Preparação
- [x] Ler arquivos base para compatibilidade: `run_all.py`, `app.py`, módulos `src/*` indicados.
- [x] Garantir dependências necessárias para análises (incl. `fairlearn`).

## Etapa 1 — Melhoria 1: Permutation Audit (Leakage indireto)
- [ ] Implementar `src/permutation_audit.py` (Permutation Importance + alert se >30%).
- [ ] Gerar outputs:
  - [ ] `outputs/reports/permutation_importance_report.md`
  - [ ] `outputs/tables/permutation_importance.csv`
  - [ ] `outputs/figures/permutation_importance.png`

## Etapa 2 — Melhoria 2: Holdout externo real
- [ ] Implementar `src/external_holdout.py` (70/30 stratified, reusar pipeline atual).
- [ ] Gerar outputs:
  - [ ] `outputs/reports/external_validation.md`
  - [ ] `outputs/tables/external_holdout_results.csv`

## Etapa 3 — Melhoria 3: Perfil do aluno criativamente resiliente
- [ ] Implementar `src/resilient_profile.py` (Cohen's d + radar/heatmap).
- [ ] Gerar outputs:
  - [ ] `outputs/reports/resilient_profile.md`
  - [ ] `outputs/tables/resilient_profile.csv`
  - [ ] `outputs/figures/resilient_profile_radar.png`
  - [ ] `outputs/figures/resilient_profile_heatmap.png`

## Etapa 4 — Melhoria 4: Interpretação de clustering
- [ ] Implementar `src/cluster_interpretation.py` e gerar `outputs/reports/cluster_interpretation.md`.

## Etapa 5 — Melhoria 5: Novas páginas no dashboard
- [ ] Atualizar `app.py` para incluir páginas **16, 17, 18** no menu (sem alterar páginas existentes).
- [ ] Página 16: Mapa das Variáveis (treemap/sunburst/barras).
- [ ] Página 17: Pipeline Científico (fluxograma + métricas de etapas).
- [ ] Página 18: Perfil do Aluno Resiliente (radar/heatmap + texto).

## Etapa 6 — Melhoria 6: Relatório de robustez final
- [ ] Implementar `outputs/reports/scientific_robustness_report.md` via módulo novo (sem tocar no pipeline atual).

## Etapa 7 — Validação final
- [ ] Rodar `python3 run_all.py` (não muda) ou um novo script auxiliar opcional para gerar complementos.
- [ ] Verificar que páginas novas renderizam sem quebrar.

