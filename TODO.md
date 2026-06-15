# TODO — Refatoração do Clusterer (Perfis de Resiliência Criativa)

- [ ] (Entendimento) Validar no src/clusterer.py o que já atende às Etapas 1–10 e quais itens faltam.
- [ ] (Etapa 1) Garantir seleção teórica automatizada via config.yaml, incluindo exclusões: target/CRT_SCORE/ESCS/variáveis marcadas como leakage.
- [ ] (Etapa 2) Rever pipeline de dimensionalidade: remover colinearidade extrema, padronizar, PCA e salvar explained_variance.csv + scree/cumulative plots.
- [ ] (Etapa 3) Implementar comparação de modelos: Agglomerative(Ward), GMM, HDBSCAN(optional), Spectral, Birch; rodar k=2..10.
- [ ] (Etapa 4) Implementar multicritério: Silhouette, Davies-Bouldin, Calinski-Harabasz; para GMM adicionar AIC/BIC; criar ranking multicritério e gerar clustering_model_comparison.csv.
- [ ] (Etapa 5) Implementar bootstrap clustering (n_bootstrap>=100): estabilidade média, ARI e Jaccard; gerar cluster_stability.csv.
- [ ] (Etapa 6) Gerar cluster_profiles.csv e cluster_profiles_interpretable.csv com: tamanho, prevalência, médias padronizadas (z), e desvios vs média global.
- [ ] (Etapa 7) Interpretabilidade por perfil: treinar RandomForest para prever cluster_id (person-centered) e gerar cluster_shap_importance.csv + cluster_shap_summary.png (usando SHAP + permutation importance quando possível).
- [ ] (Etapa 8) Visualizações: PCA 2D e UMAP 2D coloridos por cluster, heatmap, radar, dendrograma quando aplicável; salvar em outputs/figures/clustering/.
- [ ] (Etapa 9) Relatório científico automático: outputs/reports/clusterer_report.md com método, métricas, justificativa, estabilidade, interpretação, variáveis relevantes e limitações.
- [ ] (Etapa 10) Integração no dashboard: garantir aba “Perfis de Resiliência Criativa” exibindo comparação, estabilidade, gráficos e descrição textual dos perfis.
- [ ] Rodar `python3 -m src.main --stage clusterer` para validar execução end-to-end e existência de todos os artefatos.
- [ ] Rodar smoke check no dashboard: garantir que todos os paths referenciados existem e não quebram a UI.

