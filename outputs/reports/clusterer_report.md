# Clusterer (Fase 7)

- Dataset: `pisa_brasil_estudo_limpo.csv`
- Features usadas: 50
- k / componentes: 4
- Melhor algoritmo (silhouette): **hierarchical**

## Métricas

| algorithm    |   n_clusters |   silhouette |   calinski_harabasz |   davies_bouldin |
|:-------------|-------------:|-------------:|--------------------:|-----------------:|
| hierarchical |            4 |     0.289634 |             177.623 |          2.61746 |
| kmeans       |            4 |     0.217256 |             241.716 |          2.31929 |
| gmm          |            4 |     0.049729 |             124.21  |          3.06549 |

## Artefatos

- `outputs/tables/clusterer_metrics.csv`
- `outputs/tables/clusterer_labels.csv`
