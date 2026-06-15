# Clusterer científico — Perfis de Resiliência Criativa (PISA 2022)

## Seleção das Variáveis

- Features teóricas selecionadas: **80**
- Features após remoção de colinearidade: **80**
- Critérios: person-centered e sem leakage (exclusões por leakage, target e variáveis sensíveis a vazamento).

## Redução de Dimensionalidade

- PCA retido: **58** componentes (alvo 0.85)
- Artefatos: `outputs/figures/clustering/scree_plot.png` e `outputs/figures/clustering/cumulative_variance.png`

## Comparação dos Algoritmos

Tabela completa de métricas e estabilidade (top ranking por cluster_score):

```csv
algorithm,k,silhouette,davies_bouldin,calinski_harabasz,ari_mean,ari_std,jaccard_mean,jaccard_std,cluster_score,ranking_final
agglomerative_ward,2,0.32408959623397027,3.1714130019809534,106.71330709694483,0.08553563095295172,0.16105737085620422,0.8740260365347818,0.03414300129144632,0.686488902643945,1.0
agglomerative_ward,7,0.2129124292638656,2.6223684656686195,88.49797075907463,0.29371635324964857,0.10690661778103505,0.5482294654228164,0.05711987375920861,0.39573930730147516,2.0
agglomerative_ward,3,0.2270603158328897,3.8161758155678562,104.57702481101684,0.1027619401994317,0.11071024505428868,0.655089073782684,0.04824665806010721,0.3700723383549639,3.0
agglomerative_ward,6,0.21299422300287174,3.0392302267412403,91.98209842838855,0.255561749499636,0.11416908784575565,0.5506060584970532,0.04918707595317531,0.36950953735950753,4.0
birch,2,0.3862550392503515,3.392696390549647,97.71124753249657,,,,,0.35667230925038695,5.0
gmm,2,0.2730795270379486,6.0607391976317455,85.38634875399232,,,,,0.33548223354634965,6.0
agglomerative_ward,5,0.21030190237797808,3.2074826766307822,96.33165939267079,0.2080395906241698,0.1159834323514752,0.5476169267242024,0.04076626307357667,0.31661586378971085,7.0
agglomerative_ward,4,0.21222377996878444,3.687446032248566,100.62695020105036,0.15355929995707399,0.10530990941710003,0.5458686625702375,0.03585124603109661,0.26805178240237804,8.0
birch,3,0.19682166528970774,4.221692996397777,95.29794248307778,,,,,0.2605472661643996,9.0
birch,4,0.1991020561104891,3.8987454166543625,91.99272466253386,,,,,0.24337150890295617,10.0
spectral,2,0.12676048501239884,6.046028776404232,71.22966683553287,,,,,0.20864902529473808,11.0
birch,5,0.1459178262644634,3.32940467168181,89.92283109942608,,,,,0.18853426318573424,12.0
agglomerative_ward,8,0.19662511064936739,2.511544361470426,86.31396768395564,,,,,0.18495765925142982,13.0
agglomerative_ward,9,0.17549359494788674,2.8032107384773997,84.69599983087201,,,,,0.17661799419593854,14.0
birch,6,0.14874679778242875,3.1839810852772206,86.12944567964067,,,,,0.17572405324193385,15.0
```

## Estabilidade dos Clusters

- Bootstrap ARI/Jaccard implementado com `cluster_stability.csv`

```csv
algorithm,k,ari_mean,ari_std,jaccard_mean,jaccard_std,stability_mean,bootstrap_cluster_count_mean,bootstrap_noise_rate_mean
agglomerative_ward,2,0.08553563095295172,0.16105737085620422,0.8740260365347818,0.03414300129144632,0.7083969260056289,2.0,0.0
agglomerative_ward,3,0.1027619401994317,0.11071024505428868,0.655089073782684,0.04824665806010721,0.6032350219411999,3.0,0.0
agglomerative_ward,4,0.15355929995707399,0.10530990941710003,0.5458686625702375,0.03585124603109661,0.5613241562743873,4.0,0.0
agglomerative_ward,5,0.2080395906241698,0.1159834323514752,0.5476169267242024,0.04076626307357667,0.5758183610181437,5.0,0.0
agglomerative_ward,6,0.255561749499636,0.11416908784575565,0.5506060584970532,0.04918707595317531,0.5891934666234356,6.0,0.0
agglomerative_ward,7,0.29371635324964857,0.10690661778103505,0.5482294654228164,0.05711987375920861,0.5975438210238203,7.0,0.0
```

## Solução Escolhida

- Modelo: **agglomerative_ward** com `k=2`
- Justificativa: seleção por score multicritério (qualidade estrutural + estabilidade via ARI/Jaccard) — não apenas silhouette.

## Perfis Identificados

```csv
cluster_id,cluster_name,size,prevalence,top_above,top_below
0,Alta — Engajamento Criativo,3589,0.9360980699008868,CR590Q14S;CR590Q04S;CR590Q22S;ICTRES;CR590Q23S;CR590Q05S;CR590Q03S;CR590Q40TT;CR590Q36TT;CR590Q26TT,CR590Q03TT;CR590Q33TT;CR590Q43TT;CR590Q14TT;CR590Q38TT;CR590Q20TT;CR590Q27TT;CR590Q23TT;CR590Q47TT;CR590Q22TT
1,Alta — Engajamento Criativo,245,0.0639019300991132,CR590Q03TT;CR590Q33TT;CR590Q43TT;CR590Q14TT;CR590Q38TT;CR590Q20TT;CR590Q27TT;CR590Q23TT;CR590Q47TT;CR590Q22TT,CR590Q14S;CR590Q04S;CR590Q22S;ICTRES;CR590Q23S;CR590Q05S;CR590Q03S;CR590Q40TT;CR590Q36TT;CR590Q26TT
```

## Associação com Resiliência Criativa

Distribuição de resilientes por perfil (odds ratio e risco relativo):

```csv
cluster_id,n,n_resilientes,prevalence_resilientes,odds_ratio,risk_relative,global_resilient_prevalence
1,245,14,0.05714285714285714,1.389494949579325,1.367238095316241,0.04277516953573292
0,3589,150,0.041794371691278906,0.7196859551758386,0.731401504555577,0.04277516953573292
```

## Variáveis Mais Relevantes

Quais variáveis distinguem os perfis (SHAP + Permutation Importance):



## Limitações

- Natureza descritiva (person-centered) — não causalidade.
- Estabilidade depende de amostragem bootstrap.
- HDBSCAN é opcional; quando indisponível, rótulos ficam em -1 e métricas podem ser menos informativas.

## Implicações Educacionais

- Perfis sugerem intervenções diferenciadas por padrão psicossocial/engajamento: tecnologia, apoio familiar, clima escolar e processos de tarefa.

## Artefatos principais

- `outputs/tables/clustering_model_comparison.csv`
- `outputs/tables/cluster_stability.csv`
- `outputs/tables/cluster_profiles_interpretable.csv`
- `outputs/tables/cluster_resilience_association.csv`
- `outputs/tables/cluster_shap_importance.csv`
- `outputs/figures/clustering/`
