# EDA report

- Dataset: `pisa_brasil_estudo_limpo.csv`
- Linhas: 3834
- Colunas: 1279
- Duplicatas (linhas inteiras): 0

## Missingness (top 25 colunas)

|            |   missing_ratio |
|:-----------|----------------:|
| CR567Q06S  |        0.926708 |
| CR543Q13S  |        0.924622 |
| CR567Q06F  |        0.922274 |
| CR552Q04F  |        0.921492 |
| CR561Q06S  |        0.921492 |
| CR567Q11SF |        0.919405 |
| CR567Q11SD |        0.919144 |
| CR567Q13S  |        0.918884 |
| CR567Q11SC |        0.918362 |
| CR567Q11SA |        0.918101 |
| CR567Q11S  |        0.917319 |
| CR567Q08SD |        0.916797 |
| CR567Q08SE |        0.916797 |
| CR567Q08SC |        0.916275 |
| CR568Q15S  |        0.916275 |
| CR567Q10S  |        0.915754 |
| CR567Q08SB |        0.915493 |
| CR567Q08SA |        0.915232 |
| CR567Q11F  |        0.914189 |
| CR541Q11F  |        0.913928 |
| CR561Q08S  |        0.913406 |
| CR544Q14SE |        0.913146 |
| CR567Q10F  |        0.913146 |
| CR567Q08S  |        0.913146 |
| CR544Q14SC |        0.912885 |

## Descritivas (numéricas)

- Tabela: `eda_numeric_describe.csv`

## Outliers (IQR) — top 20

| column     |   outlier_iqr_count |
|:-----------|--------------------:|
| CR220Q06A  |                 299 |
| CR560Q06A  |                 296 |
| CR545Q06A  |                 240 |
| CR559Q03A  |                 240 |
| CR545Q07A  |                 239 |
| W_FSTUWT   |                 221 |
| CR559Q04A  |                 206 |
| CR220Q01V  |                 192 |
| CR220Q04A  |                 182 |
| CR220Q05A  |                 182 |
| CR220Q02F  |                 170 |
| CR560Q10A  |                 160 |
| CR560Q03A  |                 154 |
| CR545Q07F  |                 151 |
| CR220Q02V  |                 143 |
| CR545Q02A  |                 140 |
| CR559Q01A  |                 125 |
| CR543Q13A  |                 125 |
| CR590Q02TT |                 123 |
| CR590Q08TT |                 123 |

## Target distribution (se existir no dataframe)

| target_col   |      mean |   sum |
|:-------------|----------:|------:|
| target_A     | 0.0427752 |   164 |
| target_B     | 0.0646844 |   248 |
| target_C     | 0.0182577 |    70 |
| target_D     | 0.299948  |  1150 |

## Artefatos de visualização

- Missing values: `outputs/figures/missing_values.png` (fase 1)
- Data quality: `outputs/figures/data_quality.png` (fase 1)
- EDA figures: `outputs/figures/eda/*`

> Próximas etapas: Fase 6 (perfil dos resilientes) e Fase 7 (clusterização).
