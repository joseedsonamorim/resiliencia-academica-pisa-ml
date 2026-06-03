# Target comparison (A/B/C/D)

- Dataset: pisa_brasil_estudo_limpo.csv
- ESCS proxy: **ESCS**
- CRT proxy: **CRT_SCORE**

## Estatísticas
| target_def   |    n |   n_pos |   prevalence |
|:-------------|-----:|--------:|-------------:|
| A            | 3834 |     164 |    0.0427752 |
| B            | 3834 |     248 |    0.0646844 |
| C            | 3834 |      70 |    0.0182577 |
| D            | 3834 |    1150 |    0.299948  |

## Notas das definições
A: ESCS <= Q1 (Q0.25=-1.6970) e CRT >= Q3 (Q0.75=0.4925)
B: ESCS <= P30 (P0.30=-1.5200) e CRT >= P70 (P0.70=0.4762)
C: ESCS <= Q1 (Q0.25=-1.6970) e CRT >= P90 (P0.90=0.5606)
D: score composto (CRT z - ESCS z) com corte em P70
