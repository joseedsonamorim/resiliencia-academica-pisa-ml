# Target comparison (A/B/C/D) - Plausible Values Pooled

- Dataset: PISA2022_BRASIL_FULL.csv
- ESCS proxy: **ESCS**
- CRT proxy: **10 Plausible Values (Rubin's Rules)**
- Pesos amostrais (W_FSTUWT): **sim**
- N referência: 10,798 estudantes

## Estatísticas de prevalência (Média dos PVs)
```csv
target_def,n,n_pos,prevalence
A,10798,314.4,0.029116503056121505
B,10798,528.4,0.048934987960733464
C,10798,87.1,0.00806630857566216
D,10798,3144.6,0.29122059640674197
```

## Nota metodológica 1A
A resiliência foi computada independentemente para cada um dos 10 *Plausible Values* da OCDE. O limiar (P75, P90, etc) é recalculado 10 vezes. A estatística final apresentada é o pooled average das 10 definições. No treinamento de Machine Learning, os modelos devem prever e combinar esses 10 vetores de resposta separadamente.
