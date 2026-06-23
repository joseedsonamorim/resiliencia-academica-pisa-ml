# Análise de Sensibilidade Multi-Target (Fase 12 — SR-7)

- Dataset: `pisa_brasil_estudo_limpo.csv`
- Targets avaliados: A, B, C, D
- Features: 4
- Protocolo: holdout 20% + CV 5×2
- Modelos no screening: 8
- Data leakage controlado (CC-1): limiares calculados exclusivamente no treino

## Justificativa metodológica

A validade de constructo de *resiliência criativa* depende de como o constructo é operacionalizado. Diferentes limiares (A/B/C/D) capturam aspectos distintos do fenômeno e geram prevalências diferentes. Se os achados são robustos entre definições, a conclusão principal é mais forte; se variam, a escolha de operacionalização deve ser justificada com base na teoria e reportada transparentemente (Cook & Campbell, 1979; Messick, 1995).

## Resumo de resultados por target

> **Critério primário de seleção:** CV Average Precision (PR-AUC) no treino.
> O holdout AUC-ROC é reportado como métrica secundária de comparabilidade.

```csv
target,status,prevalence,n_total,n_pos,n_train,n_test,best_model,cv_ap_mean,cv_ap_std,cv_auc_mean,holdout_ap,holdout_auc,holdout_brier
A,ok,0.04381846635367762,3834,168,3067,767,gaussian_nb,0.2006183186022516,0.04577960671732533,0.8750167036856457,0.1821631038648549,0.8800729594163246,0.05454486349567037
B,ok,0.06416275430359937,3834,246,3067,767,gaussian_nb,0.26639831291333244,0.03277669617811904,0.8646332384679978,0.1856418930021419,0.8415382866238417,0.07746098577249114
C,ok,0.018779342723004695,3834,72,3067,767,knn_distance,0.11743095518703064,0.0787643092038091,0.747963758387519,0.05807504772453459,0.7865248226950355,0.02006463692801845
D,ok,0.30099113197704747,3834,1154,3067,767,logistic_regression,0.6731707544822866,0.03882180732247899,0.8155076669105519,0.7127995983458287,0.8394415778560061,0.1675818445406535
```

## Interpretação de robustez

**BAIXA ROBUSTEZ** — A variação de CV AP entre targets é de 0.5557 (> 0.15). As conclusões são sensíveis à operacionalização do target; a escolha de target A deve ser rigorosamente justificada teoricamente e as diferenças entre targets devem ser discutidas extensivamente.

## Rank Stability Index (Spearman ρ entre rankings de modelos)

> Correlação de Spearman dos rankings dos modelos por CV AP entre pares de targets.
> ρ ≥ 0.80 indica alta concordância (os modelos são ordenados de forma similar entre targets).
> ρ < 0.50 indica que a escolha do target muda qual modelo é melhor — sinal de alerta.

```csv
target_1,target_2,spearman_rho,p_value,interpretation
A,B,0.6190476190476191,0.1017330374542648,Concordância moderada
A,C,-0.38095238095238104,0.3518125531175649,Baixa concordância — achados sensíveis à operacionalização
A,D,0.5000000000000001,0.20703124999999997,Concordância moderada
B,C,-0.4285714285714286,0.2894032248467902,Baixa concordância — achados sensíveis à operacionalização
B,D,0.8571428571428572,0.006530017254715293,Alta concordância
C,D,-0.2380952380952381,0.5701563208157682,Baixa concordância — achados sensíveis à operacionalização
```

## Recomendações para o manuscrito

1. **Reportar target primário:** justificar a escolha de target A com base na literatura (ex.: prevalência similar à da literatura PISA; alinhamento com Q1/Q3).
2. **Análise de sensibilidade:** incluir esta tabela como Apêndice ou Supplementary Material.
3. **Interpretação diferencial:** se targets produzirem modelos muito diferentes, discutir as implicações teóricas de cada operacionalização.
4. **Transparência:** reportar que o mesmo pipeline (sem otimização específica por target) foi aplicado a todas as definições.

## Figuras

- `outputs/figures/sensitivity/sensitivity_comparison.png` — métricas por target
- `outputs/figures/sensitivity/sensitivity_model_heatmap.png` — AP por modelo × target
