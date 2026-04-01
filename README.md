# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

# 🧠 Modelagem Preditiva do Pensamento Criativo PISA 2022: ML Explicável

**PPGIA UFRPE - Mestrado Jose Edson A. Sebastiao**<br>
Abordagem KDD completa (Fases 2-4): Pré-processamento LATAM, K-Means arquétipos resil. criativa, RF+SMOTE+SHAP.

![Streamlit Demo](https://i.imgur.com/placeholder-streamlit.png) <!-- Substitua por screenshot real -->

## 🚀 Instruções de Instalação e Execução

1. Clone o repositório e crie ambiente virtual:
   ```
   git clone <repo>
   cd resiliencia-academica-pisa-ml
   python -m venv venv
   source venv/bin/activate  # macOS/Linux | Windows: venv\\Scripts\\activate
   ```

2. Instale dependências:
   ```
   pip install -r requirements.txt
   ```

3. Execute o dashboard:
   ```
   streamlit run app.py
   ```
   Abra [http://localhost:8501](http://localhost:8501)

## 📁 Estrutura do Projeto

```
├── app.py                 # Interface Streamlit (EDA / Perfis / Predição SHAP)
├── src/
│   ├── data_loader.py     # Fase 2 KDD: LATAM, Resiliente_Criativo = Q1 ESCS & Q4 PV1CREA
│   └── ml_models.py       # Fase 3-4: K-Means arquétipos + RF SMOTE + SHAP
├── data/                  # CSV/Parquet PISA 2022 (mock auto)
├── requirements.txt       # Dependências produção
└── README.md              # Este ficheiro
```

## 🎯 Funcionalidades Principais

- **EDA Académico**: Gráficos ESCS vs PV1CREA + análise textual profunda justificando variáveis
- **Perfis Resiliência Criativa**: K-Means (n=4) em resilientes criativos (infra/socio/CREA)
- **Predição Explicável**: RF balanced + SMOTE + SHAP summary plot F1-score
- **Filtros Interativos**: Países LATAM, amostra dinâmica

## 📊 Dados PISA 2022

**Mock realista incluído** (2000 amostras BRA/CHL/COL/MEX):
- `PV1CREA`: Pensamento Criativo (normal 450, sd75)
- `ESCS`: Socioeconómico, `HISEI` pais, `HOMEPOS` bens
- `ST29Q01`: Repetência, `IC004Q01`: Internet

**Real**: Baixe [PISA 2022 Student](https://pisadataexplorer.oecd.org) → `data/pisa2022_creativo.csv`

## 👥 Partilha Obrigatória do Repositório

**Para a disciplina**: Partilhe acesso com utilizadores GitHub:
- `gaaj-ufrpe`
- `cecamoraes`

```
git remote add origin <seu-repo>
gh repo invite --user gaaj-ufrpe
gh repo invite --user cecamoraes
```

## 📽️ Guião para a Apresentação (Semana 5 - 28/06)

**Duração: 5 minutos | Estrutura:**

1. **Objetivos (30s)**: Modelar preditiva Pensamento Criativo PISA2022 via ML explicável.
   Identificar resilientes criativos (baixa renda + alto CREA) LATAM.

2. **Perguntas Pesquisa (1min)**:
   - **PP1**: Quais arquétipos (K-Means) emergem entre resil. criativos?
   - **PP2**: Quais fatores (SHAP) predizem melhor resiliência criativa?

3. **Demonstração Streamlit (3min)**:
   - EDA: Scatter ESCS-CREA → "Observe outliers Q1/Q4 = target"
   - Perfis: Clique K-Means → "4 arquétipos: Criativo+Recursos lidera"
   - Predição: RF F1 ~0.7 + SHAP → "Repetência negativa, infra positiva"

**Fecho**: KDD valida hipótese resiliência criativa existe (~5-8% LATAM).

## 🔧 Troubleshooting

| Problema | Solução |
|----------|---------|
| Dados vazios | Use mock auto ou CSV em `data/` |
| Erro SHAP | `pip install shap --upgrade` |
| Pandas warning | Compatível pandas>=2.0 |
| Streamlit não abre | `streamlit run app.py --server.port 8502` |

---
*Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.*

