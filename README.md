# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

# 🧠 Análise de Pensamento Criativo PISA 2022 com ML Explicável

**PPGIA UFRPE - Mestrado Jose Edson A. Sebastiao**  
Abordagem KDD completa (Fases 2-4): Pré-processamento LATAM, agrupamento de arquétipos via K-Means, predição com Random Forest + SMOTE + interpretabilidade SHAP, dashboard interativo Streamlit.

---

## 🎯 Objetivo

Identificar e modelar **alunos resilientes criativos** (status socioeconômico baixo + pensamento criativo alto) em países latinoamericanos (Brasil, Chile, Colômbia, México) usando dados PISA 2022, com foco em interpretabilidade através de SHAP.

---

## ✨ Características Principais

### 🖥️ **Interface Streamlit (Design Apple-Style)**
- **Layout Moderno**: Paleta branco/cinza claro com tipografia sistema Apple
- **4 Abas Interativas**:
  1. **📈 Exploração de Dados (EDA)**: Scatter plot ESCS vs Criatividade + histogramas
  2. **👥 Perfis de Resiliência**: Clustering K-Means com 4 arquétipos + características
  3. **🎯 Predição e SHAP**: Random Forest + explicabilidade TreeExplainer
  4. **📥 Exportação**: Download de resultados + cache de modelos

### 📊 **Análise de Dados**
- **Variáveis Principais**:
  - `ESCS`: Índice socioeconômico (Economic, Social and Cultural Status)
  - `PV1CREA`: Pensamento Criativo (plausible value)
  - `HISEI`: Status ocupacional dos pais
  - `HOMEPOS`: Posse de bens em casa
  - `ST29Q01`: Repetência escolar
  - `IC004Q01`: Acesso à internet
  - `Resiliente_Criativo`: Target (Q1 ESCS & Q4 PV1CREA = 1)

- **Mock Automático**: 2000 amostras com correlações realistas (ESCS-PV1CREA ρ≈0.30, HOMEPOS-ESCS ρ≈0.80)
- **Suporte Dados Reais**: Carrega SAS files do PISA ou CSV/Parquet

### 🤖 **Machine Learning**
- **K-Means Clustering**: Agrupa alunos resilientes criativos em 4 perfis distintos
- **Random Forest + SMOTE**: Modelo balanceado com performance F1: 0.68, ROC-AUC: 0.868
- **SHAP TreeExplainer**: Interpretação feature importance via shapley values
- **Validação Cruzada**: 5-fold stratified para estabilidade

---

## 🚀 Instruções de Instalação

### 1️⃣ **Clone e Ambiente Virtual**
```bash
git clone <seu-repo>
cd resiliencia-academica-pisa-ml
python -m venv venv
source venv/bin/activate  # macOS/Linux
# Windows: venv\Scripts\activate
```

### 2️⃣ **Instale Dependências**
```bash
pip install -r requirements.txt
```

### 3️⃣ **Execute o Dashboard**
```bash
streamlit run app.py
```
Abra no navegador: **[http://localhost:8501](http://localhost:8501)**

---

## 📁 Estrutura do Projeto

```
├── app.py                      # Interface Streamlit (4 abas + sidebar)
├── src/
│   ├── data_loader.py          # Carregamento e limpeza de dados PISA
│   ├── ml_models.py            # K-Means, Random Forest, SHAP
│   └── export_utils.py         # Persistência de modelos com joblib
├── models/                     # Cache de modelos treinados
├── data/                       # CSV/Parquet/SAS (opcional, mock default)
├── requirements.txt            # Dependências Python
├── README.md                   # Este arquivo
└── venv/                       # Ambiente virtual
```

---

## 📋 Dependências

| Pacote | Versão | Propósito |
|--------|--------|----------|
| `streamlit` | ≥1.28.0 | Framework web interativo |
| `pandas` | ≥2.0.0 | Manipulação de dados |
| `scikit-learn` | ≥1.3.0 | ML (K-Means, Random Forest) |
| `imbalanced-learn` | ≥0.11.0 | SMOTE para balanceamento |
| `shap` | ≥0.43.0 | Explicabilidade SHAP |
| `matplotlib` | ≥3.7.0 | SHAP plots |
| `altair` | ≥5.0.0 | Gráficos interativos |
| `pyreadstat` | ≥1.1.0 | Leitura de SAS files |
| `joblib` | ≥1.3.0 | Serialização de modelos |

---

## 🔄 Fluxo de Dados (Fases KDD)

```
Dados PISA 2022
    ↓
Fase 2: PRÉ-PROCESSAMENTO (data_loader.py)
  • Seleção: BRA/CHL/COL/MEX
  • Engenharia: Resiliente_Criativo = (ESCS ≤ Q1) & (PV1CREA ≥ Q4)
  • Limpeza: Clipping [-3,3], remoção missing
    ↓
Fase 3: MODELAGEM (ml_models.py)
  • K-Means: n_clusters=4 em resilientes criativos
  • Random Forest: 5-fold CV + SMOTE balancing
    ↓
Fase 4: INTERPRETAÇÃO (SHAP)
  • TreeExplainer: Importância de features
  • Summary plot: Visualização SHAP
    ↓
Dashboard Streamlit: 4 abas interativas
```

---

## 🎯 Uso do Dashboard

### **Sidebar - Controles**
- 🗂️ **Guia de Uso**: Modo passo-a-passo (expandível)
- 🎯 **Filtros**: Selecione países, tamanho de amostra
- ⚙️ **Executar Análises**: Botões para K-Means e Random Forest
- 📌 **Sobre**: Informações do projeto

### **KPI Cards**
- Número de estudantes filtrados
- % Alunos resilientes criativos
- Média ESCS (socioeconomia)
- Média PV1CREA (criatividade)

### **Aba 1: Exploração de Dados**
- **Scatter plot interativo**: ESCS vs PV1CREA (zoom, hover)
- **Histograma**: Distribuição de criatividade
- Análise descritiva por país

### **Aba 2: Perfis de Resiliência**
- **Gráfico**: Distribuição entre 4 arquétipos
- **Tabela**: Características de cada perfil (média/std)
- Preview dos dados agrupados

### **Aba 3: Predição e SHAP**
- **Métricas**: F1-Score, ROC-AUC, Recall
- **SHAP Summary Plot**: Importância global de features
- Cross-validation metrics

### **Aba 4: Exportação**
- **CSV Download**: Dados processados + previsões
- **Model Cache**: Salvar/carregar modelos treinados

---

## 📊 Dados PISA 2022

### **Mock Automático** (Recomendado para Testes)
- 2000 amostras distribuídas entre BRA (600), CHL (500), COL (500), MEX (400)
- Correlações realistas via Gaussian Copula
- Outliers clippados automaticamente

### **Dados Reais** (Uso Avançado)
Baixe em [PISA Data Explorer](https://pisadataexplorer.oecd.org):
1. Selecione PISA 2022 Student Data
2. Exporte para CSV/SAS
3. Coloque em `data/` (app detecta automaticamente)

---

## 🎨 Design & UX

### **Paleta de Cores (Apple-Style)**
- **Fundo**: Branco puro (#FFFFFF)
- **Texto Primário**: Preto (#000000)
- **Texto Secundário**: Cinza escuro (#6E7681)
- **Ação Principal**: Azul corporativo (#0A66C2)
- **Alertas**: Verde (sucesso), azul (info), amarelo (aviso), vermelho (erro)

### **Tipografia**
- Font Stack: `-apple-system, BlinkMacSystemFont, 'Segoe UI'`
- Headings: 600 weight, -0.3px letter-spacing
- Espaçamento: 2.5rem top padding, gaps harmoniosos

### **Componentes**
- KPI Cards: Bordas 1px, border-radius 10px, hover effects
- Botões: Rounded corners 8px, estados hover com shadow
- Tabs: Indicador azul para ativo
- Expandables: Fundo branco, borda cinza

---

## 🔧 Troubleshooting

| Problema | Solução |
|----------|---------|
| **Dados vazios** | Verifique `data/` ou deixe mock gerar automaticamente |
| **ImportError: shap** | `pip install shap --upgrade` |
| **Pandas FutureWarning** | Normal com pandas ≥2.0, compatível |
| **Streamlit não abre** | `streamlit run app.py --server.port 8502` (porta alternativa) |
| **SHAP plot não aparece** | Confirme Random Forest treinou (clique botão 🤖 Treinar) |
| **Lentidão ao carregar** | Reduza amostra no slider ou use dados filtrados |

---

## 📈 Resultados Esperados

### **Estatísticas PISA 2022 (Mock)**
- **Total de Estudantes**: ~2000
- **Resilientes Criativos**: ~3.5% (71 estudantes)
- **Random Forest - F1-Score**: 0.68
- **Random Forest - ROC-AUC**: 0.868
- **K-Means - Arquétipos**: 4 grupos distintos

### **Insights SHAP**
- Features mais importantes: `ESCS`, `HOMEPOS`, `HISEI`
- Tendência: Maior criatividade em alunos com recursos familiares e status socioeconômico alto
- Anomalia: Pequeno grupo resiliente (baixa renda, alta criatividade)

---

## 👥 Contato & Atribuição

**Autor**: Jose Edson A. Sebastiao  
**Programa**: PPGIA - Mestrado em Inteligência Artificial  
**Universidade**: UFRPE - Universidade Federal Rural de Pernambuco  
**Data**: Abril 2026

---

## 📝 Licença

**Copyright (c) 2026 Jose Edson Amorim Sebastiao**  
Todos os direitos reservados.

---

## ✅ Checklist de Funcionalidades

- [x] Dashboard Streamlit com 4 abas
- [x] Carregamento de dados PISA com mock automático
- [x] K-Means clustering (4 arquétipos)
- [x] Random Forest com SMOTE balancing
- [x] SHAP TreeExplainer
- [x] Design Apple-style
- [x] Filtros interativos (países, amostra)
- [x] Gráficos interativos (Altair, Matplotlib)
- [x] Exportação CSV + cache de modelos
- [x] Contraste visual 100% adequado
- [x] Responsividade web completa

