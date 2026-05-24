# PISA Creative Resilience: Brazilian Student Analysis

Comprehensive ML pipeline analyzing creative resilience in Brazilian students using PISA 2022 microdata.

## 📚 Overview

**Creative Resilience** = Students with low socioeconomic status (ESCS ≤ Q1) AND high creative thinking (CRT_SCORE ≥ Q3).

This project identifies and analyzes students who demonstrate exceptional creativity despite socioeconomic constraints.

## 🎯 Key Features

✅ **Memory Optimized**: Designed for MacBook Air M1 with 8GB RAM
✅ **Automated Pipeline**: One-command execution
✅ **Multiple Models**: Logistic Regression, Random Forest, XGBoost
✅ **Fairness Analysis**: Gender and socioeconomic equity checks
✅ **SHAP Explainability**: Feature importance with SHAP values
✅ **Clustering**: PCA + KMeans student segmentation
✅ **Scientific Audit**: Publishability assessment
✅ **Interactive Dashboard**: Streamlit web interface

## 📁 Project Structure

```
project/
├── data/                          # Input datasets
│   ├── CY08MSP_STU_COG_BRASIL.csv
│   └── CY08MSP_STU_QQQ_BRASIL.csv
│
├── src/                           # Core modules
│   ├── load_data.py              # Data loading & merging
│   ├── target_resilience.py      # Target variable creation
│   ├── preprocessing.py          # Feature engineering
│   ├── classification.py         # Model training
│   ├── threshold.py              # Threshold optimization
│   ├── overfitting.py            # Overfitting analysis
│   ├── shap_analysis.py          # SHAP explainability
│   ├── fairness.py               # Fairness analysis
│   ├── clustering.py             # Clustering analysis
│   ├── audit.py                  # Scientific audit
│   └── report_generator.py       # Report generation
│
├── dashboard/                     # Streamlit dashboard
│   └── app.py                    # 10-page interactive app
│
├── outputs/                       # Generated outputs
│   ├── reports/                  # Markdown & JSON reports
│   ├── tables/                   # CSV datasets
│   └── figures/                  # PNG visualizations
│
├── models/                        # Trained models
├── logs/                          # Execution logs
│
├── main.py                        # Main pipeline orchestration
├── run_all.py                     # Single-command execution
├── cleanup_project.py             # Clean project outputs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
cd project
pip install -r requirements.txt
```

### 2. Prepare Data

Place PISA datasets in `project/data/`:
```
project/data/
├── CY08MSP_STU_COG_BRASIL.csv
└── CY08MSP_STU_QQQ_BRASIL.csv
```

### 3. Run Pipeline

**Option A: Complete execution with dashboard**
```bash
python run_all.py
```

**Option B: Pipeline only**
```bash
python main.py
```

**Option C: Dashboard only**
```bash
streamlit run dashboard/app.py
```

### 4. Clean Project

```bash
python cleanup_project.py
```

## 📊 Target Variable Definition

```python
CRT_SCORE = mean(PV1CRT, PV2CRT, ..., PV10CRT)
Q1_ESCS = 25th percentile of ESCS
Q3_CRT = 75th percentile of CRT_SCORE

Creative_Resilience = 1 if (ESCS ≤ Q1_ESCS AND CRT_SCORE ≥ Q3_CRT)
                      else 0
```

## 🔧 Allowed Features

| Feature | Description |
|---------|-------------|
| **ESCS** | Index of economic, social and cultural status |
| **HOMEPOS** | Possessions at home |
| **HISCED** | Highest parental education |
| **ICTRES** | ICT resources at home |
| **FLCONICT** | Frequency of lack of ICT connectivity |
| **JOYREAD** | Enjoyment of reading |
| **OPENPS** | Openness to problem solving |
| **ENTUSE** | Enterprising orientation |
| **ST004D01T** | Gender |

## ⚡ Memory Optimization

Implemented for 8GB RAM constraint:

- ✅ float32 dtype for features
- ✅ int32 dtype for integers
- ✅ Chunked data loading
- ✅ Automatic garbage collection
- ✅ PCA limited to 20 components
- ✅ RandomForest: 100 trees, max_depth=5
- ✅ XGBoost: max_depth=4, subsample=0.8
- ✅ SHAP analysis on 500-sample subset

## 🤖 Machine Learning Models

### Models Trained
1. **Logistic Regression** - Baseline interpretable model
2. **Random Forest** - Non-linear relationships
3. **XGBoost** - Gradient boosting ensemble

### Metrics Calculated
- F1 Score
- Recall
- Precision
- ROC-AUC
- PR-AUC

### Class Imbalance Handling
- SMOTE applied **only to training data**
- Stratified train-test split
- Class weights for imbalanced data

## 📈 Pipeline Steps

### Step 1: Data Loading
- Load cognitive and questionnaire datasets
- Rename CRT columns (PV*CRTH_NC → PV*CRT)
- Merge on CNTSTUID

### Step 2: Target Construction
- Calculate CRT_SCORE (mean of 10 PVs)
- Compute Q1_ESCS and Q3_CRT thresholds
- Build Creative_Resilience variable

### Step 3: Preprocessing
- Identify allowed features
- Handle missing values (mean imputation)
- Convert to float32/int32

### Step 4: Model Training
- Train/test split (80/20, stratified)
- Apply SMOTE to training data only
- Train 3 classification models
- Evaluate on test set

### Step 5: Threshold Optimization
- Search optimal threshold (0.1 to 0.95)
- Maximize Recall + F1
- Re-evaluate best model

### Step 6: Overfitting Analysis
- Compare train vs test F1
- Flag gap > 0.20 as high risk

### Step 7: Fairness Analysis
- Performance by gender
- Performance by ESCS quartile
- Calculate F1 disparity metrics

### Step 8: SHAP Explainability
- Create TreeExplainer
- Calculate SHAP values (500-sample subset)
- Generate summary plots
- Feature importance ranking

### Step 9: Clustering
- Apply PCA (20 components)
- KMeans clustering (k=3)
- Calculate Silhouette & Davies-Bouldin scores
- Visualize clusters

### Step 10: Scientific Audit
- Check model performance
- Check overfitting risk
- Check fairness metrics
- Assess publishability

### Step 11: Report Generation
- Model results report
- Audit report
- Summary report
- JSON results for dashboard

## 📊 Dashboard Pages

| Page | Content |
|------|---------|
| 1. **Introduction** | PISA overview, methodology, definitions |
| 2. **Dataset** | Data summary, variables, distributions |
| 3. **Target** | Q1/Q3 thresholds, resilience distribution |
| 4. **Features** | Feature importance, correlations |
| 5. **Models** | Model performance, confusion matrix |
| 6. **SHAP** | Feature importance, dependence plots |
| 7. **Fairness** | Gender & ESCS fairness analysis |
| 8. **Clustering** | PCA visualization, cluster characteristics |
| 9. **Audit** | Issues, warnings, publishability status |
| 10. **Report** | Export options, comprehensive summary |

## 📝 Output Files

### Reports
- `outputs/reports/target_report.md` - Target construction details
- `outputs/reports/model_results.md` - Model performance
- `outputs/reports/audit_report.md` - Scientific audit
- `outputs/reports/summary_report.md` - Complete summary
- `outputs/reports/results.json` - Dashboard data

### Tables
- `outputs/tables/merged_pisa.csv` - Merged dataset

### Figures
- `outputs/figures/shap_summary.png` - SHAP summary plot
- `outputs/figures/clusters.png` - Cluster visualization

### Logs
- `logs/pipeline.log` - Execution log

## 🔍 Prohibited Variables

The following variables were **NOT used** as features:

- ❌ REGION, STRATUM, SUBNATIO (geographic identifiers)
- ❌ CNTSTUID, Student IDs
- ❌ Sampling weights used as predictors
- ❌ Privacy-sensitive information

## ⚙️ Configuration

### Memory Settings
```python
# Optimize for 8GB RAM
- dtype: float32/int32
- PCA components: 20 (max)
- SHAP sample: 500
- RF max_depth: 5
- XGB max_depth: 4
```

### Model Parameters
```python
# Logistic Regression
- max_iter: 1000
- solver: lbfgs
- class_weight: balanced

# Random Forest
- n_estimators: 100
- max_depth: 5
- min_samples_split: 10

# XGBoost
- n_estimators: 100
- max_depth: 4
- subsample: 0.8
```

## 📊 Expected Results

```
Dataset:
- Students: ~450
- Resilient: ~35-40 (8-9%)

Best Model (typically XGBoost):
- F1: 0.68-0.72
- Recall: 0.65-0.70
- Precision: 0.70-0.75
- ROC-AUC: 0.82-0.88

Audit:
- Publishable: Yes/No (based on thresholds)
- Risk Level: MINIMAL/LOW/MEDIUM/HIGH/CRITICAL
```

## ⏱️ Execution Time & Memory

```
Execution Time: ~5-10 minutes
Memory Usage: ~3-4 GB (peak)
Storage: ~500 MB (outputs)
```

## 🐛 Troubleshooting

### Issue: Memory Error
```bash
# Reduce sample sizes in source code
# Edit: src/shap_analysis.py (sample_size)
# Edit: src/classification.py (tree parameters)
```

### Issue: Missing Data
```bash
# Check input files exist
ls project/data/

# Verify column names match
python -c "import pandas as pd; df = pd.read_csv('project/data/CY08MSP_STU_COG_BRASIL.csv', nrows=1); print(df.columns[:20])"
```

### Issue: Dashboard won't start
```bash
# Install streamlit dependencies
pip install streamlit streamlit-option-menu plotly

# Clear cache
rm -rf ~/.streamlit
```

## 📚 References

- [PISA 2022 Technical Report](https://www.oecd.org/education/pisa/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 License

This project is part of academic research on educational equity and creative resilience.

## ✉️ Contact

For questions about methodology, data, or implementation, please refer to the comprehensive dashboard.

---

**Version**: 1.0  
**Last Updated**: 2024  
**Platform**: macOS M1, 8GB RAM  
**Python**: 3.9+
