# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Módulo 2: Modelagem (Fases 3 e 4 do KDD) - Pensamento Criativo PISA 2022.

Fase 3: K-Means nos Resiliente_Criativo==1 (arquétipos criativos).
Fase 4: RF Classifier + SMOTE treino + SHAP explicabilidade para prever Resiliente_Criativo.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap

__rastreio_ml__ = "jeas_pisa_modelos_2026_ufrpe"

CREATIVITY_FEATURES_KMEANS = ['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']  # PV1CREA removido real data não tem
CREATIVITY_FEATURES_RF = ['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']  # Sem leak PV1CREA

def kmeans_resilientes_criativos(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, KMeans]:
    """Fase 3 KDD (Clustering): K-Means em alunos Resiliente_Criativo==1.
    
    Identifica arquétipos de resiliência criativa aplicando K-Means
    APENAS na população de alunos resilientes criativos (baixa renda + 
    alto desempenho criativo).
    
    Features utilizadas (padronizadas):
    - ESCS: Índice Status Socioeconômico
    - HISEI: Ocupação parental (ISEI)
    - HOMEPOS: Posse bens/dispositivos em casa
    - ST29Q01: Repetência escolar (binária)
    - IC004Q01: Internet em casa (binária)
    
    Arquétipos (interpretação):
    - Cluster 0: Criativo+Recursos (ESCS alto, HOMEPOS alto)
    - Cluster 1: Infra-Escola (IC004Q01 alto, suporte educacional)
    - Cluster 2: Motivado (motivation proxies)
    - Cluster 3: Desvantagens (ESCS baixo, ST29Q01 alto)
    
    Args:
        df: DataFrame limpo com coluna Resiliente_Criativo (binária).
        n_clusters: Número de clusters a identificar (padrão 4).
        
    Returns:
        Tuple[df_clusters, kmeans_model]:
        - df_clusters: DataFrame resilientes com colunas 'cluster' e 'arquétipo'
        - kmeans_model: Modelo KMeans treinado
        
    Raises:
        ValueError: Se resilientes_criativos < n_clusters
        
    Example:
        >>> df_clusters, kmeans = kmeans_resilientes_criativos(df, n_clusters=4)
        >>> print(df_clusters.groupby('arquétipo').size())
    """
    df_res = df[df['Resiliente_Criativo'] == 1].copy()
    if len(df_res) < n_clusters:
        raise ValueError(f"Resilientes criativos insuficientes: {len(df_res)} < {n_clusters}")
    
    X = df_res[CREATIVITY_FEATURES_KMEANS].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_res['cluster'] = kmeans.fit_predict(X_scaled)
    
    archetypes = {
        0: 'Criativo+Recursos', 1: 'Infra-Escola', 
        2: 'Motivado', 3: 'Desvantagens'
    }
    df_res['arquétipo'] = df_res['cluster'].map(archetypes)
    
    print(f"✅ K-Means Fase 3 concluído:")
    for i in range(n_clusters):
        count = (df_res['cluster'] == i).sum()
        print(f"   Cluster {i} ({archetypes.get(i, 'N/A')}): {count} alunos")
    
    return df_res, kmeans

def rf_classifier(df: pd.DataFrame, cv_folds: int = 5) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Fase 4 KDD (Classificação): RF prevendo Resiliente_Criativo com SMOTE + Cross-Validation.
    
    Features: socio + repetência/infra (sem PV1CREA para evitar leak).
    SMOTE apenas treino; RF balanced; validação cruzada (cv=5).
    
    Args:
        df: DataFrame limpo com target Resiliente_Criativo.
        cv_folds: Número de folds para validação cruzada (padrão 5).
        
    Returns:
        model (RF treinado), X_train_balanced, X_test, report_dict (com CV scores e métricas).
    """
    X = df[CREATIVITY_FEATURES_RF].fillna(0)
    y = df['Resiliente_Criativo']
    
    # Split treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # SMOTE no treino apenas
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    # Modelo Random Forest
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced', 
        random_state=42, n_jobs=-1
    )
    
    # Treinar modelo final
    model.fit(X_train_bal, y_train_bal)
    
    # Predições teste
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas simples
    test_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    test_precision = precision_score(y_test, y_pred, zero_division=0)
    test_recall = recall_score(y_test, y_pred, zero_division=0)
    try:
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        test_roc_auc = 0.0
    
    # Validação cruzada (CV=5)
    cv_splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Função scoring customizada com SMOTE
    def cv_with_smote(model, X, y, cv):
        cv_scores = {'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}
        for train_idx, val_idx in cv_splitter.split(X, y):
            X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # SMOTE apenas CV treino
            X_cv_train_bal, y_cv_train_bal = smote.fit_resample(X_cv_train, y_cv_train)
            
            # Treinar
            model_cv = RandomForestClassifier(
                n_estimators=200, max_depth=10, class_weight='balanced', 
                random_state=42, n_jobs=-1
            )
            model_cv.fit(X_cv_train_bal, y_cv_train_bal)
            
            # Prever
            y_val_pred = model_cv.predict(X_cv_val)
            y_val_proba = model_cv.predict_proba(X_cv_val)[:, 1]
            
            # Calcular métricas
            cv_scores['f1'].append(f1_score(y_cv_val, y_val_pred, zero_division=0))
            cv_scores['precision'].append(precision_score(y_cv_val, y_val_pred, zero_division=0))
            cv_scores['recall'].append(recall_score(y_cv_val, y_val_pred, zero_division=0))
            try:
                cv_scores['roc_auc'].append(roc_auc_score(y_cv_val, y_val_proba))
            except:
                cv_scores['roc_auc'].append(0.0)
        
        return cv_scores
    
    cv_results = cv_with_smote(model, X_train, y_train, cv_splitter)
    
    # Compilar relatório
    report = {
        'test': test_report,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_roc_auc': test_roc_auc,
        'cv_f1_mean': np.mean(cv_results['f1']),
        'cv_f1_std': np.std(cv_results['f1']),
        'cv_precision_mean': np.mean(cv_results['precision']),
        'cv_precision_std': np.std(cv_results['precision']),
        'cv_recall_mean': np.mean(cv_results['recall']),
        'cv_recall_std': np.std(cv_results['recall']),
        'cv_roc_auc_mean': np.mean(cv_results['roc_auc']),
        'cv_roc_auc_std': np.std(cv_results['roc_auc']),
        'cv_scores_detailed': cv_results
    }
    
    print(f"✅ Random Forest treinado:")
    print(f"   Test F1-score: {test_f1:.3f} | Precision: {test_precision:.3f} | Recall: {test_recall:.3f} | ROC-AUC: {test_roc_auc:.3f}")
    print(f"   CV F1-score: {report['cv_f1_mean']:.3f} ± {report['cv_f1_std']:.3f}")
    print(f"   CV ROC-AUC: {report['cv_roc_auc_mean']:.3f} ± {report['cv_roc_auc_std']:.3f}")
    
    return model, pd.DataFrame(X_train_bal, columns=CREATIVITY_FEATURES_RF), X_test, report

def shap_explainer(model: Any, X_train_sample: pd.DataFrame) -> Tuple[np.ndarray, Any]:
    """Fase 4 Explicabilidade: SHAP TreeExplainer.
    
    Args:
        model: RF treinado.
        X_train_sample: Amostra treino para compute (classe positiva).
    Returns:
        shap_values[1] (classe 1), explainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sample)
    return shap_values[1] if isinstance(shap_values, list) else shap_values, explainer

