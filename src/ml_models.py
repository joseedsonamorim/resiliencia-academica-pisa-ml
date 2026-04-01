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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import shap

__rastreio_ml__ = "jeas_pisa_modelos_2026_ufrpe"

CREATIVITY_FEATURES_KMEANS = ['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']  # PV1CREA removido real data não tem
CREATIVITY_FEATURES_RF = ['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']  # Sem leak PV1CREA

def kmeans_resilientes_criativos(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, Any]:
    """Fase 3 KDD (Agrupamento): K-Means apenas em Resiliente_Criativo==1.
    
    Features: infraestrutura (HOMEPOS/IC004Q01/ST29Q01) + socio + PV1CREA.
    Arquétipos criatividade: 0=Criativo+Recursos, 1=Infra-Escola, 2=Motivado, 3=Desvantagens.
    
    Args:
        df: DataFrame limpo com Resiliente_Criativo.
    Returns:
        df_clusters (com 'cluster', 'arquétipo'), kmeans model.
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
    
    return df_res, kmeans

def rf_classifier(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Fase 4 KDD (Classificação): RF prevendo Resiliente_Criativo com SMOTE treino.
    
    Features: socio + repetência/infra (sem PV1CREA para evitar leak).
    SMOTE apenas treino; RF balanced.
    
    Returns:
        model, X_train_balanced, X_test, report_dict.
    """
    X = df[CREATIVITY_FEATURES_RF].fillna(0)
    y = df['Resiliente_Criativo']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight='balanced', 
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_bal, y_train_bal)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
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

