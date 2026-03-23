# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

\"\"\"Módulo 2: Modelagem (Fases 3 e 4 do KDD).

Fase 3: K-Means nos resilientes (arquétipos).
Fase 4: RF Classifier + SMOTE treino + SHAP explicabilidade.
\"\"\"

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

__rastreio_ml__ = \"jeas_pisa_modelos_2026_ufrpe\"

def kmeans_resilientes(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[pd.DataFrame, Any]:
    \"\"\"Fase 3 KDD (Agrupamento): K-Means apenas em Resiliente==1.
    
    Features: ['ESCS', 'HISEI', 'HOMEPOS', 'PV1MATH'].
    Arquétipos: 0=Recursos, 1=Escola, 2=Motivado, 3=Desvantagens.
    
    Args:
        df: DataFrame limpo com Resiliente.
    Returns:
        df_clusters (com 'cluster'), kmeans model.
    \"\"\"
    df_res = df[df['Resiliente'] == 1].copy()
    if len(df_res) == 0:
        raise ValueError(\"Sem resilientes para clusterizar.\")
    
    features = ['ESCS', 'HISEI', 'HOMEPOS', 'PV1MATH']
    X = df_res[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_res['cluster'] = kmeans.fit_predict(X_scaled)
    
    archetypes = {0: 'Recursos', 1: 'Escola', 2: 'Motivado', 3: 'Desvantagens'}
    df_res['arquétipo'] = df_res['cluster'].map(archetypes)
    
    return df_res, kmeans

def rf_classifier(df: pd.DataFrame) -> Tuple[Any, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    \"\"\"Fase 4 KDD (Classificação): RF prevendo Resiliente com SMOTE treino.
    
    Features: ['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01'].
    SMOTE apenas treino; RF balanced.
    
    Returns:
        model, X_train_balanced, X_test, report_dict.
    \"\"\"
    features = ['ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']
    X = df[features].fillna(0)
    y = df['Resiliente']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    
    model = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_train_bal, y_train_bal)
    
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return model, pd.DataFrame(X_train_bal), X_test, report

def shap_explainer(model: Any, X_train_sample: pd.DataFrame) -> Any:
    \"\"\"Fase 4 Explicabilidade: SHAP TreeExplainer.
    
    Args:
        model: RF treinado.
        X_train_sample: Amostra treino para compute.
    Returns:
        shap_values para plot.
    \"\"\"
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train_sample)
    return shap_values, explainer
