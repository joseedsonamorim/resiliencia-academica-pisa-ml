# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Utilitários de exportação e cache - Pensamento Criativo PISA 2022.

Funções para salvar/carregar modelos com joblib e gerar relatórios PDF.
"""

import os
import hashlib
import pickle
from datetime import datetime
from typing import Any, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = "models"

def ensure_models_dir() -> str:
    """Cria diretório models/ se não existir.
    
    Returns:
        Caminho para diretório models/.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)
    return MODELS_DIR

def get_data_hash(df: pd.DataFrame) -> str:
    """Calcula hash SHA256 do DataFrame para cache invalidation.
    
    Args:
        df: DataFrame para hash.
        
    Returns:
        String hash SHA256 (primeiros 8 chars).
    """
    data_bytes = pd.util.hash_pandas_object(df, index=True).values
    hash_obj = hashlib.sha256(data_bytes)
    return hash_obj.hexdigest()[:8]

def save_model(model: Any, model_name: str = "rf_model") -> str:
    """Salva modelo treinado com joblib.
    
    Args:
        model: Modelo treinado (RandomForestClassifier, KMeans, etc).
        model_name: Nome base arquivo (sem extensão).
        
    Returns:
        Caminho arquivo salvo.
    """
    ensure_models_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")
    joblib.dump(model, filename)
    print(f"✅ Modelo salvo: {filename}")
    return filename

def load_model(model_name: str = "rf_model") -> Optional[Any]:
    """Carrega modelo mais recente de dado nome.
    
    Args:
        model_name: Nome base modelo (busca variante mais recente).
        
    Returns:
        Modelo carregado ou None se não encontrado.
    """
    ensure_models_dir()
    files = [f for f in os.listdir(MODELS_DIR) if f.startswith(model_name) and f.endswith('.pkl')]
    if not files:
        return None
    latest = sorted(files)[-1]
    path = os.path.join(MODELS_DIR, latest)
    model = joblib.load(path)
    print(f"✅ Modelo carregado: {latest}")
    return model

def generate_pdf_report(df: pd.DataFrame, model_metrics: dict, clusters_info: dict = None) -> bytes:
    """Gera PDF com relatório análise (placeholder para reportlab).
    
    Args:
        df: DataFrame processado.
        model_metrics: Dict com F1, Precision, Recall, ROC-AUC.
        clusters_info: Dict opcional com info clusters.
        
    Returns:
        Bytes PDF (placeholder).
    """
    # TODO: Implementar com reportlab quando disponível
    report_text = f"""
    RELATÓRIO - MODELAGEM PREDITIVA PENSAMENTO CRIATIVO PISA 2022
    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    DADOS:
    - Total amostras: {len(df)}
    - Resiliência Criativa: {df['Resiliente_Criativo'].mean():.1%}
    - ESCS média: {df['ESCS'].mean():.2f}
    - PV1CREA média: {df['PV1CREA'].mean():.0f}
    
    MÉTRICAS RANDOM FOREST:
    - F1-Score: {model_metrics.get('test_f1', 0):.3f}
    - Precision: {model_metrics.get('test_precision', 0):.3f}
    - Recall: {model_metrics.get('test_recall', 0):.3f}
    - ROC-AUC: {model_metrics.get('test_roc_auc', 0):.3f}
    - CV F1 (5-fold): {model_metrics.get('cv_f1_mean', 0):.3f} ± {model_metrics.get('cv_f1_std', 0):.3f}
    
    CLUSTERS:
    {clusters_info if clusters_info else 'N/A'}
    """
    return report_text.encode('utf-8')

def export_csv_predictions(df: pd.DataFrame, predictions: np.ndarray, 
                          probabilities: np.ndarray = None) -> str:
    """Exporta CSV com dados + predições.
    
    Args:
        df: DataFrame original.
        predictions: Array predições (0/1).
        probabilities: Array probabilidades [0..1] (opcional).
        
    Returns:
        Caminho arquivo CSV.
    """
    df_export = df.copy()
    df_export['pred_Resiliente_Criativo'] = predictions
    if probabilities is not None:
        df_export['prob_Resiliente_Criativo'] = probabilities
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"predictions_{timestamp}.csv"
    df_export.to_csv(filename, index=False)
    print(f"✅ Exportado: {filename}")
    return filename
