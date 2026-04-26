# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Utilitários de cache - Pensamento Criativo PISA 2022.

Funções para salvar/carregar modelos treinados com joblib.
"""

import os
from typing import Any, Optional
import joblib
from pathlib import Path
from datetime import datetime

MODELS_DIR = "models"

def ensure_models_dir() -> str:
    """Cria diretório models/ se não existir.
    
    Returns:
        Caminho para diretório models/.
    """
    Path(MODELS_DIR).mkdir(exist_ok=True)
    return MODELS_DIR

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
