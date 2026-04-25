# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Utilitários para processamento PISA - Pensamento Criativo 2026.

Funções auxiliares para codificação de variáveis, normalização, etc.
Atualizado para Resiliência Criativa (PV1CREA) vs legado PV1MATH.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

def get_features() -> list:
    """Features socioeconômicas principais.
    
    Returns:
        Lista de features socio para análise.
    """
    return ['ESCS', 'HISEI', 'HOMEPOS']

def encode_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variável gênero ST01Q01 em binária.
    
    Args:
        df: DataFrame com coluna ST01Q01 (Female/Male).
        
    Returns:
        DataFrame com coluna ST01Q01_encoded (1=Female, 0=Male).
    """
    df = df.copy()
    if 'ST01Q01' in df.columns:
        df['ST01Q01_encoded'] = (df['ST01Q01'] == 'Female').astype(int)
    return df

def normalize_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Normaliza features com StandardScaler.
    
    Args:
        X: DataFrame ou array com features.
        
    Returns:
        X_scaled (normalizado), scaler (para inversão futura).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


