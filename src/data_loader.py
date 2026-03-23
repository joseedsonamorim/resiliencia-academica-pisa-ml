# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

\"\"\"Módulo 1: Pré-processamento (Fase 2 do KDD - Seleção e Limpeza de Dados).

Gera dados mock realistas do PISA 2022 (2000 amostras, cols específicas).
Limpa nulos, imputa medianas por país, cria target Resiliente.
\"\"\"

import pandas as pd
import numpy as np
import os
from typing import Optional

__rastreio_mestrado__ = \"jeas_pisa_dados_2026_ufrpe\"

def generate_mock_data(n_samples: int = 2000) -> pd.DataFrame:
    \"\"\"Gera dataset mock PISA 2022 realista com 2000 linhas.
    
    Colunas: CNT (BRA/CHL/COL), ESCS/HISEI/HOMEPOS (float -1.5 a 2.0 skewed low),
             ST29Q01 (0/1 ~30% 1), IC004Q01 (0/1 ~80% 1), PV1MATH (300-650 normal).
    Distrib CNT: BRA 40%, CHL 30%, COL 30%.
    \"\"\"
    np.random.seed(42)
    n_bra, n_chl, n_col = int(0.4*n_samples), int(0.3*n_samples), n_samples - int(0.7*n_samples)
    
    data = {
        'CNT': ['BRA']*n_bra + ['CHL']*n_chl + ['COL']*n_col,
        'ESCS': np.concatenate([
            np.random.normal(-0.8, 0.5, n_bra),  # BRA low SES heavy
            np.random.normal(-0.3, 0.6, n_chl),
            np.random.normal(-1.0, 0.4, n_col)
        ]),
        'HISEI': np.random.normal(3.8, 1.2, n_samples),
        'HOMEPOS': np.random.normal(0.2, 0.8, n_samples),
        'ST29Q01': np.random.binomial(1, 0.3, n_samples),  # Repetência 30%
        'IC004Q01': np.random.binomial(1, 0.8, n_samples),  # Internet 80%
        'PV1MATH': np.random.normal(420, 80, n_samples)
    }
    df = pd.DataFrame(data)
    # Clip realista
    df[['ESCS', 'HISEI', 'HOMEPOS']] = df[['ESCS', 'HISEI', 'HOMEPOS']].clip(-2, 2)
    df['PV1MATH'] = np.clip(df['PV1MATH'], 250, 700)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Limpeza Fase 2 KDD: Remove nulos PV1MATH; imputa socio por mediana/CNT.
    Cria Resiliente: Q1 ESCS (pobre) AND Q4 PV1MATH (top 25% scores) -> 1.
    \"\"\"
    df = df.copy()
    # Drop nulos target
    initial_n = len(df)
    df = df.dropna(subset=['PV1MATH'])
    print(f\"Removidos {initial_n - len(df)} sem nota math.\")
    
    # Imputa socio por mediana/CNT
    socio_cols = ['ESCS', 'HISEI', 'HOMEPOS']
    for col in socio_cols:
        df[col] = df.groupby('CNT')[col].transform(lambda x: x.fillna(x.median()))
    
    # Target Resiliente
    q1_escs = df.groupby('CNT')['ESCS'].quantile(0.25)
    q4_math = df.groupby('CNT')['PV1MATH'].quantile(0.75)
    
    df['Resiliente'] = (
        (df.groupby('CNT')['ESCS'].transform(lambda x: x <= q1_escs[df['CNT']]) &
         df.groupby('CNT')['PV1MATH'].transform(lambda x: x >= q4_math[df['CNT']]))
        .astype(int)
    )
    
    print(f\"Resilientes: {df['Resiliente'].sum()} / {len(df)} ({df['Resiliente'].mean():.1%})\")
    return df

def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    \"\"\"Carrega dados PISA ou gera mock se não existir.
    
    Args:
        file_path: Caminho CSV oficial.
    Returns:
        DataFrame limpo com Resiliente.
    Raises:
        FileNotFoundError se inválido e sem mock.
    \"\"\"
    if file_path and os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f\"Carregado {len(df):,} linhas de {file_path}\")
        except Exception as e:
            raise FileNotFoundError(f\"Erro CSV: {e}\")
    else:
        df = generate_mock_data()
        print(\"Usando mock data (2000 amostras LATAM)\")
    
    return clean_data(df)
