# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

"""Módulo 1: Pré-processamento (Fase 2 KDD) - PISA 2022 Pensamento Criativo.

Carrega SAS7BDAT reais STU_QQQ_SAS + CRT_SAS, merge CNT+CNTSTUID,
filtro LATAM, limpeza, Resiliente_Criativo. Fallback mock.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional
import pyreadstat

__rastreio_mestrado__ = "jeas_pisa_dados_2026_ufrpe"

LATAM_COUNTRIES = ['BRA', 'CHL', 'COL', 'MEX']

def load_real_sas_data() -> pd.DataFrame:
    """Carrega arquivos SAS reais e faz merge interno.
    
    Socio: data/STU_QQQ_SAS/CY08MSP_STU_QQQ.SAS7BDAT (ESCS, HISEI, HOMEPOS, ST29Q01)
    CRT: data/CRT_SAS/CY08MSP_CRT_COG.SAS7BDAT (PV1CRT)
    Merge: ['CNT', 'CNTSTUID']
    """
    stu_path = '/Users/macbookair/Documents/GitHub/resiliencia-academica-pisa-ml/data/STU_QQQ_SAS/CY08MSP_STU_QQQ.SAS7BDAT'
    crt_path = '/Users/macbookair/Documents/GitHub/resiliencia-academica-pisa-ml/data/CRT_SAS/CY08MSP_CRT_COG.SAS7BDAT'
    
    if not os.path.exists(stu_path) or not os.path.exists(crt_path):
        raise FileNotFoundError("Arquivos SAS ausentes. Descompacte STU_QQQ_SAS.zip / CRT_SAS.zip.")
    
    # Ler socioeconômico
    print("Carregando STU_QQQ...")
    df_stu, _ = pyreadstat.read_dta(stu_path)
    
    # Ler criatividade (PV1CRT primeiro plausible)
    print("Carregando CRT_COG...")
    df_crt, _ = pyreadstat.read_dta(crt_path)
    
    # Merge interno chaves
    df_merged = pd.merge(
        df_stu[['CNT', 'CNTSTUID', 'ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01']],
        df_crt[['CNT', 'CNTSTUID', 'PV1CRT']],
        on=['CNT', 'CNTSTUID'],
        how='inner'
    )
    print(f"Merge: {len(df_merged)} registros combinados.")
    
    # Filtro LATAM
    df_merged = df_merged[df_merged['CNT'].isin(LATAM_COUNTRIES)]
    print(f"Filtros LATAM: {len(df_merged)} linhas.")
    
    return df_merged

def generate_mock_data(n_samples: int = 2000) -> pd.DataFrame:
    """Fallback mock se SAS ausentes (compatibilidade)."""
    np.random.seed(42)
    n_bra = int(0.3 * n_samples)
    n_chl = int(0.25 * n_samples)
    n_col = int(0.25 * n_samples)
    n_mex = n_samples - n_bra - n_chl - n_col
    
    data = {
        'CNT': ['BRA']*n_bra + ['CHL']*n_chl + ['COL']*n_col + ['MEX']*n_mex,
        'CNTSTUID': range(1, n_samples + 1),
        'ESCS': np.random.normal(-0.6, 0.6, n_samples),
        'HISEI': np.random.normal(3.8, 1.2, n_samples),
        'HOMEPOS': np.random.normal(0.2, 0.8, n_samples),
        'ST29Q01': np.random.binomial(1, 0.3, n_samples),
        'PV1CRT': np.random.normal(450, 75, n_samples)
    }
    df = pd.DataFrame(data)
    df[['ESCS', 'HISEI', 'HOMEPOS', 'PV1CRT']] = df[['ESCS', 'HISEI', 'HOMEPOS', 'PV1CRT']].clip(-2, 2)
    return df

def clean_data(df: pd.DataFrame, creativity_col: str = 'PV1CRT') -> pd.DataFrame:
    """Limpeza: dropna criatividade, impute socio mediana/CNT, cria Resiliente_Criativo."""
    df = df.copy()
    
    # Drop nulos criatividade
    initial_n = len(df)
    df = df.dropna(subset=[creativity_col])
    print(f"Removidos {initial_n - len(df)} sem {creativity_col}.")
    
    # Impute mediana por CNT
    socio_cols = ['ESCS', 'HISEI', 'HOMEPOS']
    for col in socio_cols:
        medians = df.groupby('CNT')[col].median()
        df[col] = df.apply(lambda row: medians[row['CNT']] if pd.isna(row[col]) else row[col], axis=1)
    
    # Resiliente_Criativo por país quartis
    q1_escs = df.groupby('CNT')['ESCS'].quantile(0.25)
    q4_crea = df.groupby('CNT')[creativity_col].quantile(0.75)
    df['Resiliente_Criativo'] = (
        (df['ESCS'] <= df['CNT'].map(q1_escs)) &
        (df[creativity_col] >= df['CNT'].map(q4_crea))
    ).astype(int)
    
    print(f"Resilientes Criativos: {df['Resiliente_Criativo'].sum()} ({df['Resiliente_Criativo'].mean():.1%})")
    return df

def load_data(file_path: Optional[str] = None) -> pd.DataFrame:
    """Carrega SAS reais ou mock.
    
    Prioridade: SAS merge → CSV fallback → mock.
    """
    __rastreio_mestrado__ = "jeas_pisa_dados_2026_ufrpe"  # Rastreador obrigatório
    
    try:
        if not file_path:
            df = load_real_sas_data()
        else:
            # CSV/Parquet fallback
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_parquet(file_path)
            df = df[df['CNT'].isin(LATAM_COUNTRIES)]
        
        df_clean = clean_data(df, 'PV1CRT')
        
    except (FileNotFoundError, KeyError) as e:
        print(f"SAS indisponível ({e}), usando mock.")
        df_mock = generate_mock_data()
        df_clean = clean_data(df_mock, 'PV1CRT')
    
    return df_clean

