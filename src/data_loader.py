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
from pyreadstat import ReadstatError

__rastreio_mestrado__ = "jeas_pisa_dados_2026_ufrpe"

LATAM_COUNTRIES = ['BRA', 'CHL', 'COL', 'MEX']

def load_real_sas_data() -> pd.DataFrame:
    """Carrega arquivos SAS reais e faz merge interno.
    
    Socio: data/STU_QQQ_SAS/CY08MSP_STU_QQQ.SAS7BDAT (ESCS, HISEI, HOMEPOS, ST29Q01)
    CRT: data/CRT_SAS/CY08MSP_CRT_COG.SAS7BDAT (PV1CRT)
    Merge: ['CNT', 'CNTSTUID']
    """
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stu_path = os.path.join(base_path, 'data', 'STU_QQQ_SAS', 'CY08MSP_STU_QQQ.SAS7BDAT')
    crt_path = os.path.join(base_path, 'data', 'CRT_SAS', 'CY08MSP_CRT_COG.SAS7BDAT')
    
    if not os.path.exists(stu_path) or not os.path.exists(crt_path):
        raise FileNotFoundError("Arquivos SAS ausentes. Descompacte STU_QQQ_SAS.zip / CRT_SAS.zip.")
    
    # Ler socioeconômico
    print("Carregando STU_QQQ...")
    df_stu, _ = pyreadstat.read_sas7bdat(stu_path)
    
    # Ler criatividade (PV1CRT primeiro plausible)
    print("Carregando CRT_COG...")
    df_crt, _ = pyreadstat.read_sas7bdat(crt_path)
    
    # Merge interno chaves
    df_merged = pd.merge(
        df_stu[['CNT', 'CNTSTUID', 'ESCS', 'HISEI', 'HOMEPOS', 'ST29Q01', 'IC004Q01']],
        df_crt[['CNT', 'CNTSTUID', 'PV1CRT']],
        on=['CNT', 'CNTSTUID'],
        how='inner'
    )

    # Padronizar nome para o app (PV1CREA)
    if 'PV1CRT' in df_merged.columns and 'PV1CREA' not in df_merged.columns:
        df_merged = df_merged.rename(columns={'PV1CRT': 'PV1CREA'})

    print(f"Merge: {len(df_merged)} registros combinados.")
    
    # Filtro LATAM
    df_merged = df_merged[df_merged['CNT'].isin(LATAM_COUNTRIES)]
    print(f"Filtros LATAM: {len(df_merged)} linhas.")
    
    return df_merged

def generate_mock_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Gera mock data PISA LATAM com correlações Gaussian copula realistas.
    
    Implementa:
    - ESCS-PV1CREA: ρ ≈ 0.30 (correlação moderada realista)
    - HOMEPOS-ESCS: ρ ≈ 0.80 (desigualdade digital forte)
    - ST29Q01 (Repetência): P condicional em ESCS
    - IC004Q01 (Internet): P condicional em ESCS
    - ESCS por país: BRA μ=-1.1, CHL μ=-0.6, COL μ=-0.8, MEX μ=-1.0
    
    Args:
        n_samples: Total amostras geradas.
        seed: Random seed para reprodutibilidade.
        
    Returns:
        DataFrame com 2000 amostras LATAM correlacionadas.
    """
    from scipy.stats import multivariate_normal, norm
    
    np.random.seed(seed)
    
    # Distribuição países
    n_bra = int(0.30 * n_samples)
    n_chl = int(0.25 * n_samples)
    n_col = int(0.25 * n_samples)
    n_mex = n_samples - n_bra - n_chl - n_col
    
    countries = ['BRA'] * n_bra + ['CHL'] * n_chl + ['COL'] * n_col + ['MEX'] * n_mex
    
    # Parâmetros por país (ESCS mean, std)
    country_params = {
        'BRA': {'escs_mean': -1.1, 'escs_std': 0.65},
        'CHL': {'escs_mean': -0.6, 'escs_std': 0.55},
        'COL': {'escs_mean': -0.8, 'escs_std': 0.60},
        'MEX': {'escs_mean': -1.0, 'escs_std': 0.63}
    }
    
    # Gerar dados base com correlações
    # Copula gaussiana: N(0,1) com correlação especificada
    rho_escs_pv1crea = 0.30  # Correlação ESCS-PV1CREA (realista PISA)
    rho_homepos_escs = 0.80  # Correlação HOMEPOS-ESCS (forte)
    
    # Matriz covariância 3D: [ESCS_norm, PV1CREA_norm, HOMEPOS_norm]
    cov_matrix = np.array([
        [1.0, rho_escs_pv1crea, rho_homepos_escs],
        [rho_escs_pv1crea, 1.0, rho_escs_pv1crea * 0.7],  # HOMEPOS correlaciona menos com PV1CREA
        [rho_homepos_escs, rho_escs_pv1crea * 0.7, 1.0]
    ])
    
    # Ensure positive definite
    try:
        np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        cov_matrix = np.eye(3)  # Fallback
    
    # Gerar amostras gaussianas correlacionadas
    samples_norm = multivariate_normal.rvs(
        mean=[0, 0, 0], cov=cov_matrix, size=n_samples
    )
    
    # Converter para ESCS, PV1CREA, HOMEPOS via inverse CDF
    escs_norm = samples_norm[:, 0]
    pv1crea_norm = samples_norm[:, 1]
    homepos_norm = samples_norm[:, 2]
    
    # Aplicar transformação por país
    escs_list = []
    for i, country in enumerate(countries):
        params = country_params[country]
        # Usar normal CDF para transformar N(0,1) → N(μ, σ)
        escs_val = norm.ppf(norm.cdf(escs_norm[i]), 
                            loc=params['escs_mean'], 
                            scale=params['escs_std'])
        escs_list.append(np.clip(escs_val, -3, 3))
    
    # PV1CREA: [300, 700] range PISA
    pv1crea = 450 + pv1crea_norm * 75
    pv1crea = np.clip(pv1crea, 300, 700)
    
    # HOMEPOS: [-3, 3] com correlação
    homepos = homepos_norm
    homepos = np.clip(homepos, -3, 3)
    
    # HISEI: Ocupação parental (menos correlado)
    hisei = np.random.normal(3.8, 1.2, n_samples)
    hisei = np.clip(hisei, 0, 100)
    
    # ST29Q01 (Repetência): Probabilidade condicional em ESCS
    st29q01 = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        escs_val = escs_list[i]
        if escs_val <= -1.0:
            p_repetencia = 0.30  # 30% chance
        elif escs_val >= 0.5:
            p_repetencia = 0.10  # 10% chance
        else:
            p_repetencia = 0.20  # 20% chance (interpolado)
        st29q01[i] = np.random.binomial(1, p_repetencia)
    
    # IC004Q01 (Internet em casa): Probabilidade condicional em ESCS
    ic004q01 = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        escs_val = escs_list[i]
        if escs_val <= -1.0:
            p_internet = 0.50  # 50% chance
        elif escs_val >= 0.5:
            p_internet = 0.90  # 90% chance
        else:
            p_internet = 0.70  # 70% chance (interpolado)
        ic004q01[i] = np.random.binomial(1, p_internet)
    
    # Montar DataFrame
    df = pd.DataFrame({
        'CNT': countries,
        'CNTSTUID': range(1, n_samples + 1),
        'ESCS': escs_list,
        'HISEI': hisei,
        'HOMEPOS': homepos,
        'ST29Q01': st29q01,
        'IC004Q01': ic004q01,
        'PV1CREA': pv1crea
    })
    
    print(f"✅ Mock data gerado: {len(df)} amostras")
    print(f"   ESCS-PV1CREA correlação: {df[['ESCS', 'PV1CREA']].corr().iloc[0,1]:.3f} (target ≈ 0.30)")
    print(f"   HOMEPOS-ESCS correlação: {df[['HOMEPOS', 'ESCS']].corr().iloc[0,1]:.3f} (target ≈ 0.80)")
    print(f"   Distribuição países: BRA {(df['CNT']=='BRA').sum()}, CHL {(df['CNT']=='CHL').sum()}, "
          f"COL {(df['CNT']=='COL').sum()}, MEX {(df['CNT']=='MEX').sum()}")
    
    return df

def clean_data(df: pd.DataFrame, creativity_col: str = 'PV1CREA') -> pd.DataFrame:
    """Limpeza: dropna criatividade, impute socio mediana/CNT, clip outliers, cria Resiliente_Criativo.
    
    Args:
        df: DataFrame bruto com dados PISA.
        creativity_col: Nome coluna criatividade (padrão PV1CREA).
        
    Returns:
        DataFrame limpo com target Resiliente_Criativo.
    """
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
    
    # Clip outliers
    df[['ESCS', 'HISEI', 'HOMEPOS']] = df[['ESCS', 'HISEI', 'HOMEPOS']].clip(-3, 3)
    df[creativity_col] = df[creativity_col].clip(300, 700)  # Range PISA
    print(f"Outliers clippados (ESCS/HOMEPOS: [-3,3], {creativity_col}: [300,700])")
    
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
    """Carrega dados PISA com fallback automático: SAS → CSV → Mock.
    
    Prioridade de carregamento:
    1. SAS7BDAT reais (data/STU_QQQ_SAS/ + data/CRT_SAS/)
    2. CSV/Parquet em file_path (se fornecido)
    3. Mock data LATAM correlacionada (fallback)
    
    Args:
        file_path: Caminho CSV/Parquet opcional. Se None, tenta SAS → mock.
        
    Returns:
        DataFrame limpo com colunas:
        - CNT, CNTSTUID, ESCS, HISEI, HOMEPOS, ST29Q01, IC004Q01, PV1CREA
        - Resiliente_Criativo (target binário)
        - Filtrado LATAM (BRA, CHL, COL, MEX)
        
    Raises:
        Nada. Sempre retorna dados (SAS → CSV → mock).
        
    Example:
        >>> df = load_data()
        >>> print(df.shape, df['Resiliente_Criativo'].mean())
        (2000, 9) 0.068
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
        
        # Se vier com PV1CRT, padroniza
        if 'PV1CRT' in df.columns and 'PV1CREA' not in df.columns:
            df = df.rename(columns={'PV1CRT': 'PV1CREA'})

        df_clean = clean_data(df, 'PV1CREA')
        
    except (FileNotFoundError, KeyError, ReadstatError, ValueError) as e:
        print(f"⚠️ SAS indisponível ({type(e).__name__}: {e}), usando mock.")
        df_mock = generate_mock_data()
        df_clean = clean_data(df_mock, 'PV1CREA')
    
    return df_clean

