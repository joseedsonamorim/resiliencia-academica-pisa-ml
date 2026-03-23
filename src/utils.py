# Copyright (c) 2026 Jose Edson Amorim Sebastiao. Todos os direitos reservados.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def resilience_score(df, read_col='PV1READ', math_col='PV1MATH', ses_col='ESCS'):
    df['resilience'] = df[read_col] + df[math_col] - df[ses_col] * 50
    return df

def create_resilience_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for subj in ['READ', 'MATH', 'SCIE']:
        pv_cols = [col for col in df.columns if col.startswith(f'PV{subj}')]
        if pv_cols:
            df[f'avg_PV{subj}'] = df[pv_cols].mean(axis=1, skipna=True)
    df['performance'] = (df.get('avg_PVREAD', 0) + df.get('avg_PVMATH', 0) + df.get('avg_PVSCIE', 0)) / 3
    df['expected_perf'] = 450 + df['ESCS'] * 100
    df['resilient'] = ((df['performance'] > df['expected_perf'] + 50) & (df['ESCS'] < 0)).astype(int)
    st.info(f\"Legacy Resilientes: {df['resilient'].mean():.1%}\")
    return df

def create_resilient_quartiles_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    escs_q1 = df['ESCS'].quantile(0.25)
    math_q4 = df['PV1MATH'].quantile(0.75)
    df['resilient_quartiles'] = ((df['ESCS'] <= escs_q1) & (df['PV1MATH'] >= math_q4)).astype(int)
    st.info(f\"Resil quartiles: {df['resilient_quartiles'].mean():.1%} ({df['resilient_quartiles'].sum()}/{len(df)})\")
    return df

def prepare_cluster_data(df: pd.DataFrame):
    df = create_resilient_quartiles_target(df.copy())
    resilient_df = df[df['resilient_quartiles'] == 1].copy()
    if len(resilient_df) < 4:
        st.warning(\"Poucos resilientes, usando todos.\")
        resilient_df = df.copy()
    features = ['ESCS', 'HISEI', 'HOMEPOS', 'PV1MATH']
    X = resilient_df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    st.info(f\"Clustering {len(X)} amostras.\")
    return X_scaled, scaler, resilient_df[features], features

def get_features():
    return ['ESCS', 'HISEI', 'HOMEPOS']

def encode_gender(df: pd.DataFrame):
    df['ST01Q01_encoded'] = (df['ST01Q01'] == 'Female').astype(int)
    return df


