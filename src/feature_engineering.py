"""
Feature Engineering Pipeline
Run: python src/feature_engineering.py

Reads data/processed/clean.csv, engineers features,
saves data/processed/features.csv and models/preprocessor.pkl
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys

from sklearn.preprocessing import MinMaxScaler

INPUT_PATH       = 'data/processed/clean.csv'
OUTPUT_PATH      = 'data/processed/features.csv'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'

RISK_FEATURES = {
    'equipment_health_score':  -1,
    'storm_risk_index':         1,
    'maintenance_overdue_days': 1,
    'inspection_score':        -1,
    'voltage_fluctuation_pct':  1,
    'oil_quality_score':       -1,
}


def drop_temp_cols(df):
    temp_cols = ['age_bin', 'load_bin', 'storm_bin']
    existing = [c for c in temp_cols if c in df.columns]
    if existing:
        df.drop(columns=existing, inplace=True)
        print(f'[drop temp]  {existing}')
    return df


def create_interactions(df):
    df['age_x_maintenance_overdue'] = df['asset_age_years'] * df['maintenance_overdue_days']
    df['load_x_health']             = df['load_utilization_pct'] * df['equipment_health_score']
    df['temp_x_storm']              = df['transformer_temperature_c'] * df['storm_risk_index']
    df['voltage_x_overdue']         = df['voltage_fluctuation_pct'] * df['maintenance_overdue_days']
    print('[interactions] age_x_maintenance_overdue, load_x_health, temp_x_storm, voltage_x_overdue')
    return df


def create_composite_risk(df, features_weights):
    scaler = MinMaxScaler()
    cols = list(features_weights.keys())
    scaled = pd.DataFrame(
        scaler.fit_transform(df[cols]),
        columns=cols, index=df.index
    )
    df['composite_risk_score'] = sum(
        scaled[col] * w for col, w in features_weights.items()
    )
    means = df.groupby('grid_failure_flag')['composite_risk_score'].mean()
    assert means[1] > means[0], 'Composite risk score must be higher for failed assets'
    print(f'[risk score] composite_risk_score created  (fail mean={means[1]:.3f}, ok mean={means[0]:.3f})')
    return df, scaler


def encode_categoricals(df):
    cat_cols = [c for c in ['asset_type', 'substation_region', 'manufacturer'] if c in df.columns]
    before = df.shape[1]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    print(f'[ohe]        {before} → {df.shape[1]} columns after encoding {cat_cols}')
    return df


def run():
    if not os.path.exists(INPUT_PATH):
        print(f'ERROR: {INPUT_PATH} not found. Run src/data_cleaning.py first.')
        sys.exit(1)

    df = pd.read_csv(INPUT_PATH)
    print(f'[load]       {df.shape[0]:,} rows × {df.shape[1]} cols')

    df = drop_temp_cols(df)
    df = create_interactions(df)
    df, scaler = create_composite_risk(df, RISK_FEATURES)
    df = encode_categoricals(df)

    print(f'[final]      {df.shape[0]:,} rows × {df.shape[1]} cols  |  missing={df.isnull().sum().sum()}')

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'[save]       {OUTPUT_PATH}')

    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
    with open(PREPROCESSOR_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'[save]       {PREPROCESSOR_PATH}')


if __name__ == '__main__':
    run()
