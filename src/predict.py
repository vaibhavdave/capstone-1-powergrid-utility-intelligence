"""
Prediction / Scoring Script
Run: python src/predict.py --input data/raw/new_assets.csv --output outputs/predictions.csv

Scores a new CSV of assets and assigns risk tiers.
The input CSV must have the same columns as the raw training data.
"""
import argparse
import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler

MODEL_PATH        = 'models/best_model.pkl'
PREPROCESSOR_PATH = 'models/preprocessor.pkl'
THRESHOLD_PATH    = 'models/optimal_threshold.pkl'

RISK_FEATURE_WEIGHTS = {
    'equipment_health_score':  -1,
    'storm_risk_index':         1,
    'maintenance_overdue_days': 1,
    'inspection_score':        -1,
    'voltage_fluctuation_pct':  1,
    'oil_quality_score':       -1,
}

ADMIN_COLS   = ['legacy_asset_code', 'grid_cluster_id',
                'monitoring_batch_id', 'administrative_reference', 'asset_id']
OUTLIER_COLS = ['rainfall_mm', 'avg_outage_duration_minutes', 'power_load_mw']
IMPUTE_COLS  = ['oil_quality_score', 'inspection_score']


def assign_risk_tier(prob):
    if prob >= 0.70:   return 'HIGH'
    elif prob >= 0.45: return 'MEDIUM'
    return 'LOW'


def apply_cleaning(df):
    df['asset_type'] = df['asset_type'].str.strip().str.title()
    df = df.drop_duplicates()
    for col in IMPUTE_COLS:
        if col in df.columns:
            df[col] = df.groupby('asset_type')[col].transform(lambda x: x.fillna(x.median()))
    for col in OUTLIER_COLS:
        if col in df.columns:
            lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lo, upper=hi)
    to_drop = [c for c in ADMIN_COLS if c in df.columns]
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    return df


def apply_features(df, scaler):
    df['age_x_maintenance_overdue'] = df['asset_age_years'] * df['maintenance_overdue_days']
    df['load_x_health']             = df['load_utilization_pct'] * df['equipment_health_score']
    df['temp_x_storm']              = df['transformer_temperature_c'] * df['storm_risk_index']
    df['voltage_x_overdue']         = df['voltage_fluctuation_pct'] * df['maintenance_overdue_days']

    feat_cols = list(RISK_FEATURE_WEIGHTS.keys())
    scaled = pd.DataFrame(
        scaler.transform(df[feat_cols]),
        columns=feat_cols, index=df.index
    )
    df['composite_risk_score'] = sum(
        scaled[col] * w for col, w in RISK_FEATURE_WEIGHTS.items()
    )

    cat_cols = [c for c in ['asset_type', 'substation_region', 'manufacturer'] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def run(input_path, output_path):
    for path in [MODEL_PATH, PREPROCESSOR_PATH, THRESHOLD_PATH]:
        if not os.path.exists(path):
            print(f'ERROR: {path} not found. Run src/train.py first.')
            sys.exit(1)

    with open(MODEL_PATH, 'rb') as f:         model = pickle.load(f)
    with open(PREPROCESSOR_PATH, 'rb') as f:  scaler = pickle.load(f)
    with open(THRESHOLD_PATH, 'rb') as f:     threshold = pickle.load(f)

    df_raw = pd.read_csv(input_path)
    print(f'Loaded {len(df_raw):,} assets from {input_path}')

    df = apply_cleaning(df_raw.copy())
    df = apply_features(df, scaler)

    # Align columns with training feature set
    train_cols = pd.read_csv('data/processed/features.csv', nrows=0).drop(
        columns=['grid_failure_flag'], errors='ignore'
    ).columns.tolist()

    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_cols]

    probs = model.predict_proba(df)[:, 1]
    preds = (probs >= threshold).astype(int)
    tiers = [assign_risk_tier(p) for p in probs]

    df_raw = df_raw.iloc[:len(df)].copy()
    df_raw['failure_probability'] = probs
    df_raw['predicted_failure']   = preds
    df_raw['risk_tier']           = tiers

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df_raw.to_csv(output_path, index=False)

    print(f'Scored {len(df_raw):,} assets → {output_path}')
    print(df_raw['risk_tier'].value_counts().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score grid assets for failure risk.')
    parser.add_argument('--input',  required=True,  help='Path to input CSV (raw asset data)')
    parser.add_argument('--output', default='outputs/predictions.csv', help='Path to output CSV')
    args = parser.parse_args()
    run(args.input, args.output)
