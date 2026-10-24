"""
Data Cleaning Pipeline
Run: python src/data_cleaning.py

Reads raw CSV, applies all cleaning steps, saves data/processed/clean.csv
"""
import pandas as pd
import os
import sys

RAW_PATH    = 'data/raw/PowerGrid_Utility_Intelligence.csv'
OUTPUT_PATH = 'data/processed/clean.csv'

ADMIN_COLS   = ['legacy_asset_code', 'grid_cluster_id',
                'monitoring_batch_id', 'administrative_reference', 'asset_id']
OUTLIER_COLS = ['rainfall_mm', 'avg_outage_duration_minutes', 'power_load_mw']
IMPUTE_COLS  = ['oil_quality_score', 'inspection_score']


def load_data(path):
    df = pd.read_csv(path)
    print(f'[load]       {df.shape[0]:,} rows × {df.shape[1]} cols')
    return df


def normalize_asset_type(df):
    df['asset_type'] = df['asset_type'].str.strip().str.title()
    print(f'[casing]     asset_type normalized → {sorted(df["asset_type"].unique())}')
    return df


def remove_duplicates(df):
    before = len(df)
    df = df.drop_duplicates()
    print(f'[dedup]      removed {before - len(df)} duplicate rows → {len(df):,} rows')
    return df


def impute_missing(df, cols):
    for col in cols:
        n_missing = df[col].isnull().sum()
        df[col] = df.groupby('asset_type')[col].transform(lambda x: x.fillna(x.median()))
        print(f'[impute]     {col}: filled {n_missing} missing values (grouped median by asset_type)')
    return df


def winsorize(df, cols, lower_pct=0.01, upper_pct=0.99):
    for col in cols:
        lo = df[col].quantile(lower_pct)
        hi = df[col].quantile(upper_pct)
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f'[winsorize]  {col}: clipped to [{lo:.2f}, {hi:.2f}]')
    return df


def drop_admin_columns(df, admin_cols, target='grid_failure_flag'):
    present = [c for c in admin_cols if c in df.columns]
    if not present:
        return df
    corrs = df[present + [target]].corr()[target].drop(target)
    to_drop = [c for c in present if abs(corrs.get(c, 0)) < 0.05]
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    print(f'[admin drop] dropped {to_drop}')
    return df


def run():
    if not os.path.exists(RAW_PATH):
        print(f'ERROR: raw data not found at {RAW_PATH}')
        sys.exit(1)

    df = load_data(RAW_PATH)
    df = normalize_asset_type(df)
    df = remove_duplicates(df)
    df = impute_missing(df, IMPUTE_COLS)
    df = winsorize(df, OUTLIER_COLS)
    df = drop_admin_columns(df, ADMIN_COLS)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f'[save]       {OUTPUT_PATH}  |  {df.shape[0]:,} rows × {df.shape[1]} cols')


if __name__ == '__main__':
    run()
