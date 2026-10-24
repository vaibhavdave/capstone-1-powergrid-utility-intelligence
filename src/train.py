"""
Model Training Script
Run: python src/train.py

Trains the best model (LightGBM) with Optuna-tuned hyperparameters.
Saves models/best_model.pkl and models/optimal_threshold.pkl
"""
import pandas as pd
import numpy as np
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

FEATURES_PATH     = 'data/processed/features.csv'
MODEL_PATH        = 'models/best_model.pkl'
THRESHOLD_PATH    = 'models/optimal_threshold.pkl'
RANDOM_STATE      = 42

# Best hyperparameters from Optuna (Story 05) — LightGBM won (AUC-PR=0.8982, Recall=0.8131)
BEST_PARAMS = {
    'n_estimators':      449,
    'max_depth':         6,
    'learning_rate':     0.0267,
    'num_leaves':        63,
    'subsample':         0.861,
    'colsample_bytree':  0.751,
    'min_child_samples': 18,
    'is_unbalance':      True,
    'random_state':      RANDOM_STATE,
    'n_jobs':           -1,
    'verbose':          -1
}


def find_optimal_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1_scores[:-1].argmax()
    return float(thresholds[best_idx])


def run():
    if not os.path.exists(FEATURES_PATH):
        print(f'ERROR: {FEATURES_PATH} not found. Run src/feature_engineering.py first.')
        sys.exit(1)

    df = pd.read_csv(FEATURES_PATH)
    y  = df['grid_failure_flag']
    X  = df.drop(columns=['grid_failure_flag'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')

    X_train_sm, y_train_sm = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    print(f'After SMOTE: {len(X_train_sm):,} training samples')

    model = LGBMClassifier(**BEST_PARAMS)

    print('Training LightGBM...')
    model.fit(X_train_sm, y_train_sm)

    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    print(f'\nTest Set Results (threshold={threshold:.4f}):')
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    print(f'AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}')
    print(f'AUC-PR:  {average_precision_score(y_test, y_prob):.4f}')

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(THRESHOLD_PATH, 'wb') as f:
        pickle.dump(threshold, f)

    print(f'\nSaved: {MODEL_PATH}')
    print(f'Saved: {THRESHOLD_PATH}  (threshold={threshold:.4f})')


if __name__ == '__main__':
    run()
