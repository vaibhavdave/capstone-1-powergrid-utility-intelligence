"""
Model Evaluation Script
Run: python src/evaluate.py

Loads best model and prints evaluation metrics on the held-out test set.
"""
import pandas as pd
import pickle
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    average_precision_score, confusion_matrix
)

FEATURES_PATH  = 'data/processed/features.csv'
MODEL_PATH     = 'models/best_model.pkl'
THRESHOLD_PATH = 'models/optimal_threshold.pkl'
RANDOM_STATE   = 42


def run():
    for path in [FEATURES_PATH, MODEL_PATH, THRESHOLD_PATH]:
        if not os.path.exists(path):
            print(f'ERROR: {path} not found.')
            sys.exit(1)

    df = pd.read_csv(FEATURES_PATH)
    y  = df['grid_failure_flag']
    X  = df.drop(columns=['grid_failure_flag'])

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(THRESHOLD_PATH, 'rb') as f:
        threshold = pickle.load(f)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f'=== Evaluation Report ===')
    print(f'Model:      {type(model).__name__}')
    print(f'Threshold:  {threshold:.4f}')
    print()
    print(classification_report(y_test, y_pred, target_names=['No Failure', 'Failure']))
    print(f'AUC-ROC:    {roc_auc_score(y_test, y_prob):.4f}')
    print(f'AUC-PR:     {average_precision_score(y_test, y_prob):.4f}')
    print()
    print(f'Confusion Matrix:')
    print(f'  True Positives  (caught failures):    {tp:,}')
    print(f'  False Positives (unnecessary checks): {fp:,}')
    print(f'  False Negatives (missed failures):    {fn:,}')
    print(f'  True Negatives  (correct no-failure): {tn:,}')


if __name__ == '__main__':
    run()
