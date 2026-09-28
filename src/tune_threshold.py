"""
Find the optimal probability threshold for Random Forest.
Imports extract_features from train_classifier to avoid duplication.
"""

import numpy as np
import joblib
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score)

from train_classifier import extract_features, split_supervised


def main():
    # Load data
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm = d["X"]
    X_raw  = d["X_raw"]
    y      = d["y"]

    F, _ = extract_features(X_norm, X_raw)
    train_ids, val_ids, test_ids = split_supervised(y)

    X_val,  y_val  = F[val_ids],  y[val_ids]
    X_test, y_test = F[test_ids], y[test_ids]

    # Load trained RF
    bundle = joblib.load("models/random_forest.pkl")
    rf = bundle["model"]

    # Predict probabilities
    val_prob  = rf.predict_proba(X_val)[:, 1]
    test_prob = rf.predict_proba(X_test)[:, 1]

    print(f"Val theft ratio: {y_val.mean():.4f}")
    print(f"Test theft ratio: {y_test.mean():.4f}")
    print(f"\nVal probability distribution:")
    print(f"  Normal: mean={val_prob[y_val==0].mean():.4f}, "
          f"max={val_prob[y_val==0].max():.4f}")
    print(f"  Theft : mean={val_prob[y_val==1].mean():.4f}, "
          f"max={val_prob[y_val==1].max():.4f}")

    # Scan thresholds
    print("\n=== Threshold scan on VAL ===")
    print(f"{'Thresh':>8} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print("-" * 40)
    best_f1, best_thresh = 0, 0.5
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pred = (val_prob > thresh).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred, zero_division=0)
        marker = ""
        if f > best_f1:
            best_f1, best_thresh = f, thresh
            marker = " ←"
        print(f"{thresh:>8.2f} | {p:>7.4f} | {r:>7.4f} | {f:>7.4f}{marker}")

    print(f"\nBest threshold (val): {best_thresh}")
    print(f"Best val F1: {best_f1:.4f}")

    # Evaluate on test with best threshold
    test_pred = (test_prob > best_thresh).astype(int)

    print(f"\n=== TEST metrics @ threshold = {best_thresh} ===")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, test_prob):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")

    # Compare with default 0.5
    default_pred = (test_prob > 0.5).astype(int)
    print(f"\n=== TEST metrics @ threshold = 0.50 (default) ===")
    print(f"Precision: {precision_score(y_test, default_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, default_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, default_pred, zero_division=0):.4f}")


if __name__ == "__main__":
    main()