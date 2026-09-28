"""
XGBoost classifier for electricity theft detection.

XGBoost handles class imbalance better than Random Forest via
scale_pos_weight. Trains with early stopping on validation set,
then tunes threshold and reports final test metrics.
"""

import os
import numpy as np
import xgboost as xgb
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)
import joblib

from train_classifier import extract_features, split_supervised


def main():
    # ── Load data ──
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm = d["X"]
    X_raw  = d["X_raw"]
    y      = d["y"]
    print(f"Loaded X_norm={X_norm.shape}, X_raw={X_raw.shape}, y={y.shape}")
    print(f"Class balance: normal={np.sum(y==0)}, theft={np.sum(y==1)}")

    # ── Features ──
    F, feat_names = extract_features(X_norm, X_raw)

    train_ids, val_ids, test_ids = split_supervised(y)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    X_train, y_train = F[train_ids], y[train_ids]
    X_val,   y_val   = F[val_ids],   y[val_ids]
    X_test,  y_test  = F[test_ids],  y[test_ids]

    # ── Compute scale_pos_weight ──
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / pos
    print(f"scale_pos_weight = {spw:.2f} (neg/pos)")

    # ── XGBoost model ──
    print("\nTraining XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",       # PR-AUC — best metric for imbalanced data
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",        # fast
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"\nBest iteration: {model.best_iteration}")
    print(f"Best val PR-AUC: {model.best_score:.4f}")

    # ── Predict probabilities ──
    val_prob  = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    # ── Threshold scan on val ──
    print("\n=== Threshold scan on VAL ===")
    print(f"{'Thresh':>8} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print("-" * 40)
    best_f1, best_thresh = 0, 0.5
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        pred = (val_prob > thresh).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred, zero_division=0)
        marker = ""
        if f > best_f1:
            best_f1, best_thresh = f, thresh
            marker = " ←"
        print(f"{thresh:>8.2f} | {p:>7.4f} | {r:>7.4f} | {f:>7.4f}{marker}")

    print(f"\nBest threshold: {best_thresh}")
    print(f"Best val F1: {best_f1:.4f}")

    # ── Evaluate on test ──
    test_pred = (test_prob > best_thresh).astype(int)

    print(f"\n=== TEST metrics @ threshold = {best_thresh} ===")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, test_prob):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
    print("\nFull report:")
    print(classification_report(y_test, test_pred,
                                target_names=["Normal", "Theft"],
                                zero_division=0))

    # ── Top features ──
    print("\n=== Top 10 Feature Importances ===")
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:10]:
        print(f"  {feat_names[i]:<20} {importances[i]:.4f}")

    # ── Save ──
    os.makedirs("models", exist_ok=True)
    path = "models/xgboost.pkl"
    joblib.dump({
        "model": model,
        "feature_names": feat_names,
        "best_threshold": best_thresh,
    }, path)
    print(f"\n✅ Model saved to {path}")


if __name__ == "__main__":
    main()
