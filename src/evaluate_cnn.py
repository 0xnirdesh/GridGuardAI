"""
Evaluate the trained 1D-CNN with threshold tuning.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)

from model_cnn import TheftCNN
from train_cnn import SGCCDataset, DEVICE, BATCH_SIZE
from train_classifier import split_supervised


def main():
    print(f"Device: {DEVICE}")

    ckpt = torch.load("models/cnn.pt", map_location="cpu")
    model = TheftCNN().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded (best val PR-AUC: {ckpt['best_val_auc']:.4f})")

    raw_mean = ckpt["raw_mean"]
    raw_std  = ckpt["raw_std"]

    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm, X_raw, y = d["X"], d["X_raw"], d["y"]
    X_raw_n = (X_raw - raw_mean) / (raw_std + 1e-6)

    _, val_ids, test_ids = split_supervised(y)
    val_ds  = SGCCDataset(X_norm, X_raw_n, y, val_ids)
    test_ds = SGCCDataset(X_norm, X_raw_n, y, test_ids)
    val_loader  = DataLoader(val_ds,  batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    def get_probs(loader):
        model.eval()
        probs, labs = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                labs.append(yb.numpy())
        return np.concatenate(probs), np.concatenate(labs)

    val_prob,  y_val  = get_probs(val_loader)
    test_prob, y_test = get_probs(test_loader)

    # Threshold scan
    print("\n=== Threshold scan on VAL ===")
    print(f"{'Thresh':>8} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print("-" * 40)
    best_f1, best_thresh = 0, 0.5
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
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


if __name__ == "__main__":
    main()

