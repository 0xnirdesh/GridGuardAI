"""
Evaluate the trained LSTM autoencoder for electricity theft detection.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix)

from dataset import SGCCWindowDataset, split_customers
from model_autoencoder import LSTMAutoencoder


DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"


def per_window_errors(model, loader, device):
    criterion = nn.MSELoss(reduction="none")
    errs, labs, cids = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, cb in loader:
            xb = xb.to(device)
            recon = model(xb)
            e = criterion(recon, xb).mean(dim=(1, 2)).cpu().numpy()
            errs.append(e)
            labs.append(yb.numpy())
            cids.append(cb.numpy())
    return (np.concatenate(errs),
            np.concatenate(labs),
            np.concatenate(cids))


def main():
    print(f"Device: {DEVICE}")

    ckpt = torch.load("models/autoencoder.pt", map_location="cpu")
    cfg = ckpt["config"]
    model = LSTMAutoencoder(hidden_dim=cfg["hidden_dim"],
                            latent_dim=cfg["latent_dim"],
                            num_layers=cfg["num_layers"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded model (val loss {ckpt['best_val_loss']:.6f})")

    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    _, val_ids, test_ids = split_customers(y)

    W = cfg["window_size"]
    val_ds  = SGCCWindowDataset(X, y, val_ids,  window_size=W, stride=W)
    test_ds = SGCCWindowDataset(X, y, test_ids, window_size=W, stride=W)
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    val_errs, val_labs, _ = per_window_errors(model, val_loader, DEVICE)
    normal_val_errs = val_errs[val_labs == 0]
    threshold = np.percentile(normal_val_errs, 95)
    print(f"Normal val errs: mean={normal_val_errs.mean():.6f}, "
          f"p95={threshold:.6f}")

    test_errs, test_labs, test_cids = per_window_errors(model, test_loader, DEVICE)
    preds = (test_errs > threshold).astype(int)

    print("\n=== Window-level metrics (test) ===")
    print(f"Accuracy : {np.mean(preds == test_labs):.4f}")
    print(f"Precision: {precision_score(test_labs, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(test_labs, preds, zero_division=0):.4f}")
    print(f"F1       : {f1_score(test_labs, preds, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(test_labs, test_errs):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(test_labs, test_errs):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(test_labs, preds)}")

    cust_flag, cust_true = {}, {}
    for cid, p, l in zip(test_cids, preds, test_labs):
        cust_flag[cid] = max(cust_flag.get(cid, 0), p)
        cust_true[cid] = l
    cust_ids   = np.array(list(cust_flag.keys()))
    cust_preds = np.array([cust_flag[c] for c in cust_ids])
    cust_labs  = np.array([cust_true[c] for c in cust_ids])

    print("\n=== Customer-level metrics (test) ===")
    print(f"Precision: {precision_score(cust_labs, cust_preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(cust_labs, cust_preds, zero_division=0):.4f}")
    print(f"F1       : {f1_score(cust_labs, cust_preds, zero_division=0):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(cust_labs, cust_preds)}")


if __name__ == "__main__":
    main()
