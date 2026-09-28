"""
Train 1D-CNN on raw sequences (no feature engineering).
Handles class imbalance via pos_weight in BCE loss.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)

from model_cnn import TheftCNN
from train_classifier import split_supervised


# ─────────────────────────────────────────────
DEVICE      = "mps" if torch.backends.mps.is_available() else \
              "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 64
EPOCHS      = 40
LR          = 1e-3
SEED        = 42
MODEL_PATH  = "models/cnn.pt"
os.makedirs("models", exist_ok=True)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class SGCCDataset(Dataset):
    """2-channel input: [normalized, raw]"""
    def __init__(self, X_norm, X_raw, y, ids):
        self.X = np.stack([X_norm[ids], X_raw[ids]], axis=1).astype(np.float32)
        # shape: (n, 2, T)
        self.y = y[ids].astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])


# ─────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm, X_raw, y = d["X"], d["X_raw"], d["y"]
    print(f"Loaded X_norm={X_norm.shape}, X_raw={X_raw.shape}")

    train_ids, val_ids, test_ids = split_supervised(y)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    print(f"Train theft: {np.sum(y[train_ids]==1)}, "
          f"Val theft: {np.sum(y[val_ids]==1)}, "
          f"Test theft: {np.sum(y[test_ids]==1)}")

    # Standardize raw channel (mean 0, std 1) — helps CNN
    raw_mean = X_raw[train_ids].mean()
    raw_std  = X_raw[train_ids].std()
    X_raw_n  = (X_raw - raw_mean) / (raw_std + 1e-6)
    print(f"Raw normalization: mean={raw_mean:.3f}, std={raw_std:.3f}")

    train_ds = SGCCDataset(X_norm, X_raw_n, y, train_ids)
    val_ds   = SGCCDataset(X_norm, X_raw_n, y, val_ids)
    test_ds  = SGCCDataset(X_norm, X_raw_n, y, test_ids)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = TheftCNN(in_channels=2, seq_len=X_norm.shape[1]).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Class imbalance via BCE pos_weight
    pos = (y[train_ids] == 1).sum()
    neg = (y[train_ids] == 0).sum()
    pos_weight = torch.tensor([float(neg) / float(pos)], dtype=torch.float32, device=DEVICE)
    print(f"pos_weight = {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    best_val_auc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        t0, train_loss = time.time(), 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ── Validate ──
        model.eval()
        val_loss, val_probs, val_labels = 0.0, [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * xb.size(0)
                val_probs.append(torch.sigmoid(logits).cpu().numpy())
                val_labels.append(yb.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_probs  = np.concatenate(val_probs)
        val_labels = np.concatenate(val_labels)
        val_auc = average_precision_score(val_labels, val_probs)

        scheduler.step(val_loss)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"val_PR-AUC={val_auc:.4f} | {dt:.1f}s")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state": model.state_dict(),
                "raw_mean": float(raw_mean),
                "raw_std":  float(raw_std),
                "best_val_auc": best_val_auc,
            }, MODEL_PATH)
            print(f"   ✔ saved (best val PR-AUC={best_val_auc:.4f})")

    print(f"\n✅ Done. Best val PR-AUC: {best_val_auc:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
