"""
Train GNN on the KNN graph.

Full-batch training on 42K nodes. Uses pos_weight to handle class
imbalance. Early stopping on validation PR-AUC.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from model_gnn import TheftGNN


# ─────────────────────────────────────────────
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS      = 200
LR          = 1e-3
WEIGHT_DECAY = 5e-5
HIDDEN_DIM  = 128
NUM_LAYERS  = 3
DROPOUT     = 0.3
PATIENCE    = 25               # early stopping
SEED        = 42
MODEL_PATH  = "models/gnn.pt"
os.makedirs("models", exist_ok=True)


def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data ──
    d = np.load("data/processed/gnn_data.npz", allow_pickle=True)
    X = torch.from_numpy(d["node_features"]).float()
    edge_index = torch.from_numpy(d["edge_index"]).long()
    y = torch.from_numpy(d["y"]).long()

    splits = np.load("data/processed/gnn_splits.npz")
    train_ids = torch.from_numpy(splits["train_ids"]).long()
    val_ids   = torch.from_numpy(splits["val_ids"]).long()
    test_ids  = torch.from_numpy(splits["test_ids"]).long()

    n, in_dim = X.shape
    print(f"Graph: {n:,} nodes, {edge_index.shape[1]:,} edges, {in_dim} features")
    print(f"Train: {len(train_ids):,}, Val: {len(val_ids):,}, Test: {len(test_ids):,}")

    # Build masks
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask   = torch.zeros(n, dtype=torch.bool)
    test_mask  = torch.zeros(n, dtype=torch.bool)
    train_mask[train_ids] = True
    val_mask[val_ids]     = True
    test_mask[test_ids]   = True

    # Move to device
    X = X.to(DEVICE)
    edge_index = edge_index.to(DEVICE)
    y = y.to(DEVICE)
    train_mask = train_mask.to(DEVICE)
    val_mask = val_mask.to(DEVICE)
    test_mask = test_mask.to(DEVICE)

    # ── Model ──
    model = TheftGNN(in_dim=in_dim, hidden_dim=HIDDEN_DIM,
                     num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ── Loss with class balancing ──
    y_train = y[train_mask]
    pos = (y_train == 1).sum().item()
    neg = (y_train == 0).sum().item()
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=DEVICE)
    print(f"pos_weight = {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8)

    # ── Training loop ──
    best_val_auc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        optimizer.zero_grad()
        logits = model(X, edge_index)
        loss = criterion(logits[train_mask], y_train.float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            logits = model(X, edge_index)
            val_probs = torch.sigmoid(logits[val_mask]).cpu().numpy()
        val_labels = y[val_mask].cpu().numpy()
        val_auc = average_precision_score(val_labels, val_probs)

        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | "
                  f"val_PR-AUC={val_auc:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "in_dim": in_dim,
                    "hidden_dim": HIDDEN_DIM,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                },
                "best_val_auc": best_val_auc,
            }, MODEL_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE})")
                break

    print(f"\n✅ Done. Best val PR-AUC: {best_val_auc:.4f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
