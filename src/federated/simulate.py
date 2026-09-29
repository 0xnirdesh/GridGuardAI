"""
Federated Learning simulation.

Splits data into 5 Non-IID clients (simulating 5 DISCOMs), then runs
FedAvg for multiple rounds. Compares against centralized baseline.
"""

import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)

from .model import TheftMLP
from .client import FederatedClient
from .server import federated_averaging

# Import feature extractor and splitter
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from train_classifier import extract_features, split_supervised


# ─────────────────────────────────────────────
DEVICE        = "mps" if torch.backends.mps.is_available() else "cpu"
NUM_CLIENTS   = 5
NUM_ROUNDS    = 20
LOCAL_EPOCHS  = 3
BATCH_SIZE    = 64
LR            = 1e-3
SEED          = 42
os.makedirs("models", exist_ok=True)


def split_non_iid(y, n_clients, seed=42):
    """
    Split customers into N clients with a NON-IID distribution:
    each client gets a different theft ratio.
    """
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    theft_idx  = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    rng.shuffle(theft_idx)

    # Assign theft ratios to clients (from low to high)
    theft_ratios = np.linspace(0.03, 0.15, n_clients)
    total_theft = len(theft_idx)
    total_normal = len(normal_idx)

    client_indices = []
    theft_ptr = 0
    normal_ptr = 0

    for i in range(n_clients):
        ratio = theft_ratios[i]
        n_theft = int(total_theft * ratio / theft_ratios.sum())
        n_normal = int(total_normal / n_clients)

        theft_slice = theft_idx[theft_ptr:theft_ptr + n_theft]
        normal_slice = normal_idx[normal_ptr:normal_ptr + n_normal]
        theft_ptr += n_theft
        normal_ptr += n_normal

        client_ids = np.concatenate([theft_slice, normal_slice])
        client_indices.append(client_ids)

    # Any leftover goes to last client
    if theft_ptr < total_theft:
        client_indices[-1] = np.concatenate(
            [client_indices[-1], theft_idx[theft_ptr:]]
        )
    if normal_ptr < total_normal:
        client_indices[-1] = np.concatenate(
            [client_indices[-1], normal_idx[normal_ptr:]]
        )

    return client_indices


def evaluate_model(model, X, y, device):
    """Return (probs, labels)."""
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X).float().to(device)
        logits = model(X_t)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs, y


def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load features ──
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm, X_raw, y = d["X"], d["X_raw"], d["y"]
    print(f"Loaded X={X_norm.shape}, y={y.shape}")

    F, _ = extract_features(X_norm, X_raw)

    # Standardize features globally
    mu = F.mean(axis=0, keepdims=True)
    sigma = F.std(axis=0, keepdims=True) + 1e-6
    F_std = ((F - mu) / sigma).astype(np.float32)

    # ── Split into train/val/test (same as before) ──
    train_ids, val_ids, test_ids = split_supervised(y)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # ── Split train into N Non-IID clients ──
    client_splits = split_non_iid(y[train_ids], NUM_CLIENTS, seed=SEED)
    print(f"\n=== Client distribution ({NUM_CLIENTS} DISCOMs) ===")
    for i, ids in enumerate(client_splits):
        abs_ids = train_ids[ids]
        n_theft = (y[abs_ids] == 1).sum()
        n_normal = (y[abs_ids] == 0).sum()
        ratio = n_theft / len(abs_ids) * 100
        print(f"  Client {i+1}: {len(abs_ids):>6,} samples "
              f"(theft: {n_theft:>4}, {ratio:.2f}%)")

    # ── Create client objects ──
    clients = []
    for i, ids in enumerate(client_splits):
        abs_ids = train_ids[ids]
        clients.append(FederatedClient(i, F_std[abs_ids], y[abs_ids], device=DEVICE))

    # ── Initialize global model ──
    global_model = TheftMLP(in_dim=F_std.shape[1]).to(DEVICE)
    global_state = global_model.state_dict()
    print(f"\nGlobal model params: {sum(p.numel() for p in global_model.parameters()):,}")

    # Validation and test tensors
    X_val, y_val   = F_std[val_ids],  y[val_ids]
    X_test, y_test = F_std[test_ids], y[test_ids]

    # ── Federated training loop ──
    print(f"\n=== Federated Training ({NUM_ROUNDS} rounds) ===")
    best_val_auc = 0.0
    history = []

    for rnd in range(1, NUM_ROUNDS + 1):
        t0 = time.time()
        round_losses = []
        client_states = []
        client_sizes = []

        # Each client trains locally
        for client in clients:
            state, size, loss = client.train(
                global_state, epochs=LOCAL_EPOCHS,
                lr=LR, batch_size=BATCH_SIZE
            )
            client_states.append(state)
            client_sizes.append(size)
            round_losses.append(loss)

        # Server aggregates
        global_state = federated_averaging(client_states, client_sizes)

        # Update global model
        global_model.load_state_dict(global_state)

        # Evaluate on validation
        val_probs, _ = evaluate_model(global_model, X_val, y_val, DEVICE)
        val_auc = average_precision_score(y_val, val_probs)

        history.append({
            "round": rnd,
            "avg_loss": np.mean(round_losses),
            "val_pr_auc": val_auc,
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "model_state": global_state,
                "round": rnd,
                "best_val_auc": best_val_auc,
            }, "models/federated_model.pt")

        dt = time.time() - t0
        print(f"Round {rnd:02d}/{NUM_ROUNDS} | "
              f"avg_loss={np.mean(round_losses):.4f} | "
              f"val_PR-AUC={val_auc:.4f} | "
              f"{dt:.1f}s"
              + ("  ✔ saved" if val_auc >= best_val_auc else ""))

    print(f"\n✅ Federated training done. Best val PR-AUC: {best_val_auc:.4f}")

    # ── Final evaluation on TEST ──
    ckpt = torch.load("models/federated_model.pt", map_location="cpu",
                      weights_only=False)
    global_model.load_state_dict(ckpt["model_state"])

    test_probs, _ = evaluate_model(global_model, X_test, y_test, DEVICE)

    # Threshold scan on val
    val_probs, _ = evaluate_model(global_model, X_val, y_val, DEVICE)
    best_f1, best_thresh = 0, 0.5
    for thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        pred = (val_probs > thresh).astype(int)
        f = f1_score(y_val, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_thresh = f, thresh

    test_pred = (test_probs > best_thresh).astype(int)

    print(f"\n=== FEDERATED TEST metrics @ threshold = {best_thresh} ===")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, test_probs):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, test_probs):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
    print(f"\n{classification_report(y_test, test_pred, target_names=['Normal','Theft'], zero_division=0)}")

    # ── Compare with centralized XGBoost ──
    print("=== Comparison ===")
    print(f"{'Model':<25} | {'F1':>7} | {'PR-AUC':>7}")
    print("-" * 50)
    print(f"{'XGBoost (centralized)':<25} | {0.4337:>7.4f} | {0.4369:>7.4f}  ← reference")
    print(f"{'Federated MLP (ours)':<25} | "
          f"{f1_score(y_test, test_pred, zero_division=0):>7.4f} | "
          f"{average_precision_score(y_test, test_probs):>7.4f}")
    print(f"\nPrivacy: Federated model trained WITHOUT sharing raw customer data.")


if __name__ == "__main__":
    main()
