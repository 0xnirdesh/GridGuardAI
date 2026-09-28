"""
Train the LSTM autoencoder on NORMAL customers only.
Saves the best model to models/autoencoder.pt
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SGCCWindowDataset, split_customers
from model_autoencoder import LSTMAutoencoder


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEVICE       = "mps" if torch.backends.mps.is_available() else \
               "cuda" if torch.cuda.is_available() else "cpu"
WINDOW       = 30
STRIDE       = 30
BATCH_SIZE   = 128
EPOCHS       = 30
LR           = 1e-3
LATENT_DIM   = 32
HIDDEN_DIM   = 64
NUM_LAYERS   = 2
SEED         = 42

MODEL_PATH   = "models/autoencoder.pt"
os.makedirs("models", exist_ok=True)


def main():
    print(f"Device: {DEVICE}")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    print(f"Loaded X={X.shape}, y={y.shape}")

    # Split
    train_ids, val_ids, test_ids = split_customers(y)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # Datasets — training uses NORMAL ONLY
    train_ds = SGCCWindowDataset(X, y, train_ids,
                                 window_size=WINDOW, stride=STRIDE,
                                 normal_only=True)
    val_ds   = SGCCWindowDataset(X, y, val_ids,
                                 window_size=WINDOW, stride=STRIDE)
    print(f"Train windows: {len(train_ds):,} | Val windows: {len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    # Model
    model = LSTMAutoencoder(hidden_dim=HIDDEN_DIM,
                            latent_dim=LATENT_DIM,
                            num_layers=NUM_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.MSELoss()

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        t0, train_loss = time.time(), 0.0
        for xb, _, _ in train_loader:
            xb = xb.to(DEVICE)
            recon = model(xb)
            loss  = criterion(recon, xb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # ── Validate (on NORMAL val windows only) ──
        model.eval()
        val_loss, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                mask = (yb == 0)
                if mask.sum() == 0:
                    continue
                xb_n = xb[mask].to(DEVICE)
                recon = model(xb_n)
                loss  = criterion(recon, xb_n)
                val_loss += loss.item() * xb_n.size(0)
                n_val += xb_n.size(0)
        val_loss = val_loss / max(n_val, 1)

        scheduler.step(val_loss)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train_loss={train_loss:.6f} | "
              f"val_loss={val_loss:.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{dt:.1f}s")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "hidden_dim": HIDDEN_DIM,
                    "latent_dim": LATENT_DIM,
                    "num_layers": NUM_LAYERS,
                    "window_size": WINDOW,
                },
                "best_val_loss": best_val,
            }, MODEL_PATH)
            print(f"   ✔ saved (best val={best_val:.6f})")

    print(f"\n✅ Done. Best val loss: {best_val:.6f}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
