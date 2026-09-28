"""
PyTorch Dataset for SGCC electricity theft detection.

Key ideas:
  - Windows are generated lazily (index math), not stored.
  - Split is done at the CUSTOMER level to prevent leakage.
  - Training set contains ONLY normal customers (for the autoencoder).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


# ─────────────────────────────────────────────
# Customer-level split
# ─────────────────────────────────────────────
def split_customers(y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Return index arrays for train/val/test customers.

    - Train: only NORMAL customers (autoencoder sees only normal data).
    - Val:   mix of normal + theft (for threshold tuning).
    - Test:  mix of normal + theft (final evaluation).
    """
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    theft_idx  = np.where(y == 1)[0]

    rng.shuffle(normal_idx)
    rng.shuffle(theft_idx)

    # Normal customers -> train / val / test
    n = len(normal_idx)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    train_n = normal_idx[:n_train]
    val_n   = normal_idx[n_train:n_train + n_val]
    test_n  = normal_idx[n_train + n_val:]

    # Theft customers -> split between val / test only
    t = len(theft_idx)
    n_val_t = int(t * 0.5)
    val_t   = theft_idx[:n_val_t]
    test_t  = theft_idx[n_val_t:]

    train_idx = train_n
    val_idx   = np.concatenate([val_n, val_t])
    test_idx  = np.concatenate([test_n, test_t])

    return train_idx, val_idx, test_idx


# ─────────────────────────────────────────────
# Lazy-window Dataset
# ─────────────────────────────────────────────
class SGCCWindowDataset(Dataset):
    """
    Produces windows of shape (window_size, 1) from the (N, T) matrix.

    Parameters
    ----------
    X           : np.ndarray, shape (N, T)
    y           : np.ndarray, shape (N,)
    customer_ids: index array into X/y (which customers this dataset covers)
    window_size : int
    stride      : int
    normal_only : bool  -> if True, drop theft customers entirely (for training)
    """

    def __init__(self, X, y, customer_ids, window_size=30, stride=30,
                 normal_only=False):
        super().__init__()
        self.window_size = window_size
        self.stride      = stride

        ids = np.asarray(customer_ids)
        if normal_only:
            ids = ids[y[ids] == 0]

        self.X = X[ids].astype(np.float32)     # (M, T)
        self.y = y[ids].astype(np.int64)       # (M,)
        self.customer_ids = ids                # original indices

        # Precompute (customer, start) pairs — cheap, integer only
        T = self.X.shape[1]
        n_windows_per_customer = (T - window_size) // stride + 1
        self.n_per_customer = n_windows_per_customer
        self.total = len(ids) * n_windows_per_customer

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        cust_i = idx // self.n_per_customer
        win_i  = idx %  self.n_per_customer
        start  = win_i * self.stride
        end    = start + self.window_size

        window = self.X[cust_i, start:end]           # (window_size,)
        window = torch.from_numpy(window).unsqueeze(-1)  # (window_size, 1)
        label  = int(self.y[cust_i])
        return window, label, int(self.customer_ids[cust_i])
