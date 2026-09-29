"""
Federated client — simulates one DISCOM.

Trains locally on own data, returns updated weights to server.
Never sends raw data.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .model import TheftMLP


class FederatedClient:
    def __init__(self, client_id, X, y, device="cpu"):
        self.client_id = client_id
        self.device = device
        self.X = torch.from_numpy(X).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        self.n_samples = len(y)

        # Compute class weight locally
        pos = (y == 1).sum()
        neg = (y == 0).sum()
        self.pos_weight = torch.tensor([neg / max(pos, 1)],
                                        dtype=torch.float32, device=device)

    def train(self, global_state_dict, epochs=3, lr=1e-3, batch_size=64):
        """
        Train locally starting from global weights.
        Returns: updated state_dict, num_samples, train_loss
        """
        model = TheftMLP(in_dim=self.X.shape[1]).to(self.device)
        model.load_state_dict(global_state_dict)
        model.train()

        criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        loader = DataLoader(
            TensorDataset(self.X, self.y),
            batch_size=batch_size, shuffle=True
        )

        total_loss = 0.0
        for _ in range(epochs):
            for xb, yb in loader:
                logits = model(xb)
                loss = criterion(logits, yb)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * xb.size(0)

        return (
            {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            self.n_samples,
            total_loss / (self.n_samples * epochs),
        )
