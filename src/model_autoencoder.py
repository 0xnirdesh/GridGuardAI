"""
LSTM Autoencoder for time-series anomaly detection.

Encoder: LSTM that compresses (T, 1) -> latent vector (hidden_dim,)
Decoder: LSTM that reconstructs (T, 1) from the latent vector
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, latent_dim=32, num_layers=2):
        super().__init__()
        self.window_size = None   # set on first forward (or explicit)

        # ── Encoder ──
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)

        # ── Decoder ──
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output_fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (B, T, 1)
        B, T, _ = x.shape

        # Encode: take last hidden state
        _, (h, _) = self.encoder(x)          # h: (num_layers, B, hidden)
        h_last = h[-1]                       # (B, hidden)
        z = self.enc_fc(h_last)              # (B, latent)

        # Decode: repeat latent across time, then LSTM
        d = self.dec_fc(z)                   # (B, hidden)
        d = d.unsqueeze(1).repeat(1, T, 1)   # (B, T, hidden)
        out, _ = self.decoder(d)             # (B, T, hidden)
        recon = self.output_fc(out)          # (B, T, 1)
        return recon
