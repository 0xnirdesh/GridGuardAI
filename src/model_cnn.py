"""
1D-CNN for electricity theft detection.

Treats each customer's time-series as a 2-channel signal:
  - Channel 0: normalized consumption
  - Channel 1: raw consumption
"""

import torch
import torch.nn as nn


class TheftCNN(nn.Module):
    def __init__(self, in_channels=2, seq_len=1033):
        super().__init__()
        self.seq_len = seq_len

        # ── Block 1 ──
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # ── Block 2 ──
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # ── Block 3 ──
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)

        # ── Block 4 ──
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(128)

        # ── Global pooling + classifier ──
        self.gap = nn.AdaptiveAvgPool1d(1)     # (B, 128, 1)
        self.gmp = nn.AdaptiveMaxPool1d(1)     # (B, 128, 1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, 1)            # 128*2 channels

    def forward(self, x):
        # x: (B, 2, T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = torch.relu(self.bn4(self.conv4(x)))

        # Combine avg + max pooling (captures both "typical" and "extreme" patterns)
        avg = self.gap(x).squeeze(-1)
        mx  = self.gmp(x).squeeze(-1)
        x = torch.cat([avg, mx], dim=1)

        x = self.dropout(x)
        return self.fc(x).squeeze(-1)
