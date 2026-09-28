
GridGuardAI/
├── data/
│ ├── raw/ # original sgcc.csv (not in repo)
│ └── processed/ # cleaned .npz (regenerated)
├── models/ # trained checkpoints
└── src/
├── inspect_data.py # dataset exploration
├── preprocess.py # cleaning + normalization
├── dataset.py # PyTorch Dataset (lazy windows)
├── model_autoencoder.py # Layer 1 model
├── train_autoencoder.py # training loop
└── evaluate_autoencoder.py # metrics

tex# GridGuardAI

AI-based electricity theft detection using multi-layer deep learning.

## Problem

Non-Technical Losses (NTL) — electricity theft, meter tampering, and billing fraud — cost DISCOMs billions of rupees annually. Manual inspection of millions of meters is impossible.

## Approach

Three-layer AI architecture:

| Layer | Technique | Purpose |
|-------|-----------|---------|
| 1 | LSTM Autoencoder | Per-customer anomaly detection |
| 2 | Graph Neural Network | Grid-topology-aware reasoning |
| 3 | Federated Learning | Privacy-preserving collaborative training |

## Dataset

**SGCC** (State Grid Corporation of China)
- 42,372 customers × 1,034 days (2014–2016)
- 3,618 theft cases (~8.5%)
- Source: [GitHub](https://github.com/henryRDlab/ElectricityTheftDetection)

## Project Structure

