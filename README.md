# ⚡ GridGuardAI

**AI-powered electricity theft detection system** — detecting Non-Technical Losses (NTL) in smart grids using multi-layer deep learning.

![Python](https://img.shields.io/badge/python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Problem

Non-Technical Losses (NTL) — electricity theft, meter tampering, and billing fraud — cost DISCOMs billions of rupees annually. Manual inspection of millions of meters is impossible.

**GridGuardAI** provides an AI solution that:
- Detects suspicious consumption patterns automatically
- Reasons over grid topology to find organized theft
- Preserves consumer privacy via federated learning
- Offers a web dashboard for utility operators

---

## 🏗️ Architecture

Three-layer AI system:

| Layer | Technique | Purpose |
|-------|-----------|---------|
| **1** | LSTM Autoencoder + XGBoost | Per-customer anomaly detection |
| **2** | Graph Neural Network | Grid-topology reasoning |
| **3** | Federated Learning | Privacy-preserving training |

Plus (planned):
- FastAPI backend — REST API for predictions
- Web dashboard — CSV upload, results visualization

---

## 📊 Dataset

**SGCC** (State Grid Corporation of China)
- 42,372 customers × 1,034 days (2014–2016)
- 3,615 theft cases (~8.5%)
- Source: [GitHub](https://github.com/henryRDlab/ElectricityTheftDetection)

---

## 📈 Results

Trained and compared multiple models on the SGCC dataset.

### Layer 1 — Classical ML

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|-----|--------|---------|
| LSTM Autoencoder | 0.23 | 0.82 | 0.37 | 0.22 | 0.45 |
| Random Forest | 0.64 | 0.13 | 0.21 | 0.32 | 0.75 |
| SMOTE + XGBoost | 0.39 | 0.40 | 0.40 | 0.38 | 0.78 |
| 1D-CNN | — | — | 0.36 | 0.36 | — |
| **XGBoost (best)** | **0.45** | **0.42** | **0.43** | **0.44** | **0.81** |

### Layer 2 — Graph Neural Network

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|-------|-----------|--------|-----|--------|---------|
| GraphSAGE GNN | 0.29 | 0.57 | 0.39 | 0.39 | 0.80 |

**Best model:** XGBoost with 100 engineered features.

**Key insight:** Raw magnitude features (mean, std, trend slope) + monthly consumption patterns are the strongest predictors of theft. Feature engineering gave bigger gains than model architecture changes.

---

## 🚀 Quick Start


# Clone
git clone https://github.com/0xnirdesh/GridGuardAI.git
cd GridGuardAI

# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download SGCC dataset → place at data/raw/sgcc.csv
# Then run:
python src/preprocess.py
python src/train_xgboost.py

---
## 📁 Project Structure 


GridGuardAI/
├── data/
│ ├── raw/ # Original SGCC CSV (not in repo)
│ └── processed/ # Cleaned .npz files
├── models/ # Trained model checkpoints
├── src/
│ ├── # Data Pipeline
│ ├── inspect_data.py # Explore raw dataset
│ ├── preprocess.py # Cleaning, normalization, X_raw
│ ├── check_processed.py # Verify processed data
│ ├── dataset.py # PyTorch Dataset (lazy windows)
│ │
│ ├── # Layer 1a: LSTM Autoencoder
│ ├── model_autoencoder.py # LSTM autoencoder architecture
│ ├── train_autoencoder.py # Training loop
│ ├── evaluate_autoencoder.py # Metrics + threshold tuning
│ │
│ ├── # Layer 1b: Classical ML
│ ├── train_classifier.py # Feature engineering (100 features)
│ ├── tune_threshold.py # RF threshold optimization
│ ├── train_xgboost.py # XGBoost (best Layer 1 model)
│ ├── train_smote_xgb.py # SMOTE + XGBoost variant
│ ├── model_cnn.py # 1D-CNN architecture
│ ├── train_cnn.py # CNN training
│ ├── evaluate_cnn.py # CNN evaluation
│ │
│ ├── # Layer 2: Graph Neural Network
│ ├── build_graph.py # Synthetic KNN graph builder
│ ├── model_gnn.py # GraphSAGE architecture
│ ├── train_gnn.py # GNN training
│ ├── evaluate_gnn.py # GNN evaluation
│ └── ensemble.py # XGBoost + GNN comparison
│
└── web/ # FastAPI + dashboard (planned)

text
---

##📈 Roadmap

- [x] Data pipeline (preprocessing + feature engineering)
- [x] Layer 1: LSTM Autoencoder baseline
- [x] Layer 1: XGBoost with 100 engineered features (F1: 0.43)
- [x] Layer 1: 1D-CNN experiment
- [x] Layer 2: Graph Neural Network (GraphSAGE)
- [ ] Layer 3: Federated Learning with differential privacy
- [ ] Web backend (FastAPI)
- [ ] Frontend dashboard with CSV upload
- [ ] Report and final documentation

---

## 👤 Author

**Nirdesh Rajput** — 3rd Year AI Lab Project

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
