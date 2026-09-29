"""
Ensemble XGBoost + GNN predictions.

Since both models have different strengths (XGBoost: precision,
GNN: recall), averaging their probabilities often beats both.
"""
import numpy as np
import torch
import joblib
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)

from model_gnn import TheftGNN
from train_classifier import extract_features, split_supervised

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")

    # ── Load splits ──
    splits = np.load("data/processed/gnn_splits.npz")
    val_ids  = splits["val_ids"]
    test_ids = splits["test_ids"]

    # ── GNN predictions ──
    d_gnn = np.load("data/processed/gnn_data.npz", allow_pickle=True)
    X_gnn = torch.from_numpy(d_gnn["node_features"]).float().to(DEVICE)
    edge_index = torch.from_numpy(d_gnn["edge_index"]).long().to(DEVICE)
    y_all = d_gnn["y"]

    ckpt = torch.load("models/gnn.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    gnn = TheftGNN(in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                   num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(DEVICE)
    gnn.load_state_dict(ckpt["model_state"])
    gnn.eval()
    with torch.no_grad():
        gnn_probs_all = torch.sigmoid(gnn(X_gnn, edge_index)).cpu().numpy()

    # ── XGBoost predictions ──
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm, X_raw = d["X"], d["X_raw"]
    F, _ = extract_features(X_norm, X_raw)
    # Standardize using GNN saved stats for consistency? No — use raw features as trained.
    # But we need same standardization as XGBoost trained on. Let's just use F directly
    # since train_xgboost.py uses raw F.

    bundle = joblib.load("models/xgboost.pkl")
    xgb_model = bundle["model"]
    xgb_probs_all = xgb_model.predict_proba(F)[:, 1]

    # ── Evaluate on val to tune ensemble weight ──
    y_val = y_all[val_ids]
    gnn_val  = gnn_probs_all[val_ids]
    xgb_val  = xgb_probs_all[val_ids]

    print("\n=== Val metrics (individual) ===")
    print(f"GNN  PR-AUC: {average_precision_score(y_val, gnn_val):.4f}")
    print(f"XGB  PR-AUC: {average_precision_score(y_val, xgb_val):.4f}")

    # ── Grid search weight on val ──
    print("\n=== Weight scan on VAL ===")
    print(f"{'w_gnn':>7} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print("-" * 40)
    best_f1, best_w, best_thresh = 0, 0.5, 0.5
    for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        ens_prob = w * gnn_val + (1 - w) * xgb_val
        # Find best threshold for this weight
        for thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
            pred = (ens_prob > thresh).astype(int)
            f = f1_score(y_val, pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_w = w
                best_thresh = thresh
        # Print at thresh=0.5 as reference
        pred = (ens_prob > 0.5).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred, zero_division=0)
        print(f"{w:>7.2f} | {p:>7.4f} | {r:>7.4f} | {f:>7.4f}")

    print(f"\nBest: w_gnn={best_w:.2f}, threshold={best_thresh:.2f}, val_F1={best_f1:.4f}")

    # ── Test with best weights ──
    y_test = y_all[test_ids]
    gnn_test = gnn_probs_all[test_ids]
    xgb_test = xgb_probs_all[test_ids]
    ens_test = best_w * gnn_test + (1 - best_w) * xgb_test
    test_pred = (ens_test > best_thresh).astype(int)

    print(f"\n=== TEST metrics (ensemble w_gnn={best_w:.2f}) ===")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, ens_test):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, ens_test):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
    print(f"\n{classification_report(y_test, test_pred, target_names=['Normal','Theft'], zero_division=0)}")

    # Compare individual
    print("\n=== TEST — individual models (baseline) ===")
    for name, prob in [("XGBoost", xgb_test), ("GNN", gnn_test)]:
        pred = (prob > 0.5).astype(int)
        print(f"{name:8s} F1={f1_score(y_test, pred, zero_division=0):.4f} | "
              f"PR-AUC={average_precision_score(y_test, prob):.4f}")


if __name__ == "__main__":
    main()
