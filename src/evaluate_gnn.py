"""
Evaluate GNN + tune threshold on test set.
"""
import numpy as np
import torch
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)
from model_gnn import TheftGNN

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    print(f"Device: {DEVICE}")
    d = np.load("data/processed/gnn_data.npz", allow_pickle=True)
    splits = np.load("data/processed/gnn_splits.npz")

    X = torch.from_numpy(d["node_features"]).float().to(DEVICE)
    edge_index = torch.from_numpy(d["edge_index"]).long().to(DEVICE)
    y = d["y"]

    val_ids  = splits["val_ids"]
    test_ids = splits["test_ids"]

    ckpt = torch.load("models/gnn.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = TheftGNN(in_dim=cfg["in_dim"], hidden_dim=cfg["hidden_dim"],
                     num_layers=cfg["num_layers"], dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded (val PR-AUC: {ckpt['best_val_auc']:.4f})")

    model.eval()
    with torch.no_grad():
        logits = model(X, edge_index)
        probs = torch.sigmoid(logits).cpu().numpy()

    val_prob = probs[val_ids]
    test_prob = probs[test_ids]
    y_val = y[val_ids]
    y_test = y[test_ids]

    # Threshold scan
    print(f"\n{'Thresh':>8} | {'Prec':>7} | {'Rec':>7} | {'F1':>7}")
    print("-" * 40)
    best_f1, best_thresh = 0, 0.5
    for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        pred = (val_prob > thresh).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f = f1_score(y_val, pred, zero_division=0)
        marker = ""
        if f > best_f1:
            best_f1, best_thresh = f, thresh
            marker = " ←"
        print(f"{thresh:>8.2f} | {p:>7.4f} | {r:>7.4f} | {f:>7.4f}{marker}")

    print(f"\nBest threshold: {best_thresh} | Best val F1: {best_f1:.4f}")

    # Test
    test_pred = (test_prob > best_thresh).astype(int)
    print(f"\n=== TEST metrics @ threshold = {best_thresh} ===")
    print(f"Precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, test_prob):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, test_prob):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, test_pred)}")
    print(f"\n{classification_report(y_test, test_pred, target_names=['Normal','Theft'], zero_division=0)}")


if __name__ == "__main__":
    main()
