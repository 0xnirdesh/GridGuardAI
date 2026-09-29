"""
Build a synthetic KNN graph from customer features.

Since SGCC dataset does not include actual grid topology, we construct
a graph based on feature similarity: each customer is connected to its
K nearest neighbors in feature space.

This captures the intuition that customers with similar consumption
patterns may belong to the same feeder / region.
"""

import os
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
import joblib

from train_classifier import extract_features, split_supervised


def build_knn_graph(features, k=10, metric="cosine"):
    """
    Build a symmetric KNN graph.
    features: (n, d) array
    Returns: edge_index (2, E) numpy array
    """
    print(f"Building KNN graph (k={k}, metric={metric})...")
    n = features.shape[0]

    # Fit KNN — k+1 because the point itself is its own neighbor
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=-1)
    nn.fit(features)
    distances, indices = nn.kneighbors(features)

    # Build edges (skip self-loop at column 0)
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].flatten()

    # Make symmetric — add both directions
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])

    edge_index = np.stack([src_sym, dst_sym], axis=0)
    print(f"  Nodes: {n:,}")
    print(f"  Edges (directed): {edge_index.shape[1]:,}")
    print(f"  Avg degree: {edge_index.shape[1] / n:.1f}")
    return edge_index


def main():
    # Load features (same 100 features as XGBoost)
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm, X_raw, y = d["X"], d["X_raw"], d["y"]
    print(f"Loaded X_norm={X_norm.shape}, y={y.shape}")

    F, feat_names = extract_features(X_norm, X_raw)

    # Standardize features (important for cosine / euclidean KNN)
    mu = F.mean(axis=0, keepdims=True)
    sigma = F.std(axis=0, keepdims=True) + 1e-6
    F_std = (F - mu) / sigma

    # Build graph
    edge_index = build_knn_graph(F_std, k=10, metric="cosine")

    # Save everything GNN training needs
    os.makedirs("data/processed", exist_ok=True)
    out = "data/processed/gnn_data.npz"
    np.savez_compressed(
        out,
        node_features=F_std.astype(np.float32),
        edge_index=edge_index.astype(np.int64),
        y=y.astype(np.int64),
        feature_names=np.array(feat_names),
        feat_mu=mu.astype(np.float32),
        feat_sigma=sigma.astype(np.float32),
    )
    print(f"\n✅ Saved to {out}")

    # Also save the train/val/test split for consistency
    train_ids, val_ids, test_ids = split_supervised(y)
    np.savez_compressed(
        "data/processed/gnn_splits.npz",
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
    )
    print(f"✅ Saved splits to data/processed/gnn_splits.npz")


if __name__ == "__main__":
    main()
