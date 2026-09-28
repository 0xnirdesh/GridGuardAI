"""
Random Forest classifier for electricity theft detection.

Uses BOTH normalized features (shape) and raw magnitude features (scale).
The combination is what makes this strong on SGCC.
"""

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score,
                             confusion_matrix, classification_report)
import joblib


# ─────────────────────────────────────────────
# SUPERVISED SPLIT
# ─────────────────────────────────────────────
def split_supervised(y, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y == 0)[0]
    theft_idx  = np.where(y == 1)[0]
    rng.shuffle(normal_idx)
    rng.shuffle(theft_idx)

    def split_class(idx, ratio_tr, ratio_val):
        n = len(idx)
        n_tr = int(n * ratio_tr)
        n_v  = int(n * ratio_val)
        return idx[:n_tr], idx[n_tr:n_tr + n_v], idx[n_tr + n_v:]

    tr_n, v_n, te_n = split_class(normal_idx, train_ratio, val_ratio)
    tr_t, v_t, te_t = split_class(theft_idx,  train_ratio, val_ratio)

    return (np.concatenate([tr_n, tr_t]),
            np.concatenate([v_n,  v_t]),
            np.concatenate([te_n, te_t]))


# ─────────────────────────────────────────────
# FEATURE ENGINEERING — uses BOTH X and X_raw
# ─────────────────────────────────────────────
def extract_features(X_norm, X_raw):
    """
    X_norm: (n, T) normalized to [0, 1]
    X_raw : (n, T) raw kWh values
    """
    print("Extracting features...")
    n, T = X_norm.shape
    feats, names = [], []

    # ── Shape features (from normalized X) ──
    mean_n = X_norm.mean(axis=1)
    std_n  = X_norm.std(axis=1)
    cv_n   = std_n / (mean_n + 1e-6)
    skew   = ((X_norm - mean_n[:, None]) ** 3).mean(axis=1) / (std_n ** 3 + 1e-6)
    kurt   = ((X_norm - mean_n[:, None]) ** 4).mean(axis=1) / (std_n ** 4 + 1e-6)
    feats += [mean_n, std_n, cv_n, skew, kurt]
    names += ["norm_mean", "norm_std", "norm_cv", "norm_skew", "norm_kurt"]

    t = np.arange(T); t_mean = t.mean(); t_var = ((t - t_mean) ** 2).sum()
    slope_n = ((X_norm - mean_n[:, None]) * (t - t_mean)).sum(axis=1) / t_var
    feats.append(slope_n); names.append("norm_slope")

    xc = X_norm - mean_n[:, None]
    autocorr = (xc[:, :-1] * xc[:, 1:]).mean(axis=1) / (std_n ** 2 + 1e-6)
    feats.append(autocorr); names.append("norm_autocorr")

    # ── Zero-day statistics (from RAW) ──
    zero_count = (X_raw < 1.0).sum(axis=1)
    zero_ratio = zero_count / T
    feats += [zero_count, zero_ratio]
    names += ["raw_zero_count", "raw_zero_ratio"]

    def max_consecutive_zeros(row, thr=1.0):
        mx_run = cur = 0
        for v in row:
            if v < thr:
                cur += 1; mx_run = max(mx_run, cur)
            else:
                cur = 0
        return mx_run
    max_zr = np.array([max_consecutive_zeros(X_raw[i]) for i in range(n)])
    feats.append(max_zr); names.append("raw_max_zero_run")

    # ── Magnitude features (from RAW) ──
    mean_r = X_raw.mean(axis=1)
    std_r  = X_raw.std(axis=1)
    med_r  = np.median(X_raw, axis=1)
    q25_r  = np.percentile(X_raw, 25, axis=1)
    q75_r  = np.percentile(X_raw, 75, axis=1)
    mn_r   = X_raw.min(axis=1)
    mx_r   = X_raw.max(axis=1)
    feats += [mean_r, std_r, med_r, q25_r, q75_r, mn_r, mx_r]
    names += ["raw_mean", "raw_std", "raw_median", "raw_q25",
              "raw_q75", "raw_min", "raw_max"]

    feats.append(np.log1p(mean_r)); names.append("raw_log_mean")
    slope_r = ((X_raw - mean_r[:, None]) * (t - t_mean)).sum(axis=1) / t_var
    feats.append(slope_r); names.append("raw_slope")
    feats.append(q75_r - q25_r); names.append("raw_iqr")
    feats.append(mx_r / (mean_r + 1e-6)); names.append("raw_peak_to_avg")

    # ── NEW: Monthly features (34 months across 1033 days) ──
    # SGCC data: Jan 2014 - Oct 2016. Approx 30 days per month.
    n_months = T // 30
    for m in range(n_months):
        start, end = m * 30, (m + 1) * 30
        month_mean = X_raw[:, start:end].mean(axis=1)
        feats.append(month_mean)
        names.append(f"month{m:02d}_mean")

    # ── NEW: Monthly volatility (std of each month's raw consumption) ──
    for m in range(n_months):
        start, end = m * 30, (m + 1) * 30
        month_std = X_raw[:, start:end].std(axis=1)
        feats.append(month_std)
        names.append(f"month{m:02d}_std")

    # ── NEW: Day-of-week patterns (avg consumption per weekday) ──
    for d in range(7):
        mask = np.array([(i % 7) == d for i in range(T)])
        dow_mean = X_raw[:, mask].mean(axis=1)
        feats.append(dow_mean)
        names.append(f"dow{d}_mean")

    # ── NEW: Rolling window volatility (30-day windows) ──
    # Coefficient of variation in successive windows
    n_windows = T // 30
    window_cvs = []
    for w in range(n_windows):
        start, end = w * 30, (w + 1) * 30
        w_mean = X_raw[:, start:end].mean(axis=1)
        w_std  = X_raw[:, start:end].std(axis=1)
        window_cvs.append(w_std / (w_mean + 1e-6))
    window_cvs = np.stack(window_cvs, axis=1)  # (n, n_windows)
    feats.append(window_cvs.mean(axis=1)); names.append("window_cv_mean")
    feats.append(window_cvs.std(axis=1));  names.append("window_cv_std")
    feats.append(window_cvs.max(axis=1));  names.append("window_cv_max")

    # ── NEW: Longest "low consumption" streak (below 25th percentile) ──
    def max_consecutive_below(row, threshold):
        mx_run = cur = 0
        for v in row:
            if v < threshold:
                cur += 1; mx_run = max(mx_run, cur)
            else:
                cur = 0
        return mx_run
    p25_per_customer = np.percentile(X_raw, 25, axis=1)
    max_low_streak = np.array([
        max_consecutive_below(X_raw[i], p25_per_customer[i])
        for i in range(n)
    ])
    feats.append(max_low_streak); names.append("raw_max_low_streak")

    F = np.stack(feats, axis=1).astype(np.float32)
    print(f"Feature matrix: {F.shape}  ({len(names)} features)")
    return F, names

    def max_consecutive_zeros(row, thr=1.0):
        mx_run = cur = 0
        for v in row:
            if v < thr:
                cur += 1; mx_run = max(mx_run, cur)
            else:
                cur = 0
        return mx_run
    max_zr = np.array([max_consecutive_zeros(X_raw[i]) for i in range(n)])
    feats.append(max_zr); names.append("raw_max_zero_run")

    # ── Magnitude features (from RAW X) — THE KEY SIGNAL ──
    mean_r = X_raw.mean(axis=1)
    std_r  = X_raw.std(axis=1)
    med_r  = np.median(X_raw, axis=1)
    q25_r  = np.percentile(X_raw, 25, axis=1)
    q75_r  = np.percentile(X_raw, 75, axis=1)
    mn_r   = X_raw.min(axis=1)
    mx_r   = X_raw.max(axis=1)
    feats += [mean_r, std_r, med_r, q25_r, q75_r, mn_r, mx_r]
    names += ["raw_mean", "raw_std", "raw_median", "raw_q25",
              "raw_q75", "raw_min", "raw_max"]

    # Log transform of mean (helps with skewed distribution)
    feats.append(np.log1p(mean_r)); names.append("raw_log_mean")

    # Raw slope (trend in actual consumption)
    slope_r = ((X_raw - mean_r[:, None]) * (t - t_mean)).sum(axis=1) / t_var
    feats.append(slope_r); names.append("raw_slope")

    # IQR (spread)
    feats.append(q75_r - q25_r); names.append("raw_iqr")

    # Peak-to-average
    feats.append(mx_r / (mean_r + 1e-6)); names.append("raw_peak_to_avg")

    F = np.stack(feats, axis=1).astype(np.float32)
    print(f"Feature matrix: {F.shape}  ({len(names)} features)")
    return F, names


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)
    X_norm = d["X"]
    X_raw  = d["X_raw"]
    y      = d["y"]
    print(f"Loaded X_norm={X_norm.shape}, X_raw={X_raw.shape}, y={y.shape}")
    print(f"Class balance: normal={np.sum(y==0)}, theft={np.sum(y==1)}")

    F, feat_names = extract_features(X_norm, X_raw)

    train_ids, val_ids, test_ids = split_supervised(y)
    print(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    X_train, y_train = F[train_ids], y[train_ids]
    X_test,  y_test  = F[test_ids],  y[test_ids]

    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    print("\n=== Customer-level metrics (test) ===")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1       : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"PR-AUC   : {average_precision_score(y_test, y_prob):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")
    print("\nFull report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Theft"],
                                zero_division=0))

    print("\n=== Top 10 Feature Importances ===")
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    for i in order[:10]:
        print(f"  {feat_names[i]:<20} {importances[i]:.4f}")

    os.makedirs("models", exist_ok=True)
    path = "models/random_forest.pkl"
    joblib.dump({"model": rf, "feature_names": feat_names}, path)
    print(f"\n✅ Model saved to {path}")


if __name__ == "__main__":
    main()