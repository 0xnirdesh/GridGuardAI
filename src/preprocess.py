"""
SGCC Electricity Theft Dataset - Preprocessing Pipeline
"""

import os
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
RAW_PATH      = "data/raw/sgcc.csv"
OUTPUT_DIR    = "data/processed"
MAX_CUSTOMERS = None          # None = use full dataset (42,372 customers)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD & RESHAPE
# ─────────────────────────────────────────────
def load_and_reshape(path):
    print("[1/5] Loading dataset...")
    df = pd.read_csv(path, low_memory=False)
    print(f"    Raw shape: {df.shape}")

    id_col    = "CONS_NO"
    label_col = "FLAG"
    date_cols = [c for c in df.columns if c not in (id_col, label_col)]
    print(f"    Date columns: {len(date_cols)}  (expected ~1034)")

    flags = df[label_col].astype(int).values
    ids   = df[id_col].values

    consumption = df[date_cols].apply(pd.to_numeric, errors="coerce")
    consumption.index = ids
    consumption.index.name = "CONS_NO"
    consumption.columns = pd.to_datetime(date_cols, format="%m/%d/%Y")

    print(f"    Consumption matrix: {consumption.shape}")
    print(f"    Class balance: normal={np.sum(flags==0)}, theft={np.sum(flags==1)}")
    return consumption, flags, ids


# ─────────────────────────────────────────────
# 2. SUBSET
# ─────────────────────────────────────────────
def subset(consumption, flags, ids, max_n):
    if max_n is None or max_n >= len(consumption):
        return consumption, flags, ids
    print(f"[2/5] Subsetting to first {max_n} customers...")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(consumption), size=max_n, replace=False)
    idx.sort()
    return consumption.iloc[idx], flags[idx], ids[idx]


# ─────────────────────────────────────────────
# 3. HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing(consumption):
    print("[3/5] Handling missing values...")
    before = consumption.isna().sum().sum()
    print(f"    Missing before: {before:,}")

    bad_days = consumption.columns[consumption.isna().mean() > 0.95]
    if len(bad_days):
        print(f"    Dropping fully-empty days: {list(bad_days.date)}")
        consumption = consumption.drop(columns=bad_days)

    consumption = consumption.interpolate(axis=1, method="linear",
                                          limit_direction="both")
    consumption = consumption.T.fillna(consumption.mean(axis=1)).T
    consumption = consumption.fillna(0.0)

    after = consumption.isna().sum().sum()
    print(f"    Missing after:  {after:,}")
    return consumption


# ─────────────────────────────────────────────
# 4. OUTLIER CLIP + NORMALIZE (returns raw too)
# ─────────────────────────────────────────────
def clean_and_normalize(consumption, sigma=3):
    print(f"[4/5] Clipping outliers (>{sigma}σ) and normalizing per customer...")
    mean = consumption.mean(axis=1)
    std  = consumption.std(axis=1).replace(0, 1)
    lower = mean - sigma * std
    upper = mean + sigma * std
    consumption = consumption.clip(lower=lower, upper=upper, axis=0)

    # Save raw values BEFORE normalization
    raw_values = consumption.values.copy()

    # Min-Max per customer
    mn = consumption.min(axis=1)
    mx = consumption.max(axis=1).replace(0, 1)
    consumption = consumption.sub(mn, axis=0).div(mx - mn, axis=0)

    print(f"    Final range: [{consumption.values.min():.3f}, "
          f"{consumption.values.max():.3f}]")
    return consumption, raw_values


# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
def save(consumption, flags, ids, raw_values):
    print("[5/5] Saving processed artifacts...")
    X     = consumption.values.astype(np.float32)
    X_raw = raw_values.astype(np.float32)
    y     = flags.astype(np.int64)
    ids   = np.array(ids)

    out = os.path.join(OUTPUT_DIR, "sgcc_clean.npz")
    np.savez_compressed(
        out,
        X=X,
        X_raw=X_raw,
        y=y,
        ids=ids,
        dates=consumption.columns.values.astype(str),
    )
    print(f"    Saved: {out}")
    print(f"    X shape: {X.shape}, X_raw shape: {X_raw.shape}, y shape: {y.shape}")
    print(f"    Memory: {X.nbytes / 1e6:.1f} MB (x2 for raw)")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    consumption, flags, ids = load_and_reshape(RAW_PATH)
    consumption, flags, ids = subset(consumption, flags, ids, MAX_CUSTOMERS)
    consumption = handle_missing(consumption)
    consumption, raw_values = clean_and_normalize(consumption)
    save(consumption, flags, ids, raw_values)
    print("\n✅ Preprocessing complete.")