"""
Create demo dataset from REAL SGCC customers with real theft labels.

This is the honest approach: model sees SGCC data it was trained on,
demo shows real-world performance.
"""

import numpy as np
import pandas as pd
import os

CONFIG = {
    "n_customers":    500,
    "seed":           42,
    "sgcc_path":      "data/raw/sgcc.csv",
    "output_csv":     "data/demo/demo_real.csv",
    "ground_truth":   "data/demo/demo_real_truth.txt",
}


def main():
    rng = np.random.default_rng(CONFIG["seed"])
    n = CONFIG["n_customers"]

    print(f"Loading SGCC data...")
    df_full = pd.read_csv(CONFIG["sgcc_path"], low_memory=False)

    id_col = "CONS_NO"
    label_col = "FLAG"

    # Sample 500 customers randomly (keeping their real labels)
    idx = rng.choice(len(df_full), size=n, replace=False)
    demo_df = df_full.iloc[idx].copy().reset_index(drop=True)

    n_thieves = int((demo_df[label_col] == 1).sum())
    print(f"  Picked {n} customers")
    print(f"  Actual thieves (from SGCC labels): {n_thieves}")
    print(f"  Honest: {n - n_thieves}")

    # Rename IDs so it looks like a fresh customer list
    demo_df[id_col] = [f"DISCOM{i+1:05d}" for i in range(n)]

    os.makedirs("data/demo", exist_ok=True)
    demo_df.to_csv(CONFIG["output_csv"], index=False)
    print(f"\n✅ Saved: {CONFIG['output_csv']}")
    print(f"   Shape: {demo_df.shape}")

    with open(CONFIG["ground_truth"], "w") as f:
        f.write("DEMO DATASET — GROUND TRUTH\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source:        SGCC real-world data\n")
        f.write(f"Total:         {n}\n")
        f.write(f"Actual thieves: {n_thieves}\n")
        f.write(f"Honest:        {n - n_thieves}\n")
        f.write(f"Theft rate:    {n_thieves/n*100:.2f}%\n\n")
        f.write(f"Expected: model should flag {int(n_thieves*0.7)}-{int(n_thieves*1.1)} customers\n")

    print(f"✅ Saved ground truth: {CONFIG['ground_truth']}")
    print()
    print("=" * 50)
    print(f"DEMO READY: {n_thieves} actual thieves out of {n} customers")
    print("=" * 50)


if __name__ == "__main__":
    main()