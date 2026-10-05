"""
Create a 10,000-customer demo dataset from REAL SGCC data.

- 10,000 random customers from SGCC (with their real FLAG labels)
- Expected ~850 thieves (SGCC has ~8.5% theft rate)
- Model should flag 800-900 customers

Use this for medium-scale demos.
"""

import numpy as np
import pandas as pd
import os

CONFIG = {
    "n_customers":    10000,
    "seed":           42,
    "sgcc_path":      "data/raw/sgcc.csv",
    "output_csv":     "data/demo/demo_10k.csv",
    "ground_truth":   "data/demo/demo_10k_truth.txt",
}


def main():
    rng = np.random.default_rng(CONFIG["seed"])
    n = CONFIG["n_customers"]

    print(f"Loading SGCC data...")
    df_full = pd.read_csv(CONFIG["sgcc_path"], low_memory=False)

    id_col = "CONS_NO"
    label_col = "FLAG"

    # Sample 10,000 customers randomly
    idx = rng.choice(len(df_full), size=n, replace=False)
    demo_df = df_full.iloc[idx].copy().reset_index(drop=True)

    n_thieves = int((demo_df[label_col] == 1).sum())
    print(f"  Picked {n:,} customers")
    print(f"  Actual thieves (from SGCC labels): {n_thieves:,}")
    print(f"  Honest: {n - n_thieves:,}")
    print(f"  Theft rate: {n_thieves/n*100:.2f}%")

    # Rename IDs
    demo_df[id_col] = [f"DISCOM{i+1:06d}" for i in range(n)]

    os.makedirs("data/demo", exist_ok=True)
    demo_df.to_csv(CONFIG["output_csv"], index=False)
    print(f"\n✅ Saved: {CONFIG['output_csv']}")
    print(f"   Shape: {demo_df.shape}")
    print(f"   File size: {os.path.getsize(CONFIG['output_csv'])/1024/1024:.1f} MB")

    with open(CONFIG["ground_truth"], "w") as f:
        f.write("10K DEMO DATASET — GROUND TRUTH\n")
        f.write("=" * 50 + "\n")
        f.write(f"Source:        SGCC real-world data\n")
        f.write(f"Total:         {n:,}\n")
        f.write(f"Actual thieves: {n_thieves:,}\n")
        f.write(f"Honest:        {n - n_thieves:,}\n")
        f.write(f"Theft rate:    {n_thieves/n*100:.2f}%\n\n")
        f.write(f"Expected: model should flag {int(n_thieves*0.7):,}-{int(n_thieves*1.1):,} customers\n")

    print(f"✅ Saved ground truth: {CONFIG['ground_truth']}")
    print()
    print("=" * 50)
    print(f"DEMO READY: {n_thieves:,} actual thieves out of {n:,} customers")
    print("=" * 50)


if __name__ == "__main__":
    main()