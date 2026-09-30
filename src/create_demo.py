"""
Create a controlled demo dataset for showing to teachers.

- 500 customers
- 20 real thieves (obvious patterns)
- 480 honest customers
- Ground truth saved separately

This lets you say: "There are exactly 20 thieves in this data"
and check how many the model catches.
"""

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
CONFIG = {
    "n_customers":    500,
    "n_thieves":      20,
    "n_days":         1033,          # SAME AS SRGCC
    "seed":           42,
    "output_csv":     "data/demo/demo_test.csv",
    "ground_truth":   "data/demo/demo_ground_truth.txt",
}
# ─────────────────────────────────────────────


def make_honest_profile(rng, n_days):
    """Normal consumption: base + seasonal + noise."""
    base = rng.uniform(8, 15)
    seasonal = 3 * np.sin(np.linspace(0, 4 * np.pi, n_days))
    weekly = 1.5 * np.sin(np.linspace(0, 2 * np.pi * n_days / 7, n_days))
    noise = rng.normal(0, 1.2, n_days)
    series = base + seasonal + weekly + noise
    return np.clip(series, 0.5, None)


def make_thief_profile(rng, n_days, theft_type):
    """Thief consumption: normal then drops significantly."""
    base = make_honest_profile(rng, n_days)

    if theft_type == "sudden_drop":
        # Consumption drops by 70% after random day
        start = rng.integers(n_days // 3, 2 * n_days // 3)
        base[start:] = base[start:] * rng.uniform(0.15, 0.35)

    elif theft_type == "meter_tamper":
        # Consumption becomes near-zero for long stretches
        for _ in range(rng.integers(4, 8)):
            start = rng.integers(0, n_days - 30)
            length = rng.integers(15, 40)
            base[start:start + length] = rng.uniform(0.0, 0.5)

    elif theft_type == "gradual_decline":
        # Gradual decline over time (billing fraud)
        decline = np.linspace(1.0, 0.2, n_days)
        base = base * decline

    elif theft_type == "night_zero":
        # Day consumption normal, but night hours zero (bypass)
        # Since we only have daily data, simulate alternating days
        for i in range(0, n_days, 2):
            base[i] = base[i] * rng.uniform(0.05, 0.3)

    return np.clip(base, 0.0, None)


def main():
    rng = np.random.default_rng(CONFIG["seed"])
    n = CONFIG["n_customers"]
    k = CONFIG["n_thieves"]
    T = CONFIG["n_days"]

    print(f"Creating demo dataset: {n} customers, {k} thieves, {T} days")

    # Pick which customer indices are thieves
    thief_idx = rng.choice(n, size=k, replace=False)
    thief_idx_set = set(thief_idx.tolist())

    theft_types = ["sudden_drop", "meter_tamper", "gradual_decline", "night_zero"]

    rows = []
    labels = []
    ids = []

    for i in range(n):
        customer_id = f"DEMO{i+1:05d}"

        if i in thief_idx_set:
            ttype = theft_types[i % len(theft_types)]
            series = make_thief_profile(rng, T, ttype)
            labels.append(1)
        else:
            series = make_honest_profile(rng, T)
            labels.append(0)

        rows.append(series)
        ids.append(customer_id)

    # Build DataFrame in SGCC format: CONS_NO | date columns | FLAG
    dates = pd.date_range("2023-01-01", periods=T, freq="D")
    date_cols = [d.strftime("%m/%d/%Y") for d in dates]

    df = pd.DataFrame(rows, columns=date_cols)
    df.insert(0, "CONS_NO", ids)
    df["FLAG"] = labels

    # Shuffle rows so thieves aren't clustered
    df = df.sample(frac=1, random_state=CONFIG["seed"]).reset_index(drop=True)

    # Save CSV
    import os
    os.makedirs("data/demo", exist_ok=True)
    df.to_csv(CONFIG["output_csv"], index=False)
    print(f"✅ Saved demo CSV: {CONFIG['output_csv']}")
    print(f"   Shape: {df.shape}")

    # Save ground truth
    with open(CONFIG["ground_truth"], "w") as f:
        f.write("DEMO DATASET — GROUND TRUTH\n")
        f.write("=" * 40 + "\n")
        f.write(f"Total customers:  {n}\n")
        f.write(f"Actual thieves:   {k}\n")
        f.write(f"Honest customers: {n - k}\n")
        f.write(f"Days of data:     {T}\n")
        f.write(f"Theft rate:       {k/n*100:.2f}%\n\n")
        f.write("Thief patterns used:\n")
        f.write("  - sudden_drop     (consumption drops 65-85% after mid-point)\n")
        f.write("  - meter_tamper    (multiple near-zero periods)\n")
        f.write("  - gradual_decline (consumption declines over time)\n")
        f.write("  - night_zero      (alternating low-consumption days)\n\n")
        f.write("Expected model output: should flag 18-25 customers as suspicious.\n")
        f.write("Perfect score: flags ~20 customers.\n")

    print(f"✅ Saved ground truth: {CONFIG['ground_truth']}")
    print()
    print("=" * 50)
    print(f"DEMO READY: {k} thieves out of {n} customers")
    print("=" * 50)
    print()
    print("Teacher ko dikhane ke liye bolo:")
    print(f'  "Is CSV me 500 customers hain, jisme {k} chori kar rahe hain."')
    print(f'  "Model ne X flag kiye — kitne sahi hain?"')


if __name__ == "__main__":
    main()
