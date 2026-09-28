import numpy as np

d = np.load("data/processed/sgcc_clean.npz", allow_pickle=True)

X = d["X"]
y = d["y"]

print("X shape:", X.shape)          # expect (5000, 1033)
print("y shape:", y.shape)          # expect (5000,)
print("X dtype:", X.dtype)          # expect float32
print("X range:", X.min(), "->", X.max())
print("Class balance: normal =", (y == 0).sum(), "| theft =", (y == 1).sum())
print("Theft ratio:", round((y == 1).mean() * 100, 2), "%")
print("NaNs in X:", np.isnan(X).sum())
