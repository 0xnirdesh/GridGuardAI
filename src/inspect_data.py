
import pandas as pd
from pathlib import Path

# Dataset location
file_path = Path("data/raw/sgcc.csv")


print("Loading dataset...")

# Read the CSV
df = pd.read_csv(file_path, low_memory=False)

print("\n========== DATASET INFORMATION ==========")

# Number of rows and columns
print("Shape:", df.shape)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# First 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Data types
print("\nColumn data types:")
print(df.dtypes)

# Missing values
print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

# Dataset size in memory
print("\nMemory usage (MB):")
print(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2))

print("\nInspection complete!")


