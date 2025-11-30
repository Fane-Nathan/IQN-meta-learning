
import pandas as pd
import sys

try:
    df = pd.read_parquet("merged_rollouts_part_000.parquet")
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows of 'epsilon' (if exists):")
    if "epsilon" in df.columns:
        print(df["epsilon"].head())
    else:
        print("'epsilon' column NOT found.")
        
    print("\nFirst 5 rows of 'learning_rate' (if exists):")
    if "learning_rate" in df.columns:
        print(df["learning_rate"].head())
    else:
        print("'learning_rate' column NOT found.")
except Exception as e:
    print(f"Error: {e}")
