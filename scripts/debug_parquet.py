
import pandas as pd
import os

def debug_parquet(path):
    print(f"Reading {path}...")
    try:
        df = pd.read_parquet(path)
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows:")
        print(df.head())
        
        if 'x' in df.columns:
            print("\n'x' column found!")
        else:
            print("\n'x' column NOT found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "save/felix_test_training/best_runs/hock_57060/rollout_data_hock_57060.parquet"
    debug_parquet(path)
