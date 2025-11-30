
import pandas as pd
import glob
import os
import numpy as np

def clean_data(data_dir):
    print(f"Checking data in {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True)
    
    if not files:
        print("No parquet files found.")
        return

    corrupted_files = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Check for NaNs in float columns
            if df.select_dtypes(include=[np.number]).isnull().any().any():
                print(f"❌ CORRUPTED: Found NaN in {os.path.basename(f)}")
                corrupted_files.append(f)
            # Check for Infs
            elif np.isinf(df.select_dtypes(include=[np.number])).any().any():
                print(f"❌ CORRUPTED: Found Inf in {os.path.basename(f)}")
                corrupted_files.append(f)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    if corrupted_files:
        print(f"\n⚠️ Found {len(corrupted_files)} corrupted files.")
        print("Deleting them now...")
        for f in corrupted_files:
            try:
                os.remove(f)
                print(f"   Deleted: {os.path.basename(f)}")
            except Exception as e:
                print(f"   Error deleting {f}: {e}")
        print("\n✅ Cleanup Complete.")
    else:
        print("\n✅ Data looks clean (no NaNs/Infs found).")

if __name__ == "__main__":
    clean_data("save/felix_test_training/good_runs")
    clean_data("save/felix_test_training/best_runs")
