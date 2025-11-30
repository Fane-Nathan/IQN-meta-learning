
import pandas as pd
import os

def clean_metrics(csv_path):
    print(f"Cleaning {csv_path}...")
    if not os.path.exists(csv_path):
        print("File not found.")
        return

    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        original_count = len(df)
        
        # Convert columns to numeric, coercing errors to NaN (handles 'tensor(...)')
        # We need to clean the 'loss' column first if it has 'tensor(' strings
        if df["loss"].dtype == object:
            df["loss"] = df["loss"].astype(str).str.replace("tensor(", "", regex=False).str.replace(")", "", regex=False)
        
        df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
        df["grad_norm"] = pd.to_numeric(df["grad_norm"], errors="coerce")
        
        # Drop rows with NaN in loss or grad_norm
        df_clean = df.dropna(subset=["loss", "grad_norm"])
        
        cleaned_count = len(df_clean)
        removed_count = original_count - cleaned_count
        
        if removed_count > 0:
            # Save back
            df_clean.to_csv(csv_path, index=False)
            print(f"✅ Removed {removed_count} rows with NaNs.")
            print(f"   Remaining rows: {cleaned_count}")
        else:
            print("✅ No NaNs found. File is already clean.")
            
    except Exception as e:
        print(f"Error cleaning CSV: {e}")

if __name__ == "__main__":
    clean_metrics("save/felix_test_training/training_metrics.csv")
