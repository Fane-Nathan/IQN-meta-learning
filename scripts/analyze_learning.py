
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def analyze_learning(data_dir=".", output_dir="analysis_results"):
    print(f"Looking for parquet files in {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "merged_rollouts_part_*.parquet"))
    
    if not files:
        # Fallback to single merged file if chunks don't exist
        if os.path.exists(os.path.join(data_dir, "merged_rollouts.parquet")):
            files = [os.path.join(data_dir, "merged_rollouts.parquet")]
        else:
            print("No merged_rollouts parquet files found. Please run merge_rollouts.py first.")
            return

    print(f"Found {len(files)} files. Loading data...")
    
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows of data.")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # --- Feature Extraction ---
    print("Extracting features...")
    
    # 1. Speed Extraction from state_float
    # Indices 58, 59, 60 are Linear Velocity x, y, z
    # We need to stack the state_float arrays to process them efficiently
    # Filter out rows with inconsistent state_float lengths
    # First, drop None values
    df = df.dropna(subset=["state_float"])
    
    # First, find the most common length (mode)
    lengths = df["state_float"].apply(len)
    mode_len = lengths.mode()[0]
    print(f"Most common state_float length: {mode_len}")
    
    # Keep only rows with that length
    valid_rows = lengths == mode_len
    n_dropped = len(df) - valid_rows.sum()
    if n_dropped > 0:
        print(f"Dropping {n_dropped} rows with inconsistent state_float lengths.")
        df = df[valid_rows].copy()
    
    state_floats = np.vstack(df["state_float"].values)
    
    # Check if we have enough columns (handling potential schema mismatches)
    if state_floats.shape[1] > 60:
        vx = state_floats[:, 58]
        vy = state_floats[:, 59]
        vz = state_floats[:, 60]
        speed = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6 # Convert m/s to km/h (approx)
        df["speed_kmh"] = speed
    else:
        print("Warning: state_float has fewer columns than expected. Skipping speed extraction.")
        df["speed_kmh"] = 0

    # 2. Steering (input_w)
    # input_w is already a column
    
    # --- Visualization ---
    print("Generating plots...")

    # Plot 1: Speed Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df["speed_kmh"], bins=50, color='skyblue', edgecolor='black')
    plt.title("Speed Distribution (km/h)")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "speed_distribution.png"))
    print(f"Saved speed_distribution.png")

    # Plot 2: Gas/Acceleration Usage (input_w appears to be Gas)
    plt.figure(figsize=(10, 6))
    # Convert boolean/int to float
    gas_data = pd.to_numeric(df["input_w"], errors='coerce').fillna(0).astype(float)
    plt.hist(gas_data, bins=10, color='salmon', edgecolor='black')
    plt.title("Gas/Acceleration Input Distribution")
    plt.xlabel("Gas Input (0.0 to 1.0)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "gas_distribution.png"))
    print(f"Saved gas_distribution.png")

    # Plot 3: Speed over Time (using index as proxy for time if step_race_times is not monotonic across files)
    # If we have 'step_race_times', we can try to use it, but merging might mess up order.
    # We'll use simple index for now.
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["speed_kmh"], alpha=0.5, linewidth=0.5)
    plt.title("Speed over Training Steps")
    plt.xlabel("Step Index")
    plt.ylabel("Speed (km/h)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "speed_over_time.png"))
    print(f"Saved speed_over_time.png")

    print("Analysis complete!")

if __name__ == "__main__":
    analyze_learning()
