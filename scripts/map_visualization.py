
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def load_data(data_dir="."):
    print(f"Looking for parquet files in {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "merged_rollouts_part_*.parquet"))
    
    if not files:
        if os.path.exists(os.path.join(data_dir, "merged_rollouts.parquet")):
            files = [os.path.join(data_dir, "merged_rollouts.parquet")]
        else:
            print("No merged_rollouts parquet files found. Please run merge_rollouts.py first.")
            return None

    print(f"Found {len(files)} files. Loading data...")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} rows of data.")
    return df

def visualize_high_res_map(df, output_dir):
    print("--- Generating High-Res Track Map ---")
    
    if "x" not in df.columns or "z" not in df.columns:
        print("Error: 'x' or 'z' columns missing.")
        return

    # 1. Plot ALL points with high transparency to define the road
    plt.figure(figsize=(12, 12), facecolor='black')
    ax = plt.gca()
    ax.set_facecolor('black')
    
    print("Plotting all points (this might take a moment)...")
    plt.scatter(df["x"], df["z"], c="white", s=1, alpha=0.02, label="Visited Area")
    
    # 2. Overlay the "Racing Line" (Low Epsilon)
    if "epsilon" in df.columns:
        print("Overlaying Racing Line...")
        # Filter for lowest 10% epsilon
        low_eps_threshold = df["epsilon"].quantile(0.10)
        racing_line = df[df["epsilon"] <= low_eps_threshold]
        
        plt.scatter(racing_line["x"], racing_line["z"], c="cyan", s=1, alpha=0.1, label="Racing Line (Low Eps)")
        
    # 3. Highlight Start and Finish
    # Start: First point of any episode (or meters_advanced ~ 0)
    # Finish: Last point of successful episodes (or max meters_advanced)
    
    # We can just take the first few points of the dataframe as "Start"
    start_points = df.head(100)
    plt.scatter(start_points["x"], start_points["z"], c="lime", s=50, marker="^", label="Start", edgecolors="black", zorder=10)
    
    # For Finish, we look for points with max 'meters_advanced_along_centerline' if available
    if "meters_advanced_along_centerline" in df.columns:
        max_dist = df["meters_advanced_along_centerline"].max()
        # Take top 0.1% furthest points
        finish_points = df[df["meters_advanced_along_centerline"] > max_dist * 0.99]
        plt.scatter(finish_points["x"], finish_points["z"], c="red", s=50, marker="X", label="Finish", edgecolors="black", zorder=10)
        
    plt.legend(loc="upper right", facecolor="black", labelcolor="white")
    
    plt.title("High-Resolution Track Map (Start=Green, Finish=Red)", color="white", fontsize=16)
    plt.xlabel("X", color="white")
    plt.ylabel("Z", color="white")
    plt.tick_params(colors="white")
    plt.axis("equal")
    plt.grid(False)
    
    output_path = os.path.join(output_dir, "high_res_track_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="black")
    plt.close()
    print(f"Map saved to {output_path}")

def main():
    output_dir = "analysis_results_map"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    visualize_high_res_map(df, output_dir)
    
    print(f"Map visualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
