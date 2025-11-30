
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    # Keep only recent data (e.g. last 500k frames) to ensure Teacher sees current performance
    max_rows = 500_000
    if len(df) > max_rows:
        print(f"Filtering to last {max_rows} rows for freshness...")
        df = df.iloc[-max_rows:].copy()
    
    # Filter out rows with inconsistent state_float lengths or None
    df = df.dropna(subset=["state_float"])
    lengths = df["state_float"].apply(len)
    mode_len = lengths.mode()[0]
    df = df[lengths == mode_len].copy()
    
    return df

def extract_features(df):
    print("Extracting features...")
    # Stack state_float
    state_floats = np.vstack(df["state_float"].values)
    
    # Speed (Indices 58, 59, 60)
    vx = state_floats[:, 58]
    vy = state_floats[:, 59]
    vz = state_floats[:, 60]
    speed = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6 # km/h
    
    df["speed_kmh"] = speed
    
    # Ensure current_zone_idx is present
    if "current_zone_idx" not in df.columns:
        print("Error: 'current_zone_idx' not found in dataframe.")
        return None
        
    return df

def analyze_zone_performance(df, output_dir):
    print("--- Zone Performance Analysis ---")
    
    # Group by Zone
    # Calculate:
    # 1. Total Frames (Time Spent)
    # 2. Avg Speed
    # 3. Crash Count (Speed < 10)
    # 4. Crash Rate
    
    # Define Crash
    df["is_crash"] = df["speed_kmh"] < 10
    
    zone_stats = df.groupby("current_zone_idx").agg(
        total_frames=("speed_kmh", "count"),
        avg_speed=("speed_kmh", "mean"),
        crash_count=("is_crash", "sum")
    ).reset_index()
    
    zone_stats["crash_rate"] = (zone_stats["crash_count"] / zone_stats["total_frames"]) * 100
    
    print("Top 5 Struggle Zones (Highest Crash Rate):")
    print(zone_stats.sort_values("crash_rate", ascending=False).head(5))
    
    # Save stats
    zone_stats.to_csv(os.path.join(output_dir, "zone_stats.csv"), index=False)
    
    # Plot Crash Rate (Top 50 Worst Zones)
    plt.figure(figsize=(14, 8))
    
    # Filter for zones with significant traffic (e.g., > 10 frames) to avoid noise
    significant_zones = zone_stats[zone_stats["total_frames"] > 10]
    
    # Get Top 50 by Crash Rate
    top_crash_zones = significant_zones.sort_values("crash_rate", ascending=False).head(50)
    
    sns.barplot(data=top_crash_zones, x="current_zone_idx", y="crash_rate", hue="current_zone_idx", palette="Reds_r", order=top_crash_zones["current_zone_idx"], legend=False)
    plt.title("Top 50 'Struggle Zones' (Highest Crash Rate)")
    plt.xlabel("Zone ID")
    plt.ylabel("Crash Rate (%)")
    plt.xticks(rotation=90) # Rotate labels
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "zone_crash_rate.png"))
    plt.close()
    
    # Plot Avg Speed
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=zone_stats, x="current_zone_idx", y="avg_speed", marker="o", color="blue")
    plt.title("Average Speed per Zone")
    plt.xlabel("Zone ID")
    plt.ylabel("Speed (km/h)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "zone_avg_speed.png"))
    plt.close()
    
    return zone_stats

def visualize_track_zones(df, output_dir):
    print("--- Track Zone Map ---")
    
    if "x" not in df.columns or "z" not in df.columns:
        return
        
    # Sample points to avoid overcrowding
    sample_df = df.sample(n=min(50000, len(df)), random_state=42)
    
    plt.figure(figsize=(10, 10))
    # Use a discrete colormap for zones
    scatter = plt.scatter(sample_df["x"], sample_df["z"], c=sample_df["current_zone_idx"], cmap="tab20", s=5, alpha=0.7)
    plt.colorbar(scatter, label="Zone ID")
    plt.title("Track Map by Zone ID")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.axis("equal")
    plt.savefig(os.path.join(output_dir, "track_zone_map.png"))
    plt.close()

def main():
    output_dir = "analysis_results_section"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    df = extract_features(df)
    if df is None:
        return
    
    analyze_zone_performance(df, output_dir)
    visualize_track_zones(df, output_dir)
    
    print(f"Section analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
