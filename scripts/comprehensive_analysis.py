
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
    
    # Steering (Approx from previous actions)
    prev_steer_left = state_floats[:, 3]
    prev_steer_right = state_floats[:, 4]
    steering = prev_steer_right - prev_steer_left
    
    df["speed_kmh"] = speed
    df["steering"] = steering
    
    # Ensure x, z, epsilon are present
    if "x" not in df.columns or "z" not in df.columns:
        print("Extracting x, z from state_float (assuming indices match)...")
        # Need to know indices for x, z if not in columns.
        # Based on previous knowledge, x, y, z might be in state_float?
        # But inspect_columns showed 'x', 'y', 'z' ARE in columns.
        pass
        
    return df

def analyze_spatial_evolution(df, output_dir):
    print("--- Spatial Evolution (Heatmaps) ---")
    
    # Bin Epsilon into "High" (Early), "Medium", "Low" (Late)
    # Epsilon usually decays.
    # Let's look at quantiles or fixed ranges.
    
    if "epsilon" not in df.columns:
        print("Epsilon column not found. Skipping spatial evolution by epsilon.")
        return

    eps_min = df["epsilon"].min()
    eps_max = df["epsilon"].max()
    print(f"Epsilon Range: {eps_min:.4f} - {eps_max:.4f}")
    
    # Create 3 bins
    df["epsilon_bin"] = pd.cut(df["epsilon"], bins=3, labels=["Low (Late)", "Medium", "High (Early)"])
    
    # Plot Heatmaps for each bin
    g = sns.FacetGrid(df, col="epsilon_bin", col_wrap=3, height=5, sharex=True, sharey=True)
    g.map_dataframe(sns.histplot, x="x", y="z", bins=50, pthresh=0.05, cmap="inferno")
    g.set_titles("{col_name}")
    g.fig.suptitle("Evolution of Racing Line (Position Heatmap)", y=1.05)
    plt.savefig(os.path.join(output_dir, "spatial_evolution_heatmap.png"))
    plt.close()
    
    # Scatter plot version (better for seeing the line)
    g = sns.FacetGrid(df, col="epsilon_bin", col_wrap=3, height=5, sharex=True, sharey=True)
    g.map_dataframe(sns.scatterplot, x="x", y="z", s=1, alpha=0.1, color="blue")
    g.set_titles("{col_name}")
    g.fig.suptitle("Evolution of Racing Line (Scatter)", y=1.05)
    plt.savefig(os.path.join(output_dir, "spatial_evolution_scatter.png"))
    plt.close()

def analyze_performance_vs_hyperparams(df, output_dir):
    print("--- Performance vs Hyperparameters ---")
    
    if "epsilon" not in df.columns:
        return

    # Aggregate by Epsilon (rounded) to reduce noise
    df["epsilon_rounded"] = df["epsilon"].round(3)
    
    agg_df = df.groupby("epsilon_rounded")[["speed_kmh"]].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=agg_df, x="epsilon_rounded", y="speed_kmh")
    plt.gca().invert_xaxis() # High epsilon (left) -> Low epsilon (right) = Time
    plt.title("Average Speed vs Epsilon (Time ->)")
    plt.xlabel("Epsilon (Exploration)")
    plt.ylabel("Average Speed (km/h)")
    plt.savefig(os.path.join(output_dir, "speed_vs_epsilon.png"))
    plt.close()
    
    # If we had Reward, we would plot that too.
    # Assuming 'reward' column might exist?
    # inspect_columns didn't show 'reward', but maybe 'step_reward'?
    # It showed 'q_values'.
    
    # Let's look for 'action_was_greedy' correlation
    if "action_was_greedy" in df.columns:
        greedy_speed = df[df["action_was_greedy"] == True]["speed_kmh"].mean()
        random_speed = df[df["action_was_greedy"] == False]["speed_kmh"].mean()
        print(f"Greedy Speed: {greedy_speed:.2f} km/h")
        print(f"Random Speed: {random_speed:.2f} km/h")
        
        plt.figure(figsize=(6, 6))
        sns.barplot(x=["Greedy", "Random"], y=[greedy_speed, random_speed])
        plt.title("Speed: Greedy vs Random Actions")
        plt.ylabel("Average Speed (km/h)")
        plt.savefig(os.path.join(output_dir, "speed_greedy_vs_random.png"))
        plt.close()

def analyze_state_distributions(df, output_dir):
    print("--- State Distribution Analysis ---")
    
    if "epsilon" not in df.columns:
        return
        
    # Use the same bins
    if "epsilon_bin" not in df.columns:
        df["epsilon_bin"] = pd.cut(df["epsilon"], bins=3, labels=["Low (Late)", "Medium", "High (Early)"])
        
    # Speed Distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="epsilon_bin", y="speed_kmh", palette="muted")
    plt.title("Speed Distribution Evolution")
    plt.xlabel("Training Stage (Epsilon)")
    plt.ylabel("Speed (km/h)")
    plt.savefig(os.path.join(output_dir, "speed_distribution_evolution.png"))
    plt.close()
    
    # Steering Distribution
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="epsilon_bin", y="steering", palette="muted")
    plt.title("Steering Distribution Evolution")
    plt.xlabel("Training Stage (Epsilon)")
    plt.ylabel("Steering Input (-1 to 1)")
    plt.savefig(os.path.join(output_dir, "steering_distribution_evolution.png"))
    plt.close()

def main():
    output_dir = "analysis_results_comprehensive"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    df = extract_features(df)
    
    analyze_spatial_evolution(df, output_dir)
    analyze_performance_vs_hyperparams(df, output_dir)
    analyze_state_distributions(df, output_dir)
    
    print(f"Comprehensive analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
