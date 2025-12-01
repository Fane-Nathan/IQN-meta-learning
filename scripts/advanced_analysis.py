import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import glob
import os

def load_data(data_dir="."):
    print(f"Looking for parquet files in {data_dir}...")
    files = sorted(glob.glob(os.path.join(data_dir, "merged_rollouts_part_*.parquet")))
    
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
    
    # Steering (from Previous Actions)
    # Index 3 (Left) and 4 (Right).
    prev_steer_left = state_floats[:, 3]
    prev_steer_right = state_floats[:, 4]
    steering = prev_steer_right - prev_steer_left # +1 for Right, -1 for Left
    
    # Gear (Index 21)
    gear = state_floats[:, 21]
    
    # Gas (from input_w if available, else from state_float index 1)
    if "input_w" in df.columns:
        gas = pd.to_numeric(df["input_w"], errors='coerce').fillna(0).astype(float)
    else:
        # Fallback to state_float index 1 (Accel)
        gas = state_floats[:, 1]
    
    # Teacher Stats (if available)
    if "teacher_frontier" in df.columns:
        df["teacher_frontier"] = df["teacher_frontier"]
    if "teacher_spawn_zone" in df.columns:
        df["teacher_spawn_zone"] = df["teacher_spawn_zone"]

    # Add to DF
    df["speed_kmh"] = speed
    df["steering"] = steering
    df["gear"] = gear
    df["gas"] = gas
    
    return df, state_floats

def crash_forensics(df, output_dir):
    print("--- Crash Forensics ---")
    # Define "Crash" or "Failure"
    # Let's define "Crash" as Speed < 10 km/h (assuming race mode)
    crash_mask = df["speed_kmh"] < 10
    crashes = df[crash_mask]
    
    print(f"Identified {len(crashes)} 'Crash/Stuck' states (Speed < 10 km/h).")
    
    if len(crashes) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(crashes["gear"], bins=6, kde=False)
        plt.title("Gear Distribution during Crashes")
        plt.savefig(os.path.join(output_dir, "crash_gear_dist.png"))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=crashes, x="x", y="z", hue="gear", palette="viridis", s=10)
        plt.title("Map Location of Crashes")
        plt.savefig(os.path.join(output_dir, "crash_map.png"))
        plt.close()

def driving_style_clustering(df, output_dir):
    print("--- Driving Style Clustering (K-Means) ---")
    features = df[["speed_kmh", "steering", "gear"]].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled_features)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="speed_kmh", y="steering", hue="cluster", palette="deep", s=5, alpha=0.5)
    plt.title("Driving Styles: Speed vs Steering")
    plt.savefig(os.path.join(output_dir, "driving_styles_speed_steering.png"))
    plt.close()
    
    print("Cluster Centers:")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=["speed_kmh", "steering", "gear"])
    print(center_df)
    center_df.to_csv(os.path.join(output_dir, "cluster_centers.csv"))

def state_space_mapping(df, state_floats, output_dir):
    print("--- State Space Mapping (PCA) ---")
    if len(df) > 10000:
        indices = np.random.choice(len(df), 10000, replace=False)
        subset_floats = state_floats[indices]
        subset_df = df.iloc[indices]
    else:
        subset_floats = state_floats
        subset_df = df
        
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(subset_floats)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=subset_df["speed_kmh"], cmap="plasma", s=2, alpha=0.6)
    plt.colorbar(label="Speed (km/h)")
    plt.title("PCA of State Space (Colored by Speed)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.savefig(os.path.join(output_dir, "pca_state_space.png"))
    plt.close()

def analyze_learning_v1_5(df, output_dir):
    print("--- v1.5 Learning Analysis (Enhanced) ---")
    
    plt.figure(figsize=(10, 6))
    if "teacher_frontier" in df.columns and df["teacher_frontier"].nunique() > 1:
        sns.histplot(data=df, x="speed_kmh", hue="teacher_frontier", palette="viridis", element="step", bins=50)
        plt.title("Speed Distribution by Curriculum Frontier")
    else:
        plt.hist(df["speed_kmh"], bins=50, color='skyblue', edgecolor='black')
        plt.title("Speed Distribution (Overall)")
    plt.xlabel("Speed (km/h)")
    plt.savefig(os.path.join(output_dir, "speed_distribution_v1.5.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(df["gas"], bins=20, color='salmon', edgecolor='black')
    plt.title("Gas/Acceleration Input Distribution")
    plt.xlabel("Gas Input")
    plt.savefig(os.path.join(output_dir, "gas_distribution_v1.5.png"))
    plt.close()

    plt.figure(figsize=(12, 6))
    plot_df = df.iloc[::10] if len(df) > 10000 else df
    
    if "teacher_frontier" in plot_df.columns:
        sns.scatterplot(data=plot_df.reset_index(), x="index", y="speed_kmh", hue="teacher_frontier", palette="viridis", s=2, alpha=0.5)
        plt.title("Learning Progress: Speed over Time (Colored by Frontier)")
    else:
        plt.plot(plot_df.index, plot_df["speed_kmh"], alpha=0.5, linewidth=0.5)
        plt.title("Learning Progress: Speed over Time")
        
    plt.xlabel("Training Step")
    plt.ylabel("Speed (km/h)")
    plt.savefig(os.path.join(output_dir, "learning_progress_v1.5.png"))
    plt.close()

def analyze_teacher_impact(df, output_dir):
    print("--- Teacher Impact Analysis ---")
    if "teacher_frontier" not in df.columns:
        print("Teacher stats not found in data.")
        return

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="teacher_frontier", y="speed_kmh", alpha=0.1, s=2)
    sns.lineplot(data=df, x="teacher_frontier", y="speed_kmh", color="red", label="Mean Speed")
    plt.title("Agent Speed vs Curriculum Frontier")
    plt.xlabel("Teacher Frontier (Zone)")
    plt.ylabel("Speed (km/h)")
    plt.savefig(os.path.join(output_dir, "speed_vs_frontier.png"))
    plt.close()

    cols = ["speed_kmh", "steering", "gear", "gas", "teacher_frontier", "teacher_spawn_zone"]
    cols = [c for c in cols if c in df.columns]
    
    corr = df[cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation: Performance vs Teacher Stats")
    plt.savefig(os.path.join(output_dir, "teacher_correlation.png"))
    plt.close()

def main():
    output_dir = "analysis_results_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    df, state_floats = extract_features(df)
    
    crash_forensics(df, output_dir)
    driving_style_clustering(df, output_dir)
    state_space_mapping(df, state_floats, output_dir)
    analyze_learning_v1_5(df, output_dir)
    analyze_teacher_impact(df, output_dir)
    
    print(f"Advanced analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
