
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
    
    # Steering/Gas (from input_w, assuming it's Gas based on previous findings)
    # But wait, input_w is Gas. Where is Steering?
    # Steering is in Previous Actions (Indices 1-20).
    # Index 1: Accel, 2: Brake, 3: Left, 4: Right (t-1)
    # Let's use the most recent previous action for "Steering" approximation
    # Index 3 (Left) and 4 (Right).
    # If Left > 0.5 -> -1, Right > 0.5 -> +1?
    # Actually, let's just use the raw values.
    prev_steer_left = state_floats[:, 3]
    prev_steer_right = state_floats[:, 4]
    steering = prev_steer_right - prev_steer_left # +1 for Right, -1 for Left
    
    # Gear (Index 21)
    gear = state_floats[:, 21]
    
    # RPM (Not explicit, but maybe correlated with speed/gear)
    
    # Add to DF
    df["speed_kmh"] = speed
    df["steering"] = steering
    df["gear"] = gear
    
    return df, state_floats

def crash_forensics(df, output_dir):
    print("--- Crash Forensics ---")
    # Define "Crash" or "Failure"
    # Low reward? Or specific event?
    # Let's assume a crash is when speed drops suddenly or reward is very low.
    # But reward is dense.
    # Let's look for "Speed < 5 km/h" while "Gas > 0.5" (Stuck?)
    # Or just low speed in general.
    
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
        
        # Analyze preceding steps?
        # Requires sequential data.
        # If df index is sequential...
        # Let's look at the "Pre-Crash" state (10 steps before).
        # This is hard if data is shuffled or chunked.
        # We'll skip sequential analysis for now and focus on state correlation.
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=crashes, x="x", y="z", hue="gear", palette="viridis", s=10)
        plt.title("Map Location of Crashes")
        plt.savefig(os.path.join(output_dir, "crash_map.png"))
        plt.close()

def driving_style_clustering(df, output_dir):
    print("--- Driving Style Clustering (K-Means) ---")
    # Features for clustering: Speed, Steering, Gear
    features = df[["speed_kmh", "steering", "gear"]].copy()
    
    # Normalize
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(scaled_features)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="speed_kmh", y="steering", hue="cluster", palette="deep", s=5, alpha=0.5)
    plt.title("Driving Styles: Speed vs Steering")
    plt.savefig(os.path.join(output_dir, "driving_styles_speed_steering.png"))
    plt.close()
    
    # Interpret clusters
    print("Cluster Centers:")
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    center_df = pd.DataFrame(centers, columns=["speed_kmh", "steering", "gear"])
    print(center_df)
    center_df.to_csv(os.path.join(output_dir, "cluster_centers.csv"))

def state_space_mapping(df, state_floats, output_dir):
    print("--- State Space Mapping (PCA) ---")
    # Use a subset if data is too large
    if len(df) > 10000:
        indices = np.random.choice(len(df), 10000, replace=False)
        subset_floats = state_floats[indices]
        subset_df = df.iloc[indices]
    else:
        subset_floats = state_floats
        subset_df = df
        
    # PCA
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
    
    # t-SNE (Optional, slow)
    # print("Running t-SNE (this may take a while)...")
    # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    # tsne_results = tsne.fit_transform(subset_floats)
    # plt.figure(figsize=(10, 8))
    # plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=subset_df["speed_kmh"], cmap="plasma", s=2, alpha=0.6)
    # plt.title("t-SNE of State Space")
    # plt.savefig(os.path.join(output_dir, "tsne_state_space.png"))
    # plt.close()

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
    
    print(f"Advanced analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
