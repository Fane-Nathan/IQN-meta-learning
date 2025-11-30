
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_data(data_dir="."):
    print(f"Looking for parquet files in {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "merged_rollouts_part_*.parquet"))
    
    if not files:
        if os.path.exists(os.path.join(data_dir, "merged_rollouts.parquet")):
            files = [os.path.join(data_dir, "merged_rollouts.parquet")]
        else:
            print("No merged_rollouts parquet files found. Please run merge_rollouts.py first.")
            return None

    # Load a subset to save memory/time for analysis
    # We don't need ALL data for RF training, a representative sample is fine.
    # Let's load the last 5 files (most recent behavior)
    files.sort(key=os.path.getmtime)
    recent_files = files[-5:]
    
    print(f"Loading {len(recent_files)} recent files for Root Cause Analysis...")
    dfs = []
    for f in recent_files:
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

def create_dataset(df):
    print("Creating dataset for Crash Prediction...")
    
    # 1. Define Target: Crash
    # Crash = Speed < 10 km/h (approx 2.7 m/s)
    # But we want to predict the crash *before* it happens.
    # So we look at frames where speed is high, but N frames later it is low.
    # Or simpler: Just classify "Crash Frames" vs "Safe Frames" based on current state?
    # The user said: "Grabs 20 frames before a crash".
    # That implies a time-series approach or labeling the "pre-crash" state as "Dangerous".
    
    # Let's extract features first.
    state_floats = np.vstack(df["state_float"].values)
    
    # Features from state_float (based on typical TM schema)
    # 0-2: Pos, 3-5: Vel, 6-8: Rot, 9-11: AngVel
    # We need to map these to meaningful names if possible.
    # Assuming standard Linesight schema:
    # 58, 59, 60 is Velocity? (Based on section_analysis)
    
    vx = state_floats[:, 58]
    vy = state_floats[:, 59]
    vz = state_floats[:, 60]
    speed = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6
    
    df["speed_kmh"] = speed
    
    # Define "Crash Event"
    # A crash is when speed drops below 10 km/h SUDDENLY?
    # Or just being stopped.
    # Let's define "Crash State" as Speed < 5.
    df["is_crashed"] = (df["speed_kmh"] < 5).astype(int)
    
    # We want to predict "Will Crash".
    # So we shift the target back by N frames (e.g., 10 frames = 1 second at 100ms/step).
    # If is_crashed is True at T, then T-10 was "Pre-Crash".
    
    # Note: This requires contiguous data. Our df is concatenated chunks.
    # Ideally we do this per-run. But for a rough RF, we can just shift and ignore the boundaries (small noise).
    
    N_FRAMES = 10
    df["will_crash"] = df["is_crashed"].shift(-N_FRAMES).fillna(0).astype(int)
    
    # Remove the actual crash frames from training (we want to predict FROM safe state)
    # So keep only rows where is_crashed is False, but will_crash is True/False.
    clean_df = df[df["is_crashed"] == 0].copy()
    
    # Balance the dataset? Crashes are rare.
    crash_samples = clean_df[clean_df["will_crash"] == 1]
    safe_samples = clean_df[clean_df["will_crash"] == 0]
    
    if len(crash_samples) == 0:
        print("No crash events found in recent data!")
        return None, None, None
        
    # Downsample safe
    safe_samples = safe_samples.sample(n=len(crash_samples) * 2, random_state=42) # 1:2 ratio
    
    data = pd.concat([crash_samples, safe_samples])
    
    # Feature Selection
    # We use the state_float components as features.
    # Let's take a subset of likely important ones.
    # Or just all of them? 
    # Let's take the first 61 (0-60).
    X = np.vstack(data["state_float"].values)[:, :61]
    y = data["will_crash"].values
    
    feature_names = [f"state_{i}" for i in range(61)]
    # Rename known ones
    feature_names[58] = "vel_x"
    feature_names[59] = "vel_y"
    feature_names[60] = "vel_z"
    # Maybe 0,1,2 are pos
    feature_names[0] = "pos_x"
    feature_names[1] = "pos_y"
    feature_names[2] = "pos_z"
    
    return X, y, feature_names

def train_crash_predictor(X, y, feature_names, output_dir):
    print("Training Random Forest...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Feature Importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 20
    top_features = []
    
    print("Top Predictors of Crashes:")
    for i in range(top_n):
        idx = indices[i]
        name = feature_names[idx]
        score = importances[idx]
        top_features.append((name, score))
        print(f"{i+1}. {name}: {score:.4f}")
        
    # Plot
    plt.figure(figsize=(12, 8))
    names = [x[0] for x in top_features]
    scores = [x[1] for x in top_features]
    sns.barplot(x=scores, y=names, palette="viridis")
    plt.title("Feature Importance: What predicts a crash 1s ahead?")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "crash_feature_importance.png"))
    plt.close()
    
    return top_features

def main():
    output_dir = "analysis_results_root_cause"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    X, y, feature_names = create_dataset(df)
    if X is None:
        return
        
    train_crash_predictor(X, y, feature_names, output_dir)
    print(f"Root Cause Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
