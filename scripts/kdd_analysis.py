import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import glob
import os
from collections import Counter

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
    state_floats = np.vstack(df["state_float"].values)
    
    # Speed
    vx = state_floats[:, 58]
    vy = state_floats[:, 59]
    vz = state_floats[:, 60]
    speed = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6 # km/h
    
    # Steering (Index 3=Left, 4=Right)
    prev_steer_left = state_floats[:, 3]
    prev_steer_right = state_floats[:, 4]
    steering = prev_steer_right - prev_steer_left
    
    # Gear & RPM
    gear = state_floats[:, 21]
    rpm = state_floats[:, 22] # Assuming RPM is at 22 based on previous context or similar
    
    # Position
    x = df["x"] if "x" in df.columns else np.zeros(len(df))
    y = df["y"] if "y" in df.columns else np.zeros(len(df))
    z = df["z"] if "z" in df.columns else np.zeros(len(df))
    
    # Zone
    zone = df["current_zone_idx"] if "current_zone_idx" in df.columns else np.zeros(len(df))

    features = pd.DataFrame({
        "speed": speed,
        "steering": steering,
        "gear": gear,
        "rpm": rpm,
        "x": x,
        "y": y,
        "z": z,
        "zone": zone
    })
    
    return features

def train_crash_predictor(features, output_dir):
    print("--- Training Crash Predictor (Decision Tree) ---")
    
    # Define Crash: Speed < 5 km/h AND Zone > 10 (ignore start)
    features["is_crash"] = (features["speed"] < 5) & (features["zone"] > 10)
    
    # Balance dataset (undersample non-crashes)
    crashes = features[features["is_crash"] == True]
    non_crashes = features[features["is_crash"] == False]
    
    if len(crashes) < 10:
        print("Not enough crashes to train model.")
        return

    n_samples = min(len(crashes), len(non_crashes))
    balanced_df = pd.concat([
        crashes.sample(n_samples),
        non_crashes.sample(n_samples)
    ])
    
    X = balanced_df[["speed", "steering", "gear", "rpm", "zone"]]
    y = balanced_df["is_crash"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    print("Model Accuracy:", clf.score(X_test, y_test))
    
    rules = export_text(clf, feature_names=["speed", "steering", "gear", "rpm", "zone"])
    print("\nCrash Rules (Decision Tree):")
    print(rules)
    
    with open(os.path.join(output_dir, "crash_rules.txt"), "w") as f:
        f.write(rules)

def mine_action_sequences(df, features, output_dir):
    print("--- Mining Action Sequences ---")
    
    # Discretize Steering
    def discretize_steer(val):
        if val < -0.5: return "Hard Left"
        if val < -0.1: return "Soft Left"
        if val > 0.5: return "Hard Right"
        if val > 0.1: return "Soft Right"
        return "Straight"
        
    features["action_label"] = features["steering"].apply(discretize_steer)
    
    # Identify Crash Events (transitions from Non-Crash to Crash)
    features["is_crash"] = (features["speed"] < 5) & (features["zone"] > 10)
    crash_indices = features.index[features["is_crash"] & ~features["is_crash"].shift(1).fillna(False)].tolist()
    
    sequences = []
    window_size = 5
    
    for idx in crash_indices:
        if idx >= window_size:
            seq = features["action_label"].iloc[idx-window_size:idx].tolist()
            sequences.append(" -> ".join(seq))
            
    if not sequences:
        print("No crash sequences found.")
        return
        
    common_sequences = Counter(sequences).most_common(10)
    
    print("\nTop 10 Crash Sequences:")
    with open(os.path.join(output_dir, "crash_sequences.txt"), "w") as f:
        f.write("Top 10 Crash Sequences:\n")
        for seq, count in common_sequences:
            print(f"{count}x: {seq}")
            f.write(f"{count}x: {seq}\n")

def main():
    output_dir = "analysis_results_kdd"
    os.makedirs(output_dir, exist_ok=True)
    
    df = load_data()
    if df is None:
        return
        
    features = extract_features(df)
    
    train_crash_predictor(features, output_dir)
    mine_action_sequences(df, features, output_dir)
    
    print(f"KDD analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
