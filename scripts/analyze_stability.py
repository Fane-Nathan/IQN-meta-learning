
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def load_metrics(data_dir="save/felix_test_training"):
    # Look for training_metrics.csv
    csv_path = os.path.join(data_dir, "training_metrics.csv")
    
    if not os.path.exists(csv_path):
        print(f"No 'training_metrics.csv' found in {data_dir}.")
        print("This analysis only works for runs with the new logging system.")
        return None
        
    print(f"Loading metrics from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Parse timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Clean 'loss' column (remove 'tensor(' and ')')
        if df["loss"].dtype == object:
            df["loss"] = df["loss"].astype(str).str.replace("tensor(", "", regex=False).str.replace(")", "", regex=False)
            df["loss"] = pd.to_numeric(df["loss"], errors="coerce")
            
        return df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def analyze_stability(df, output_dir):
    print("--- Analyzing Stability ---")
    
    # 1. Loss vs Step
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="step", y="loss", alpha=0.5)
    # Add moving average
    df["loss_ma"] = df["loss"].rolling(window=100).mean()
    sns.lineplot(data=df, x="step", y="loss_ma", color="red", label="Moving Avg (100)")
    plt.title("Training Loss over Time")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.yscale("log") # Loss often spans orders of magnitude
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(os.path.join(output_dir, "stability_loss.png"))
    plt.close()
    
    # 2. Gradient Norm vs Step
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="step", y="grad_norm", color="orange", alpha=0.5)
    plt.title("Gradient Norm over Time (Explosion Detector)")
    plt.xlabel("Step")
    plt.ylabel("Gradient Norm")
    plt.axhline(y=10, color="red", linestyle="--", label="Warning Threshold")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "stability_grad_norm.png"))
    plt.close()
    
    # 3. Learning Rate & Epsilon
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Epsilon', color=color)
    ax1.plot(df['step'], df['epsilon'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:green'
    ax2.set_ylabel('Learning Rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['step'], df['learning_rate'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Hyperparameter Schedule")
    plt.savefig(os.path.join(output_dir, "stability_hyperparams.png"))
    plt.close()

def main():
    output_dir = "analysis_results_stability"
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find the file in the current save directory
    # Assuming standard path
    df = load_metrics("save/felix_test_training")
    
    if df is None:
        # Try current dir as fallback
        df = load_metrics(".")
        
    if df is None:
        return
        
    analyze_stability(df, output_dir)
    print(f"Stability analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
