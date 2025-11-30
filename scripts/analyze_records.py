
import os
import glob
import matplotlib.pyplot as plt
import datetime
import re

def analyze_records(best_runs_dir, target_time=49.47):
    print(f"Analyzing records in {best_runs_dir}...")
    
    records = []
    
    # Pattern to match hock_XXXXX
    pattern = re.compile(r"hock_(\d+)")
    
    subdirs = glob.glob(os.path.join(best_runs_dir, "hock_*"))
    
    for d in subdirs:
        match = pattern.search(os.path.basename(d))
        if match:
            time_ms = int(match.group(1))
            time_sec = time_ms / 1000.0
            
            # Get creation time
            creation_time = os.path.getctime(d)
            dt = datetime.datetime.fromtimestamp(creation_time)
            
            records.append((dt, time_sec))
            
    if not records:
        print("No records found.")
        return

    # Sort by creation time
    records.sort(key=lambda x: x[0])
    
    dates = [r[0] for r in records]
    times = [r[1] for r in records]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(dates, times, marker='o', linestyle='-', color='b', label='AI Best Time')
    
    # Target Line
    plt.axhline(y=target_time, color='r', linestyle='--', label=f'World Record ({target_time}s)')
    
    plt.title(f"The Chase: AI vs World Record ({target_time}s)")
    plt.xlabel("Time")
    plt.ylabel("Lap Time (s)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = "record_progression.png"
    plt.savefig(output_path)
    print(f"✅ Plot saved to {output_path}")
    
    # Stats
    current_best = min(times)
    gap = current_best - target_time
    print(f"\nCurrent Best: {current_best:.2f}s")
    print(f"Target:       {target_time:.2f}s")
    print(f"Gap:          {gap:.2f}s")

if __name__ == "__main__":
    analyze_records("save/felix_test_training/best_runs")
