
import os
import glob
import shutil
import re
import sys
import numpy as np

def setup_swarm(good_runs_dir, output_dir="replays_to_render", num_samples=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        # Clean up existing files
        for f in glob.glob(os.path.join(output_dir, "*")):
            os.remove(f)
        print(f"Cleaned directory: {output_dir}")

    # Find all .inputs files
    inputs_files = glob.glob(os.path.join(good_runs_dir, "**", "*.inputs"), recursive=True)
    
    if not inputs_files:
        print("No .inputs files found.")
        return

    # Parse times
    runs = []
    pattern = re.compile(r"hock_(\d+)")
    
    for f in inputs_files:
        match = pattern.search(os.path.basename(f))
        if match:
            time_ms = int(match.group(1))
            runs.append((time_ms, f))
            
    # Sort by time (fastest first? No, we want evolution, so maybe by creation time?)
    # Actually, the user wants "training video", so we should show the PROGRESS.
    # But `good_runs` are just "good runs found so far".
    # The filenames usually have a timestamp or step count.
    # Let's check the filename format: `rollout_data_hock_300000_1128_084707_456035_explo_True.inputs`
    # It has a timestamp! `1128_084707` (Nov 28, 08:47:07).
    
    # Let's sort by modification time of the file to be safe, or parse the filename.
    # Sorting by modification time is easiest for "chronological training history".
    inputs_files.sort(key=os.path.getmtime)
    
    total_runs = len(inputs_files)
    print(f"Found {total_runs} total runs.")
    
    if total_runs == 0:
        return

    # Select N evenly spaced runs
    indices = np.linspace(0, total_runs - 1, num_samples, dtype=int)
    selected_files = [inputs_files[i] for i in indices]
    
    print(f"Selecting {len(selected_files)} runs to represent the training evolution:")
    
    for i, f in enumerate(selected_files):
        # Rename them to preserve order in the folder: 001_original_name.inputs
        new_name = f"{i:03d}_{os.path.basename(f)}"
        dest = os.path.join(output_dir, new_name)
        shutil.copy(f, dest)
        # print(f" - Copied {new_name}")

    print(f"\n✅ Prepared {len(selected_files)} runs in '{output_dir}'.")
    print("Run 'render_replays.bat' to generate the replays.")
    print("Then import ALL of them into TrackMania to see the 'Swarm' evolution!")

if __name__ == "__main__":
    setup_swarm("save/felix_test_training/good_runs")
