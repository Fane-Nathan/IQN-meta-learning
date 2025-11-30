
import os
import glob
import shutil
import re
import sys

def setup_replays(best_runs_dir, output_dir="replays_to_render"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    else:
        # Clean up existing files
        for f in glob.glob(os.path.join(output_dir, "*")):
            os.remove(f)
        print(f"Cleaned directory: {output_dir}")

    # Find all .inputs files
    inputs_files = glob.glob(os.path.join(best_runs_dir, "**", "*.inputs"), recursive=True)
    
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
            
    # Sort and take top 5
    runs.sort(key=lambda x: x[0])
    top_runs = runs[:5]
    
    print(f"Preparing Top {len(top_runs)} Replays:")
    for time_ms, f in top_runs:
        dest = os.path.join(output_dir, os.path.basename(f))
        shutil.copy(f, dest)
        print(f" - Copied {os.path.basename(f)} ({time_ms/1000:.2f}s)")

    # Create Batch Script
    bat_content = f"""@echo off
echo Starting Replay Generation...
echo This will launch TrackMania and simulate the runs.
echo Please do not touch the mouse or keyboard!
python scripts/tools/video_stuff/inputs_to_gbx.py --inputs_dir "{output_dir}" --map_path "ESL-Hockolicious.Challenge.Gbx"
echo Done! Check {output_dir} for .Replay.Gbx files.
pause
"""
    with open("render_replays.bat", "w") as f:
        f.write(bat_content)
    
    print("\n✅ Setup Complete!")
    print("To generate the game replays:")
    print("1. Close TrackMania if it is open.")
    print("2. Run 'render_replays.bat' (double-click or from terminal).")
    print("3. Wait for the process to finish.")
    print("4. Import the resulting .gbx files into TrackMania to watch them!")

if __name__ == "__main__":
    setup_replays("save/felix_test_training/best_runs")
