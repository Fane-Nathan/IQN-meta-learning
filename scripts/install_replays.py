
import os
import shutil
import glob

def install_replays(source_dir, target_base_path):
    # Construct target path
    # Usually: Documents/TrackMania/Replays/My Replays
    target_dir = os.path.join(target_base_path, "Replays", "My Replays", "Linesight_Swarm")
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")
    
    # Find generated replays
    replays = glob.glob(os.path.join(source_dir, "*.Replay.Gbx"))
    
    if not replays:
        print(f"No replays found in {source_dir}")
        return

    print(f"Installing {len(replays)} replays to {target_dir}...")
    
    for r in replays:
        dest = os.path.join(target_dir, os.path.basename(r))
        shutil.copy(r, dest)
        # print(f" - Installed {os.path.basename(r)}")
        
    print("\n✅ Replays installed successfully!")
    print(f"Go to TrackMania > Editors > Edit Replay > My Replays > Linesight_Swarm")

if __name__ == "__main__":
    # Hardcoded path based on user_config.py
    # trackmania_base_path = Path(os.path.expanduser("~")) / "OneDrive" / "Documents" / "TrackMania"
    user_home = os.path.expanduser("~")
    tm_path = os.path.join(user_home, "OneDrive", "Documents", "TrackMania")
    
    # Check if path exists, if not try without OneDrive
    if not os.path.exists(tm_path):
        tm_path = os.path.join(user_home, "Documents", "TrackMania")
        
    install_replays("replays_to_render_out", tm_path)
