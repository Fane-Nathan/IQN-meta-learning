
import joblib
from pathlib import Path

save_path = Path(r"c:\Users\felix\Documents\linesight\save\felix_test_training\accumulated_stats.joblib")

try:
    stats = joblib.load(save_path)
    print("Stats loaded successfully!")
    print(f"Keys: {stats.keys()}")
    print(f"NMG: {stats.get('cumul_number_frames_played', 'NOT FOUND')}")
    print(f"All-time Min MS: {stats.get('alltime_min_ms', 'NOT FOUND')}")
except Exception as e:
    print(f"Failed to load stats: {e}")
