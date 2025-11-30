
import joblib
import sys
import numpy as np

def check_joblib(path):
    print(f"Loading {path}...")
    try:
        data = joblib.load(path)
        print("Keys:", data.keys())
        
        if 'x' in data:
            print(f"'x' found! Length: {len(data['x'])}")
            print(f"First 5 x: {data['x'][:5]}")
        else:
            print("'x' NOT found.")
            
        if 'state_float' in data:
             print(f"'state_float' found! Length: {len(data['state_float'])}")
             # Check if coords are in state_float?
             # Usually state_float is [x, y, z, ...] or similar
             print(f"First state: {data['state_float'][0]}")
             
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "save/felix_test_training/best_runs/hock_57060/rollout_data_hock_57060.joblib"
    check_joblib(path)
