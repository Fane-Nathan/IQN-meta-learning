
import pandas as pd
import numpy as np

def inspect_structure():
    try:
        # Regenerate a small chunk if needed, but I think I deleted them.
        # So I will assume I need to run merge first or just run this script 
        # which will fail if no parquet.
        # Actually, I'll assume the user runs merge first or I'll run it.
        
        # Let's just create a dummy parquet from the merge script if it doesn't exist
        # But I can't call merge script from here easily.
        # I will just rely on the tool execution order.
        
        df = pd.read_parquet("merged_rollouts_part_000.parquet")
        
        print("--- Structure Inspection ---")
        
        # 1. state_float
        state_float = df["state_float"].iloc[0]
        print(f"state_float length: {len(state_float)}")
        print(f"state_float sample (first 50): {state_float[:50]}")
        
        # 2. car_gear_and_wheels
        # This column might be flattened or list of lists in parquet
        car_gear = df["car_gear_and_wheels"].iloc[0]
        print(f"\ncar_gear_and_wheels type: {type(car_gear)}")
        if isinstance(car_gear, np.ndarray) or isinstance(car_gear, list):
            print(f"car_gear_and_wheels length: {len(car_gear)}")
            print(f"car_gear_and_wheels values: {car_gear}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_structure()
