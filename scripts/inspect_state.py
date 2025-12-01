import json
import os
import sys

def inspect_zone_state(zone_id):
    state_path = os.path.join("states", "mastery_state.json")
    
    if not os.path.exists(state_path):
        print(f"Error: {state_path} not found.")
        return

    try:
        with open(state_path, "r") as f:
            state = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {state_path}")
        return

    # Check for zone ID as string or int
    zone_data = state.get(str(zone_id)) or state.get(int(zone_id))
    
    if zone_data:
        print(f"--- Zone {zone_id} State ---")
        # Print all keys to see what's available
        for key, value in zone_data.items():
            if key == "state_float":
                print(f"{key}: [Array of length {len(value)}]")
            else:
                print(f"{key}: {value}")
    else:
        print(f"Zone {zone_id} not found in mastery state.")
        print(f"Available zones: {sorted([int(k) for k in state.keys()])[:10]} ...")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        zone = sys.argv[1]
    else:
        zone = 453
    inspect_zone_state(zone)
