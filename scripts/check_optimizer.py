
import torch
import os

def check_optimizer(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        state_dict = torch.load(path, map_location="cpu")
        is_corrupted = False
        
        # Check param_groups
        for group in state_dict['param_groups']:
            for k, v in group.items():
                if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')): # Check for nan/inf
                     print(f"❌ CORRUPTED: Found NaN/Inf in param_group {k}")
                     is_corrupted = True

        # Check state
        for param_id, state in state_dict['state'].items():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if torch.isnan(v).any():
                        print(f"❌ CORRUPTED: Found NaN in state {param_id} key {k}")
                        is_corrupted = True
                        break
                    if torch.isinf(v).any():
                        print(f"❌ CORRUPTED: Found Inf in state {param_id} key {k}")
                        is_corrupted = True
                        break
        
        if not is_corrupted:
            print("✅ SAFE: No NaN or Inf found in optimizer state.")
        else:
            print("⚠️ Optimizer is corrupted. You must delete it.")
            
    except Exception as e:
        print(f"Error loading optimizer: {e}")

if __name__ == "__main__":
    check_optimizer("save/felix_test_training/optimizer1.torch")
