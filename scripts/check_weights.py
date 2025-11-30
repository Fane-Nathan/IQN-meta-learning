
import torch
import os

def check_weights(path):
    print(f"Checking {path}...")
    if not os.path.exists(path):
        print("File not found.")
        return

    try:
        state_dict = torch.load(path, map_location="cpu")
        is_corrupted = False
        for key, tensor in state_dict.items():
            if torch.isnan(tensor).any():
                print(f"❌ CORRUPTED: Found NaN in {key}")
                is_corrupted = True
                break
            if torch.isinf(tensor).any():
                print(f"❌ CORRUPTED: Found Inf in {key}")
                is_corrupted = True
                break
        
        if not is_corrupted:
            print("✅ SAFE: No NaN or Inf found in weights.")
        else:
            print("⚠️ Model is corrupted. You must delete it.")
            
    except Exception as e:
        print(f"Error loading weights: {e}")

if __name__ == "__main__":
    check_weights("save/felix_test_training/weights1.torch")
