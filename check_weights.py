
import torch
from pathlib import Path
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from config_files import config_copy

# Mock config needed for network creation
config_copy.float_input_dim = 27 + 3 * 40 + 4 * 5 + 4 * 4 + 1
config_copy.float_hidden_dim = 256
config_copy.conv_head_output_dim = 5632
config_copy.dense_hidden_dimension = 1024
config_copy.iqn_embedding_dimension = 64
config_copy.iqn_n = 16
config_copy.iqn_k = 64
config_copy.iqn_kappa = 5e-3
config_copy.use_ddqn = True

save_path = Path(r"c:\Users\felix\Documents\linesight\save\felix_test_training\weights1.torch")

print(f"Checking weights at: {save_path}")

try:
    # Create network structure
    network, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    
    # Try loading
    state_dict = torch.load(f=save_path, weights_only=False)
    network.load_state_dict(state_dict)
    
    print("✅ Weights loaded successfully!")
    
    # Check a value to see if it's not all zeros (random check)
    first_param = next(network.parameters())
    print(f"First parameter mean: {first_param.mean().item()}")
    
except Exception as e:
    print(f"❌ Failed to load weights: {e}")
    import traceback
    traceback.print_exc()
