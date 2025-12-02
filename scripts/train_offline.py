
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import glob
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add parent directory to path to import config and modules
sys.path.append(str(Path(__file__).parent.parent))

from config_files import config_copy
from trackmania_rl.agents.iqn import make_untrained_iqn_network
from trackmania_rl import utilities

class OfflineDataset(Dataset):
    def __init__(self, parquet_files):
        print(f"Loading {len(parquet_files)} parquet files...")
        dfs = []
        for f in tqdm(parquet_files, desc="Loading Parquet"):
            try:
                df = pd.read_parquet(f)
                # Ensure state_float is numpy array (it should be)
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        self.data = pd.concat(dfs, ignore_index=True)
        print(f"Total samples: {len(self.data)}")
        
        # Pre-convert columns to numpy arrays for faster access
        self.state_floats = np.stack(self.data['state_float'].values)
        self.actions = self.data['actions'].values.astype(np.int64)
        
        # Dummy image: (1, 120, 160) - Normalized (0 centered? No, Inferer does (img-128)/128)
        # But Trainer passes raw img? No, Inferer passes normalized.
        # IQN_Network expects float inputs.
        # We will pass zeros.
        self.dummy_img = torch.zeros((1, 120, 160), dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state_float = torch.tensor(self.state_floats[idx], dtype=torch.float32)
        action = torch.tensor(self.actions[idx], dtype=torch.long)
        return self.dummy_img, state_float, action

def train_offline():
    # Configuration
    BATCH_SIZE = 512
    EPOCHS = 5
    LR = 3e-4 # Higher LR for supervised learning?
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_DIR = Path("save/felix_test_training")
    
    print(f"Training on {DEVICE}")

    # Load Data
    parquet_files = glob.glob("merged_rollouts_part_*.parquet")
    if not parquet_files:
        print("No parquet files found!")
        return

    dataset = OfflineDataset(parquet_files)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for Windows safety

    # Initialize Network
    # We use make_untrained_iqn_network to get the architecture
    online_network, _ = make_untrained_iqn_network(jit=False, is_inference=False)
    online_network.to(DEVICE)
    online_network.train()

    # Optimizer
    optimizer = torch.optim.RAdam(online_network.parameters(), lr=LR)
    
    # Loss Function: Cross Entropy (Behavior Cloning)
    # We want to maximize prob of expert action.
    # IQN outputs Q-values. We can treat Q-values as logits?
    # Or we can use MSE loss against a target Q-value (e.g. 1 for expert, 0 for others)?
    # CrossEntropy is better for classification.
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (img, state_float, action) in enumerate(pbar):
            img = img.to(DEVICE)
            state_float = state_float.to(DEVICE)
            action = action.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            # IQN returns (batch * num_quantiles, n_actions)
            # We need to reshape/mean to get (batch, n_actions)
            q_values, _ = online_network(img, state_float, num_quantiles=32)
            
            # Reshape: (batch, num_quantiles, n_actions)
            q_values = q_values.view(BATCH_SIZE, 32, -1) # Assuming batch size is constant? 
            # Wait, last batch might be smaller.
            current_batch_size = img.size(0)
            q_values = q_values.view(current_batch_size, 32, -1)
            
            # Mean over quantiles -> Expected Q-values (Logits)
            logits = q_values.mean(dim=1)

            loss = criterion(logits, action)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        print("Saving checkpoint...")
        utilities.save_checkpoint(SAVE_DIR, online_network, online_network, optimizer, torch.cuda.amp.GradScaler())
        print("Saved.")

if __name__ == "__main__":
    train_offline()
