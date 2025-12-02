
import numpy as np
import sys

try:
    data = np.load("c:/Users/felix/Documents/linesight/maps/ESL-Hockolicious_WR_Ghost.npy")
    print(f"Shape: {data.shape}")
    
    # Check for NaNs
    if np.isnan(data).any():
        print("❌ ERROR: NaNs found in .npy file!")
    else:
        print("✅ No NaNs found.")

    # Check for Infs
    if np.isinf(data).any():
        print("❌ ERROR: Infs found in .npy file!")
    else:
        print("✅ No Infs found.")

    # Check for duplicates (distance == 0)
    diffs = np.linalg.norm(data[1:] - data[:-1], axis=1)
    min_dist = np.min(diffs)
    max_dist = np.max(diffs)
    print(f"Min distance between points: {min_dist}")
    print(f"Max distance between points: {max_dist}")

    if min_dist == 0:
        print("❌ ERROR: Duplicate consecutive points found (distance = 0)!")
        zero_indices = np.where(diffs == 0)[0]
        print(f"Indices with 0 distance: {zero_indices}")
    else:
        print("✅ No duplicate points found.")

except Exception as e:
    print(f"Error: {e}")
