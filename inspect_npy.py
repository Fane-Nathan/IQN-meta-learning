import numpy as np
import sys

try:
    data = np.load("c:/Users/felix/Documents/linesight/config_files/ESL-Hockolicious_0.5m_cl2.npy")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print("First 5 rows:")
    print(data[:5])
except Exception as e:
    print(f"Error: {e}")
