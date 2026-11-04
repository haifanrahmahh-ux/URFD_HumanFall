import pickle
import numpy as np
import os

file_paths = [
    "Dataset/urfd_mask_test_in25_out25_st5.pkl",
    "Dataset/urfd_mask_train_in25_out25_st5.pkl",
    "Dataset/urfd_mask_val_in25_out25_st5.pkl"
]

def analyze_array(name, arr):
    print(f"  🔑 Key: '{name}'")
    print(f"     Shape: {arr.shape}")
    print(f"     Dtype: {arr.dtype}")
    
    # Jika data numerik, tampilkan statistik
    if np.issubdtype(arr.dtype, np.number):
        print(f"     Range: Min={arr.min():.4f}, Max={arr.max():.4f}")
        
        # Jika ini adalah label/class (biasanya integer atau float bulat)
        # Atau jika namanya mengandung string 'class'
        if 'class' in name or (arr.ndim == 1 and np.unique(arr).size < 10):
            unique, counts = np.unique(arr, return_counts=True)
            dist = dict(zip(unique, counts))
            print(f"     Distribution: {dist}")
            
    print("-" * 30)

def inspect_detailed(data):
    # Loop semua item dalam dictionary
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                analyze_array(key, value)
            elif isinstance(value, list):
                print(f"  🔑 Key: '{key}'")
                print(f"     Type: List (Length: {len(value)})")
                print(f"     Sample[0]: {value[0]}")
                print("-" * 30)
            else:
                print(f"  🔑 Key: '{key}'")
                print(f"     Value: {value}")
                print("-" * 30)

print("="*60)
print("DETAILED PICKLE INSPECTION")
print("="*60)

for file_path in file_paths:
    print(f"\n📂 Reading: {file_path}")
    
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            inspect_detailed(data)
    else:
        print("❌ File not found.")

print("\nDONE.")