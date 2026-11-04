import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class FallMultiTaskDataset(Dataset):
    """
    Dataset class untuk membaca file pickle URFD.
    Output per item:
    1. src: Input sequence (Batch, 25, 34)
    2. mask: Input mask (Batch, 25, 34) -> 1 untuk valid, 0 untuk invalid
    3. label: Class label (mapped from -1,0,1 to 0,1,2)
    4. trg_forecast: Target forecast sequence (Batch, 25, 34)
    """
    def __init__(self, pkl_path):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"File dataset tidak ditemukan: {pkl_path}")
            
        print(f"Loading dataset: {pkl_path} ...")
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            
        self.src = data['src']                   
        self.src_mask = data['src_mask']         
        self.trg_forecast = data['trg_forecast'] 
        self.labels = data['trg_class']          
        
        print(f"Loaded {len(self.src)} samples.")
        
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, idx):
        # 1. Input Source
        src = torch.tensor(self.src[idx], dtype=torch.float32)
        
        # 2. Input Mask
        mask = torch.tensor(self.src_mask[idx], dtype=torch.float32)
        
        # 3. Target Forecast
        trg = torch.tensor(self.trg_forecast[idx], dtype=torch.float32)
        
        # 4. Label Mapping
        # Data asli: -1 (Null/Background), 0 (ADL), 1 (Fall)
        # Mapping ke index pytorch (0, 1, 2) agar kompatibel dengan CrossEntropy
        raw_label = self.labels[idx]
        label = torch.tensor(raw_label + 1, dtype=torch.long) 
        
        return src, mask, label, trg

def get_skeleton_edge_index():

    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),     # Head connections
        (5, 6),                             # Shoulder to shoulder
        (5, 11), (6, 12), (11, 12),         # Torso
        (5, 7), (7, 9),                     # Left arm
        (6, 8), (8, 10),                    # Right arm  
        (11, 13), (13, 15),                 # Left leg
        (12, 14), (14, 16),                 # Right leg
        (0, 5), (0, 6)                      # Head to torso
    ]
    
    # Buat bidirectional (bolak-balik) agar informasi mengalir dua arah
    source = [s for s, t in skeleton] + [t for s, t in skeleton]
    target = [t for s, t in skeleton] + [s for s, t in skeleton]
    
    # Tambahkan Self-loops (koneksi ke diri sendiri)
    num_nodes = 17
    source += list(range(num_nodes))
    target += list(range(num_nodes))
    
    edge_index = torch.tensor([source, target], dtype=torch.long)
    return edge_index

def create_dataloaders(train_path, val_path, test_path=None, batch_size=64, num_workers=2):
    """
    Helper function untuk membuat DataLoader.
    """
    loaders = {}
    
    # Train Loader
    if train_path and os.path.exists(train_path):
        train_ds = FallMultiTaskDataset(train_path)
        loaders['train'] = DataLoader(
            train_ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Val Loader
    if val_path and os.path.exists(val_path):
        val_ds = FallMultiTaskDataset(val_path)
        loaders['val'] = DataLoader(
            val_ds, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
    if test_path and os.path.exists(test_path):
        test_ds = FallMultiTaskDataset(test_path)
        loaders['test'] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ) 
        
    return loaders