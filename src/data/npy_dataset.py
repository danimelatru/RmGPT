import torch
from torch.utils.data import Dataset
import numpy as np
import os

class NpyDataset(Dataset):
    """
    Loads data from a SINGLE large .npy file containing all windows.
    Structure: [N_samples, Channels, Seq_Len]
    """
    def __init__(self, npy_file_path):
        """
        Args:
            npy_file_path: Path to the large .npy file (e.g., pretrain_full.npy)
        """
        if not os.path.exists(npy_file_path):
            raise FileNotFoundError(f"File not found: {npy_file_path}")
            
        print(f"[Dataset] Loading large array from {os.path.basename(npy_file_path)}...")
        
        # mmap_mode='r' allows us to access data without loading everything into RAM instantly,
        # though for 3GB it usually fits fine. It's safer for stability.
        self.data = np.load(npy_file_path, mmap_mode='r')
        
        print(f"[Dataset] Data shape: {self.data.shape}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # We assume data is already standardized to [2, 2048] float32 by process_data.py
        # np.array() forces a copy from the mmap to memory
        signal = np.array(self.data[idx])
        
        # Convert to Tensor
        signal_tensor = torch.from_numpy(signal)
        
        # Input mask (All 1s = observed)
        input_mask = torch.ones(signal_tensor.shape[-1], dtype=torch.long)
        
        return {
            "timeseries": signal_tensor,
            "input_mask": input_mask
        }