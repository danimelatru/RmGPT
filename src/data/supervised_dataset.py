import torch
from torch.utils.data import Dataset
import numpy as np
import os

class SupervisedDataset(Dataset):
    """
    Dataset for Supervised Learning (Fine-Tuning).
    Returns signal windows and their corresponding fault labels.
    """
    def __init__(self, x_path, y_path):
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Data files not found. Run scripts/process_labeled_cwru.py first.")
            
        print(f"[SupervisedDataset] Loading {os.path.basename(x_path)}...")
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        print(f"[SupervisedDataset] Loaded {len(self.X)} samples.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        signal = self.X[idx]
        label = self.y[idx]
        
        # Convert to Tensor
        signal_tensor = torch.from_numpy(signal).float() # Shape [2, 2048]
        label_tensor = torch.tensor(label).long()        # Scalar ID
        
        # Input mask (All 1s = observed)
        input_mask = torch.ones(signal_tensor.shape[-1], dtype=torch.long)

        return {
            "timeseries": signal_tensor,
            "input_mask": input_mask,
            "labels": label_tensor
        }