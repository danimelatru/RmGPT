import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys
import os

# --- PATH ADJUSTMENT ---
# Get the current directory path (RmGPT folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path (the phmd repo root)
parent_dir = os.path.dirname(current_dir)
# Add parent directory to sys.path to allow 'import phmd'
sys.path.append(parent_dir)

# Import Dataset from the phmd library in the parent folder
from phmd import Dataset as PhmdDataset

class IndustrialDataset(Dataset):
    def __init__(self, dataset_names, patch_size=256, stride=256, mode='pretrain'):
        """
        Dataset class for Industrial Time Series (RmGPT Implementation).
        
        Args:
            dataset_names (list): List of dataset names to load (e.g., ['CWRU', 'XJTU']).
            patch_size (int): Length of the patch 'P' (RmGPT uses 256)[cite: 396].
            stride (int): Sliding window stride 'S' (RmGPT uses 256)[cite: 396].
            mode (str): 'pretrain' (returns patches only for masking) or 'finetune' (returns labels).
        """
        self.patch_size = patch_size
        self.stride = stride
        self.mode = mode
        self.samples = []
        self.labels = [] # Only used if mode='finetune'

        print(f"Loading datasets: {dataset_names}...")
        self._load_and_process(dataset_names)

    def _load_and_process(self, names):
        """Loads raw data from phmd, normalizes it, and applies patching."""
        for name in names:
            try:
                print(f"Processing {name}...")
                # Load from phmd
                ds = PhmdDataset(name)
                
                # We assume ds.data is a numpy array or list of signals
                # and ds.labels contains the fault labels.
                # Note: Adjust data access (e.g., ds.data_list) depending on your phmd version.
                raw_signals = ds.data 
                raw_labels = ds.labels

                for i, signal in enumerate(raw_signals):
                    # 1. Normalization (Standardization) [cite: 222, 234]
                    # Formula: (x - mean) / std
                    # This stabilizes the input distribution across different equipment.
                    mean = np.mean(signal)
                    std = np.std(signal) + 1e-6 # Add epsilon to avoid division by zero
                    norm_signal = (signal - mean) / std

                    # 2. Patching (Eq. 8 and 9 in the paper) [cite: 287, 292]
                    # Converts the 1D/Multi-channel signal into a sequence of patches
                    patches = self._create_patches(norm_signal)
                    
                    self.samples.append(patches)
                    if self.mode == 'finetune':
                        self.labels.append(raw_labels[i])
                
                print(f" -> {name} loaded successfully.")

            except Exception as e:
                print(f"[Warning] Could not load {name}: {e}")

    def _create_patches(self, signal):
        """
        Splits a long signal into fixed-length windows (patches).
        Implements Eq. 8: l_s = floor((L - P) / S) + 1 [cite: 287]
        """
        # Convert to tensor if input is numpy
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal).float()
            
        # torch.Tensor.unfold creates a sliding view: [Num_Patches, Patch_Size]
        # Dimension 0 is usually time for 1D signals.
        
        if signal.ndim == 1:
            # Univariate signal
            patches = signal.unfold(0, self.patch_size, self.stride)
            # Shape: [Num_Patches, Patch_Size]
        else:
            # Multivariate signal (Channels, Time)
            # Unfold along the last dimension (Time)
            patches = signal.unfold(-1, self.patch_size, self.stride)
            # Rearrange to [Num_Patches, Channels, Patch_Size]
            patches = patches.permute(1, 0, 2) 
            
        return patches

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            patches: Tensor of shape [L_s, M, P] or [L_s, P]
                     L_s: Sequence length (number of patches)
                     M: Number of channels
                     P: Patch size (256)
        """
        patches = self.samples[idx]
        
        if self.mode == 'pretrain':
            # For Self-Supervised Pretraining
            # Returns only patches. Masking will be handled in the Training Loop.
            return patches
        else:
            # For Downstream Tasks (Diagnosis/Prognosis)
            return patches, self.labels[idx]

# --- Testing Block ---
if __name__ == "__main__":
    # Hyperparameters from RmGPT paper (Table III) [cite: 406]
    # Patch Length (P) = 256, Stride (S) = 256
    
    # Quick test (Ensure you have at least one dataset like CWRU downloaded via phmd)
    # If CWRU is not available, try passing a dummy name or ensure download_datasets.py ran first.
    print("--- Testing IndustrialDataset ---")
    
    try:
        # Attempt to load CWRU as a test case
        dataset = IndustrialDataset(["CWRU"], patch_size=256, stride=256)
        
        if len(dataset) > 0:
            loader = DataLoader(dataset, batch_size=4, shuffle=True)
            print(f"Dataset created with {len(dataset)} samples.")
            
            # Verify batch dimensions
            for batch in loader:
                print("\nBatch Shape verification:")
                print(f"Tensor Shape: {batch.shape}") 
                # Expected: [Batch_Size, Num_Patches, (Channels), Patch_Size]
                # Example: [4, 8, 256] for univariate signals of length 2048
                break
        else:
            print("[Info] Dataset is empty. Check if the dataset files exist.")
            
    except Exception as e:
        print(f"[Error] Test failed: {e}")
        print("Make sure 'phmd' is installed or accessible via sys.path.")