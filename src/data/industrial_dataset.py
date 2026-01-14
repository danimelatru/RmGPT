import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import sys
from phmd import datasets

class IndustrialPretrainDataset(Dataset):
    """
    Dataset for RmGPT/MOMENT pretraining.
    Wrapper around the 'phmd' library to load industrial time-series datasets.
    Handles recursive dictionary structures and standardizes Channel dimensions.
    """

    def __init__(
        self,
        dataset_names=None,
        seq_len=2048,
        stride=2048,
        n_channels=2,   # <--- NUEVO: Forzamos 2 canales para todos
        download=True,
    ):
        if dataset_names is None:
            self.dataset_names = ["CWRU", "XJTU-SY", "KAUG17", "HSG18", "MFPT"]
        else:
            self.dataset_names = dataset_names

        self.seq_len = seq_len
        self.stride = stride
        self.n_channels = n_channels # Target channels
        self.samples = []

        print(f"[Dataset] Initializing with datasets: {self.dataset_names}")
        
        for name in self.dataset_names:
            try:
                print(f"\n--- Processing {name} ---")
                ds = datasets.Dataset(name)
                
                if download:
                    try:
                        ds.download()
                    except Exception as e:
                        print(f"   [Warning] Download check: {e}")

                # 1. Select Task
                task_name, task = self._select_task_robust(ds, name)
                if task is None: continue
                
                print(f"   -> Selected Task: '{task_name}'")
                
                # 2. Load Data payload
                try:
                    data_payload = task[0] 
                except Exception as e:
                    print(f"   [Error] Failed to load data subset: {e}")
                    continue

                # 3. Process data recursively
                self._process_recursive(data_payload, source_name=name)

            except Exception as e:
                print(f" [Error] Critical failure loading {name}: {e}")

        print(f"\n[Dataset] FINAL STATUS: {len(self.samples)} windows ready for training.")

    def _select_task_robust(self, ds, ds_name):
        common_keys = ['fault', 'rul', 'classification', 'regression', 'stage', 'Diagnosis', 'Prognosis']
        for key in common_keys:
            try:
                return key, ds[key]
            except:
                pass
        
        tasks_meta = ds.meta.get("tasks", {})
        if tasks_meta:
            first = list(tasks_meta.keys())[0]
            return first, ds[first]
        return None, None

    def _process_recursive(self, data, source_name):
        # Case 1: List/Tuple
        if isinstance(data, (list, tuple)):
            for i, item in enumerate(data):
                self._process_recursive(item, f"{source_name}_{i}")
        # Case 2: Dictionary
        elif isinstance(data, dict):
            for key, item in data.items():
                self._process_recursive(item, f"{source_name}_{key}")
        # Case 3: DataFrame/Array (Actual Data)
        elif isinstance(data, (pd.DataFrame, np.ndarray)):
            self._extract_windows_from_structure(data, source_name)

    def _extract_windows_from_structure(self, data, source_name):
        signal = None
        
        # --- DATAFRAME HANDLING ---
        if isinstance(data, pd.DataFrame):
            exclude_keywords = ['unit', 'bearing', 'fault', 'label', 'target', 'class', 'id', 'setting', 'cycle', 'rul']
            valid_cols = []
            for col in data.columns:
                col_lower = str(col).lower()
                if any(k in col_lower for k in exclude_keywords): continue
                if pd.api.types.is_numeric_dtype(data[col]):
                    valid_cols.append(col)
            
            if valid_cols:
                signal = data[valid_cols].values.T # [Channels, Time]
            else:
                return 

        # --- NUMPY HANDLING ---
        elif isinstance(data, np.ndarray):
            signal = data
            if signal.ndim == 2 and signal.shape[0] > signal.shape[1]:
                signal = signal.T 

        # --- WINDOW SLICING ---
        if signal is not None:
            signal = np.nan_to_num(signal.astype(np.float32))
            
            C, T = signal.shape
            if T < self.seq_len: return
            
            count = 0
            for start in range(0, T - self.seq_len + 1, self.stride):
                window = signal[:, start : start + self.seq_len]
                self.samples.append(window)
                count += 1
            
            if count > 0:
                print(f"   -> Loaded {count} windows from {source_name} (Original Channels: {C})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # x shape is [Original_Channels, seq_len]
        x = self.samples[idx]
        
        # --- CHANNEL PADDING/TRUNCATION ---
        # We need exactly self.n_channels (e.g., 2)
        current_C = x.shape[0]
        
        # Create a container of zeros [Target_C, seq_len]
        output_tensor = torch.zeros((self.n_channels, self.seq_len), dtype=torch.float32)
        
        # Determine how many channels to copy
        # If dataset has 1, copy 1. If dataset has 3, copy 2 (truncate).
        copy_c = min(current_C, self.n_channels)
        
        # Copy data into the container
        # Note: We must convert numpy to torch here
        output_tensor[:copy_c, :] = torch.from_numpy(x[:copy_c, :])
        
        # Mask is all 1s (observed)
        input_mask = torch.ones(self.seq_len, dtype=torch.long)

        return {
            "timeseries": output_tensor, # Fixed shape [2, 2048]
            "input_mask": input_mask,
        }