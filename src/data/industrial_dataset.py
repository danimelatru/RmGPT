import torch
from torch.utils.data import Dataset
import numpy as np
from phmd import datasets


class IndustrialPretrainDataset(Dataset):
    """
    Dataset for RmGPT/MOMENT pretraining.
    Returns:
        timeseries : FloatTensor [C, seq_len]
        input_mask : LongTensor  [seq_len] (1=observed, 0=padding)
    """

    def __init__(
        self,
        dataset_names=None,
        seq_len=2048,   # Default RmGPT: 8 patches * 256
        stride=2048,    # Default: Non-overlapping windows
        download=True,
    ):
        # Default substitutes if None provided
        if dataset_names is None:
            self.dataset_names = ["CWRU", "XJTU-SY", "KAUG17", "HSG18", "MFPT"]
        else:
            self.dataset_names = dataset_names

        self.seq_len = seq_len
        self.stride = stride
        self.samples = []

        print(f"[Dataset] Loading: {self.dataset_names}")
        
        for name in self.dataset_names:
            try:
                ds = datasets.Dataset(name)
                if download:
                    try:
                        ds.download()
                    except Exception:
                        pass # phmd handles checks internally

                task = self._select_task(ds)
                if task is None: continue
                
                # Take the first subset/fold available
                data = task[0]
                X, _ = self._unpack(data)
                
                if X is not None:
                    self._extract_windows(X)
                    print(f" -> Loaded {name}: {len(self.samples)} accumulated windows")
            except Exception as e:
                print(f" [Error] Failed to load {name}: {e}")

    def _select_task(self, ds):
        tasks = ds.meta.get("tasks", {})
        # Prioritize Diagnosis/Prognosis tasks
        for key in tasks:
            if any(k in key.lower() for k in ["fault", "diagnosis", "rul", "prognosis"]):
                return ds[key]
        # Fallback to first available task
        if tasks:
            return ds[list(tasks.keys())[0]]
        return None

    def _unpack(self, payload):
        if isinstance(payload, (tuple, list)):
            return payload[0], payload[1] if len(payload) > 1 else None
        if isinstance(payload, dict):
            # Try common keys for data
            X = payload.get("data") or payload.get("features") or payload.get("X")
            y = payload.get("labels") or payload.get("targets") or payload.get("y")
            return X, y
        return payload, None

    def _extract_windows(self, signals):
        # Iterate over signals (can be list of arrays or huge 3D array)
        for signal in signals:
            # Ensure float32 and handle NaNs
            signal = np.nan_to_num(np.asarray(signal, dtype=np.float32))

            # Ensure [Channels, Time] format
            if signal.ndim == 1:
                signal = signal[None, :]  # [1, T]
            
            # If shape is [Time, Channels] (T > C usually), transpose
            if signal.shape[0] > signal.shape[1] and signal.shape[1] < 100:
                 signal = signal.T # -> [C, T]

            T = signal.shape[1]

            # Standardize (Mean/Std) per window or signal? 
            # RmGPT paper usually standardizes per window, but efficient loading does it here if needed.
            # Here we just slice. Normalization is often handled by RevIN inside the model.

            # Slicing
            for start in range(0, T - self.seq_len + 1, self.stride):
                window = signal[:, start : start + self.seq_len]
                self.samples.append(window)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]

        timeseries = torch.from_numpy(x).float()  # [C, L]
        # Since we extract fixed windows, mask is all 1s
        input_mask = torch.ones(self.seq_len, dtype=torch.long)

        return {
            "timeseries": timeseries,
            "input_mask": input_mask,
        }