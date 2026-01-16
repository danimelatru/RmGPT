import os
import numpy as np
import pandas as pd
import shutil
import random
from phmd import datasets
from tqdm import tqdm

# --- CONFIGURATION ---
PROJECT_DIR = "/gpfs/workdir/fernandeda/projects/dataset_storage"
RAW_ZIPS_DIR = os.path.join(PROJECT_DIR, "raw_zips")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "processed_data")

# RmGPT Parameters
SEQ_LEN = 2048
STRIDE = 2048
N_CHANNELS = 2

TARGET_DATASETS = ["CWRU", "XJTU-SY", "KAUG17", "HSG18", "MFPT"]
TEMP_ENV = os.path.join(PROJECT_DIR, "temp_phmd_env")
os.environ["PHMD_DATA"] = TEMP_ENV

def setup_environment():
    if os.path.exists(TEMP_ENV): shutil.rmtree(TEMP_ENV)
    os.makedirs(os.path.join(TEMP_ENV, "datasets"), exist_ok=True)
    if not os.path.exists(RAW_ZIPS_DIR): return

    for zip_file in os.listdir(RAW_ZIPS_DIR):
        if zip_file.endswith(".zip"):
            src = os.path.join(RAW_ZIPS_DIR, zip_file)
            dst = os.path.join(TEMP_ENV, "datasets", zip_file)
            if not os.path.exists(dst): os.symlink(src, dst)

def process_recursively(data, accumulator):
    if isinstance(data, (list, tuple)):
        for item in data: process_recursively(item, accumulator)
    elif isinstance(data, dict):
        for item in data.values(): process_recursively(item, accumulator)
    elif isinstance(data, (pd.DataFrame, np.ndarray)):
        extract_windows(data, accumulator)

def extract_windows(data, accumulator):
    signal = None
    if isinstance(data, pd.DataFrame):
        valid_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) 
                      and not any(x in str(c).lower() for x in ['unit', 'fault', 'label', 'setting', 'condition'])]
        if valid_cols: signal = data[valid_cols].values
    else:
        signal = data

    if signal is not None:
        signal = np.nan_to_num(signal.astype(np.float32))
        
        # Standardize Dimensions to (Channels, Time)
        if signal.ndim == 1: signal = signal[None, :]
        elif signal.shape[0] > signal.shape[1]: signal = signal.T
        
        # Check Length
        if signal.shape[1] < SEQ_LEN: return

        # Force 2 Channels
        C, T = signal.shape
        final_signal = np.zeros((N_CHANNELS, T), dtype=np.float32)
        copy_c = min(C, N_CHANNELS)
        final_signal[:copy_c, :] = signal[:copy_c, :]
        
        # Slicing (Vectorized approach for speed)
        # Instead of saving files, we append the window array to the list
        for start in range(0, T - SEQ_LEN + 1, STRIDE):
            window = final_signal[:, start : start + SEQ_LEN]
            accumulator.append(window)

def select_task_robust(ds):
    candidates = ['fault', 'Fault', 'rul', 'Diagnosis', 'Prognosis', 'classification']
    for key in candidates:
        try: return ds[key]
        except: continue
    try: return ds[list(ds.meta.get('tasks', {}).keys())[0]]
    except: return None

def main():
    setup_environment()
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    all_windows = []

    print("\n--- Processing Datasets (In-Memory) ---")
    for name in TARGET_DATASETS:
        print(f"Processing {name}...")
        try:
            ds = datasets.Dataset(name)
            task = select_task_robust(ds)
            
            if task:
                ds_windows = []
                process_recursively(task[0], ds_windows)
                count = len(ds_windows)
                print(f"   -> Extracted {count} windows.")
                all_windows.extend(ds_windows)
            else:
                print(f"   [WARNING] No task found for {name}")

        except Exception as e:
            print(f"   [ERROR] Failed {name}: {e}")

    # --- AGGREGATE AND SPLIT ---
    print("\n--- Aggregating & Saving Big Files ---")
    if not all_windows:
        print("[ERROR] No data extracted.")
        return

    # Stack into one big tensor [Total_Samples, 2, 2048]
    # 200k samples * 16KB ~= 3.2 GB (Fits easily in RAM)
    full_data = np.stack(all_windows, axis=0)
    print(f"Total Data Shape: {full_data.shape} ({full_data.nbytes / 1e9:.2f} GB)")

    # Shuffle
    indices = np.arange(len(full_data))
    np.random.shuffle(indices)
    full_data = full_data[indices]

    # Split Indices
    total = len(full_data)
    idx_70 = int(total * 0.7)
    
    data_pretrain_all = full_data[:idx_70]
    data_finetune = full_data[idx_70:]
    
    idx_val = int(len(data_pretrain_all) * 0.85)
    data_train = data_pretrain_all[:idx_val]
    data_val = data_pretrain_all[idx_val:]

    # Save Big Files
    np.save(os.path.join(OUTPUT_DIR, "pretrain_full.npy"), data_pretrain_all)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_train.npy"), data_train)
    np.save(os.path.join(OUTPUT_DIR, "pretrain_val.npy"), data_val)
    np.save(os.path.join(OUTPUT_DIR, "finetune.npy"), data_finetune)

    print("\n[SUCCESS] Saved 4 large .npy files in processed_data/")
    
    # Cleanup temp env
    if os.path.exists(TEMP_ENV): shutil.rmtree(TEMP_ENV)

if __name__ == "__main__":
    main()