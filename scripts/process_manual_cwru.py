import os
import glob
import numpy as np
import scipy.io
from tqdm import tqdm

# --- PATH CONFIGURATION ---
# Base path of your RmGPT project
PROJECT_ROOT = "/gpfs/workdir/fernandeda/projects/RmGPT"

# Path where processed data will be saved
# CORREGIDO: Usamos PROJECT_ROOT en lugar de PROJECT_DIR
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset_storage/labeled_data/CWRU")

# Path to the sibling folder containing the RAW data
RAW_DATA_ROOT = "/gpfs/workdir/fernandeda/projects/CWRU_Dataset/Data"

# Model Parameters
SEQ_LEN = 2048
STRIDE = 2048
N_CHANNELS = 2  # We will use DE (Drive End) and FE (Fan End) if available, or zero-pad.

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def get_class_from_filename(filename):
    """
    Parses the filename (e.g., 'IR007_0.mat') to determine the class ID (0-15).
    Based on the standard 16-class CWRU benchmark.
    """
    fname = filename.upper()
    
    # --- 0. NORMAL ---
    if "NORMAL" in fname or "97" in fname or "98" in fname or "99" in fname or "100" in fname:
        return 0
    
    # Detect Size (007, 014, 021)
    size = None
    if "007" in fname: size = 0
    elif "014" in fname: size = 1
    elif "021" in fname: size = 2
    
    if size is None: return -1 

    # --- 1. INNER RACE (Classes 1, 2, 3) ---
    if "IR" in fname:
        return 1 + size

    # --- 2. BALL (Classes 4, 5, 6) ---
    if "B" in fname and "OR" not in fname: 
        return 4 + size

    # --- 3. OUTER RACE (Classes 7 to 15) ---
    if "OR" in fname:
        # Detect Position (@6, @3, @12)
        position_offset = 0 # Default @6
        
        if "@3" in fname: position_offset = 1 
        elif "@12" in fname: position_offset = 2 
        elif "@6" in fname: position_offset = 0
        
        base_class = 7 + (position_offset * 3)
        return base_class + size

    return -1

def load_mat_file(filepath):
    """Loads a .mat file and extracts DE and FE signals."""
    try:
        mat = scipy.io.loadmat(filepath)
    except Exception as e:
        print(f"[ERROR] Corrupt file {filepath}: {e}")
        return None

    de_signal = None
    fe_signal = None
    
    for key in mat.keys():
        if "DE_time" in key:
            de_signal = mat[key].flatten()
        elif "FE_time" in key:
            fe_signal = mat[key].flatten()
            
    if de_signal is None:
        return None
        
    if fe_signal is None:
        fe_signal = np.zeros_like(de_signal)
    else:
        min_len = min(len(de_signal), len(fe_signal))
        de_signal = de_signal[:min_len]
        fe_signal = fe_signal[:min_len]

    # Stack: [2, T]
    signal = np.stack([de_signal, fe_signal], axis=0)
    return signal.astype(np.float32)

def process_manual_cwru():
    ensure_dir(OUTPUT_DIR)
    
    print(f"--- Manually Processing CWRU from: {RAW_DATA_ROOT} ---")
    
    all_X = []
    all_y = []
    
    # 1. Process NORMAL Folder
    normal_path = os.path.join(RAW_DATA_ROOT, "Normal")
    if os.path.exists(normal_path):
        files = glob.glob(os.path.join(normal_path, "*.mat"))
        print(f"Processing {len(files)} Normal files...")
        for f in tqdm(files):
            signal = load_mat_file(f)
            if signal is not None:
                _, T = signal.shape
                for start in range(0, T - SEQ_LEN + 1, STRIDE):
                    window = signal[:, start : start + SEQ_LEN]
                    all_X.append(window)
                    all_y.append(0) 
    else:
        print(f"[WARNING] Folder 'Normal' not found in {normal_path}")

    # 2. Process FAULT Folders
    # We check 12k_DE, 12k_FE, and 48k_DE to be comprehensive, 
    # but based on your structure, 12k_DE contains the main labeled files.
    target_folders = ["12k_DE"] 
    
    for folder in target_folders:
        fault_path = os.path.join(RAW_DATA_ROOT, folder)
        if os.path.exists(fault_path):
            files = glob.glob(os.path.join(fault_path, "*.mat"))
            print(f"Processing {len(files)} Fault files in {folder}...")
            
            for f in tqdm(files):
                fname = os.path.basename(f)
                label = get_class_from_filename(fname)
                
                if label == -1:
                    continue
                    
                signal = load_mat_file(f)
                if signal is not None:
                    _, T = signal.shape
                    for start in range(0, T - SEQ_LEN + 1, STRIDE):
                        window = signal[:, start : start + SEQ_LEN]
                        all_X.append(window)
                        all_y.append(label)
        else:
            print(f"[WARNING] Folder '{folder}' not found in {RAW_DATA_ROOT}")

    # 3. Save
    if not all_X:
        print("[ERROR] No windows extracted. Please check the paths.")
        return

    print("Stacking arrays...")
    X_arr = np.stack(all_X)
    y_arr = np.array(all_y, dtype=np.int64)
    
    unique, counts = np.unique(y_arr, return_counts=True)
    print("\n" + "="*30)
    print(f"Final Dataset Generated: {X_arr.shape}")
    print(f"Classes Found: {len(unique)} (Target: 16)")
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")
    print("="*30)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X_arr)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y_arr)
    print(f"[SUCCESS] Data saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    process_manual_cwru()