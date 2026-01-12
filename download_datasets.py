import sys
import os

# --- PATH CORRECTION ---
# Get the current directory path (RmGPT folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory path (the phmd repo root)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to sys.path to enable 'import phmd'
sys.path.append(parent_dir)
# -----------------------

print(f"Working directory: {current_dir}")
print(f"Searching for phmd library in: {parent_dir}")

try:
    from phmd import Dataset
    print("[OK] 'phmd' library successfully imported from parent directory.")
except ImportError as e:
    print(f"[ERROR] Could not import 'phmd': {e}")
    print("Ensure the folder structure is: phmd_repo/RmGPT/this_script.py")
    sys.exit(1)

# List of datasets from the RmGPT paper
# Keys: Common Name / Values: List of possible IDs in the library
datasets_targets = {
    "CWRU": ["CWRU", "CaseWestern"],
    "XJTU": ["XJTU", "XJTU-SY"],
    "SMU": ["SMU", "Gear", "Zamanian"],
    "QPZZ": ["QPZZ", "QPZZ-II", "QianPeng"],
    "SLIET": ["SLIET", "SantLongowal"]
}

missing_datasets = []

print("\n--- Verifying Dataset Availability ---")

# Attempt to get the official list of available IDs in phmd to assist search
available_ids = []
try:
    # This method usually returns a list or dictionary of available datasets
    # If the API has changed, this might fail, hence the try/except block
    if hasattr(Dataset, 'list_datasets'):
        available_ids = Dataset.list_datasets()
    elif hasattr(Dataset, 'available'):
        available_ids = Dataset.available
except:
    pass

for key, keywords in datasets_targets.items():
    found = False
    print(f"Searching for '{key}'...", end=" ")
    
    # 1. Exact search if we have the list of IDs
    if available_ids:
        for kw in keywords:
            if kw in available_ids:
                print(f"FOUND (ID: {kw})")
                found = True
                break
    
    # 2. If not found, attempt instantiation to see if it fails
    if not found:
        for name in keywords:
            try:
                # Attempt initialization (do not download yet)
                ds = Dataset(name)
                print(f"FOUND as '{name}'.")
                
                # --- AUTOMATIC DOWNLOAD (Optional) ---
                # Uncomment the following lines if you want to download automatically
                # print(f"Downloading {name}...")
                # ds.download() 
                
                found = True
                break
            except Exception:
                continue

    if not found:
        print("NOT FOUND automatically.")
        missing_datasets.append(key)

print("\n--- Summary ---")
if not missing_datasets:
    print("All set! All datasets are available.")
else:
    print(f"The following datasets are missing: {missing_datasets}")
    print("\n[Tip] If 'phmd' supports listing datasets, run: print(Dataset.list_datasets())")
    print("to see the exact names (IDs) expected by the library.")

    # Help information for missing datasets
    if "SMU" in missing_datasets:
        print(" -> SMU: Search for 'Zamanian' or 'Gear Fault' in the documentation.")
    if "QPZZ" in missing_datasets:
        print(" -> QPZZ: Sometimes listed as 'QianPeng'.")