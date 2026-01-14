import sys
import os

# Ensure phmd is found if not in the global path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from phmd import datasets
    print("[OK] phmd imported successfully.")
except ImportError:
    print("[ERROR] phmd not found. Make sure the environment is activated.")
    sys.exit(1)

# List of datasets used in your training
DATASETS_TO_DOWNLOAD = ["CWRU", "XJTU-SY", "KAUG17"]

def main():
    print("--- Starting data download on Login Node ---")
    
    for name in DATASETS_TO_DOWNLOAD:
        print(f"\nProcessing '{name}'...")
        try:
            ds = datasets.Dataset(name)
            
            # Force download
            print(f"   -> Downloading {name} (this may take a while)...")
            ds.download()
            print(f"   [OK] {name} downloaded/verified.")
            
            # Quick read verification
            print(f"   -> Attempting to read structure...")
            task_keys = list(ds.meta.get('tasks', {}).keys())
            if task_keys:
                print(f"   -> Tasks found: {task_keys}")
                # Attempt to load a small amount of data to ensure no corruption
                task = ds[task_keys[0]]
                data = task[0]
                print(f"   -> Read successful. Data is ready.")
            else:
                print(f"   [WARNING] {name} has no tasks defined in metadata.")
                
        except Exception as e:
            print(f"   [ERROR] Download/Read failed for {name}: {e}")

    print("\n--- Process finished ---")
    print("You can now submit the sbatch job.")

if __name__ == "__main__":
    main()