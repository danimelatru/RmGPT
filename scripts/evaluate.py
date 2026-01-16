import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder

# Import project modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mymoment.model.moment import MOMENT
from src.data.supervised_dataset import SupervisedDataset

def main():
    torch.manual_seed(42) # Must match finetune.py
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Evaluating Best Model on {device} ---")

    # Paths
    PROJECT_ROOT = "/gpfs/workdir/fernandeda/projects/RmGPT"
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset_storage/labeled_data/CWRU")
    BEST_MODEL_PATH = os.path.join(PROJECT_ROOT, "checkpoints/rmgpt_finetune_best.pth")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Load Data
    dataset = SupervisedDataset(
        os.path.join(DATA_DIR, "X.npy"),
        os.path.join(DATA_DIR, "y.npy")
    )
    
    encoder = LabelEncoder()
    dataset.y = encoder.fit_transform(dataset.y)
    num_classes = len(encoder.classes_)
    
    # Create descriptive class names
    class_names = [f"Class {c}" for c in encoder.classes_]

    # Split 80/20 (Recreate Validation Set)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_set = random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    print(f"Evaluating on {len(test_set)} validation samples.")

    # 2. Load Model
    model = MOMENT({
        "task_name": "classification",
        "n_class": num_classes,
        "seq_len": 2048,
        "patch_len": 256,
        "patch_stride_len": 256,
        "transformer_backbone": "google/flan-t5-small",
        "transformer_type": "encoder_only",
        "device": device,
        "d_model": 512,
        "n_channels": 2,
    }).to(device)

    if os.path.exists(BEST_MODEL_PATH):
        print(f"Loading weights from {BEST_MODEL_PATH}")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    else:
        raise FileNotFoundError("Best model checkpoint not found! Run finetuning first.")

    # 3. Inference
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["timeseries"].to(device)
            mask = batch["input_mask"].to(device)
            y = batch["labels"].to(device)

            output = model(x_enc=x, input_mask=mask)
            preds = torch.argmax(output.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 4. Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - CWRU Fault Diagnosis')
    
    save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to: {save_path}")

    # Text Report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    with open(os.path.join(RESULTS_DIR, "report.txt"), "w") as f:
        f.write(report)

if __name__ == "__main__":
    main()