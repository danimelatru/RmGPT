import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from src.mymoment.model.moment import MOMENT
from src.data.supervised_dataset import SupervisedDataset

def main():
    # --- REPRODUCIBILITY ---
    torch.manual_seed(42) # Ensures the 80/20 split is consistent
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running RmGPT Fine-Tuning (Fault Diagnosis) on {device} ---")

    # --- CONFIGURATION ---
    FREEZE_BACKBONE = False 
    EPOCHS = 50
    
    # Path to data
    DATA_DIR = "/gpfs/workdir/fernandeda/projects/RmGPT/dataset_storage/labeled_data/CWRU"
    PRETRAIN_PATH = "checkpoints/rmgpt_pretrain_final.pth"
    BEST_MODEL_PATH = "checkpoints/rmgpt_finetune_best.pth"
    
    os.makedirs("checkpoints", exist_ok=True)

    # 1. Load Dataset
    dataset = SupervisedDataset(
        os.path.join(DATA_DIR, "X.npy"),
        os.path.join(DATA_DIR, "y.npy")
    )
    
    # Remap labels
    print("Remapping labels...")
    encoder = LabelEncoder()
    dataset.y = encoder.fit_transform(dataset.y)
    
    num_classes = len(encoder.classes_)
    print(f"Detected {num_classes} unique classes.")
    
    # Save classes for evaluation later
    np.save("checkpoints/label_classes.npy", encoder.classes_)

    # Split 80/20
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 2. Model
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
    
    # Load Pretrained Weights
    if os.path.exists(PRETRAIN_PATH):
        print(f"Loading Self-Supervised weights from {PRETRAIN_PATH}...")
        pretrained_dict = torch.load(PRETRAIN_PATH, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Weights loaded successfully.")
    else:
        print("[WARNING] Checkpoint not found! Training from scratch.")

    if FREEZE_BACKBONE:
        for name, param in model.named_parameters():
            if "head" not in name: param.requires_grad = False

    # 3. Training Loop
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    print(f"Starting Training for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1): 
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            x = batch["timeseries"].to(device)
            mask = batch["input_mask"].to(device)
            y = batch["labels"].to(device)
            
            optimizer.zero_grad()
            
            output = model(x_enc=x, input_mask=mask) 
            logits = output.logits 
            
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["timeseries"].to(device)
                mask = batch["input_mask"].to(device)
                y = batch["labels"].to(device)
                
                output = model(x_enc=x, input_mask=mask)
                preds = torch.argmax(output.logits, dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / val_total
        
        # --- SAVE BEST MODEL ---
        log_msg = f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%}"
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            log_msg += f" --> New Best Saved! ({val_acc:.2%})"
        
        print(log_msg)

if __name__ == "__main__":
    main()