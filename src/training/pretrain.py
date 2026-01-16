import os
import sys
import torch
from torch.utils.data import DataLoader

# --- LOG SILENCING ---
from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
# ---------------------

from src.mymoment.model.moment import MOMENT
from src.data.npy_dataset import NpyDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running RmGPT-style Pretraining on {device} ---")

    # Hyperparameters from RmGPT Paper
    SEQ_LEN = 2048      
    PATCH_LEN = 256
    STRIDE = 256
    MASK_RATIO = 0.125  
    BATCH_SIZE = 32
    LR = 1e-5 

    # --- MODE CONFIGURATION ---
    # True  = Hyperparameter Search Mode (Uses Train/Val splits)
    # False = Final Production Mode (Uses the full 70% split)
    DO_VALIDATION = False
    
    # Path to the processed data directory
    DATA_DIR = "/gpfs/workdir/fernandeda/projects/dataset_storage/processed_data"

    # 1. Dataset Selection
    if DO_VALIDATION:
        print("[MODE] Hyperparameter Search (Train/Val Split)")
        train_path = os.path.join(DATA_DIR, "pretrain_train.npy")
        val_path = os.path.join(DATA_DIR, "pretrain_val.npy")
    else:
        print("[MODE] Final Production Training (Full 70% Split)")
        train_path = os.path.join(DATA_DIR, "pretrain_full.npy")
        val_path = None

    # Verify paths
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found: {train_path}")

    # Load Training Data
    train_dataset = NpyDataset(npy_file_path=train_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True, 
        drop_last=True
    )
    
    # Load Validation Data (Optional)
    val_loader = None
    if val_path and os.path.exists(val_path):
        val_dataset = NpyDataset(npy_file_path=val_path)
        # Validation does not need shuffle or drop_last
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    else:
        print(f"Train samples: {len(train_dataset)} (No validation)")

    # 2. Model
    model = MOMENT(
        {
            "task_name": "pre-training",
            "seq_len": SEQ_LEN,
            "patch_len": PATCH_LEN,
            "patch_stride_len": STRIDE,
            "transformer_backbone": "google/flan-t5-small", 
            "transformer_type": "encoder_only",
            "mask_ratio": MASK_RATIO,
            "device": device,
            "d_model": 512,
            "n_channels": 2, # Critical: Matches the standardized NPY format
        }
    ).to(device)

    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = torch.nn.MSELoss(reduction="none")

    print("Starting training loop...")
    
    for epoch in range(1, 51): # 50 Epochs
        
        # --- TRAIN LOOP ---
        model.train()
        total_loss, steps = 0.0, 0
        
        for batch in train_loader:
            x = batch["timeseries"].to(device)           # [B, 2, 2048]
            input_mask = batch["input_mask"].to(device)  # [B, 2048]

            # Forward pass
            out = model(x_enc=x, input_mask=input_mask)

            # Reconstruction Loss
            recon_loss = criterion(out.reconstruction, x)

            # Compute Target Mask
            # (1 - out.pretrain_mask) selects the masked patches (Target)
            target_mask = (1 - out.pretrain_mask) * input_mask 
            target_mask = target_mask.unsqueeze(1) # Broadcast to channels [B, 1, L]

            # Compute Scalar Loss (Masked MSE)
            loss = (recon_loss * target_mask).sum() / (target_mask.sum() + 1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        avg_train_loss = total_loss / max(steps, 1)

        # --- VALIDATION LOOP ---
        if val_loader:
            model.eval()
            val_loss_accum = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["timeseries"].to(device)
                    input_mask = batch["input_mask"].to(device)

                    out = model(x_enc=x, input_mask=input_mask)
                    recon_loss = criterion(out.reconstruction, x)
                    
                    target_mask = (1 - out.pretrain_mask) * input_mask 
                    target_mask = target_mask.unsqueeze(1)
                    
                    loss = (recon_loss * target_mask).sum() / (target_mask.sum() + 1e-6)
                    val_loss_accum += float(loss.item())
                    val_steps += 1
            
            avg_val_loss = val_loss_accum / max(val_steps, 1)
            print(f"Epoch {epoch}/50 - Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/50 - Train Loss: {avg_train_loss:.6f}")

    # --- SAVE CHECKPOINT ---
    print("--- Saving Model ---")
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/rmgpt_pretrain_final.pth"
    
    # Save state dict
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()