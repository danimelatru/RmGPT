import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

from src.mymoment.model.moment import MOMENT
from src.data.industrial_dataset import IndustrialPretrainDataset


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Running RmGPT-style Pretraining on {device} ---")

    # Hyperparameters from RmGPT Paper
    SEQ_LEN = 2048      # 8 patches * 256
    PATCH_LEN = 256
    STRIDE = 256
    MASK_RATIO = 0.125  # (1/8) Informational only

    # 1. Dataset
    dataset = IndustrialPretrainDataset(
        dataset_names=["CWRU", "XJTU-SY", "KAUG17"],
        seq_len=SEQ_LEN,
        stride=SEQ_LEN,
        download=True,
    )
    print(f"Dataset loaded. Samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 2. Model
    model = MOMENT(
        {
            "task_name": "pre-training",
            "seq_len": SEQ_LEN,
            "patch_len": PATCH_LEN,
            "patch_stride_len": STRIDE,
            "transformer_backbone": "google/flan-t5-small", # Change to 'base' if there is enough GPU
            "transformer_type": "encoder_only",
            "mask_ratio": MASK_RATIO,
            "device": device,
            "d_model": 512, # Matches with RmGPT small
        }
    ).to(device)

    # 3. Optimizer
    # Start with 1e-5 for stability/sanity check (Paper uses 3e-7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss(reduction="none")

    print("Starting training loop...")
    model.train()

    for epoch in range(1, 21): # 20 Epochs
        total_loss, steps = 0.0, 0

        for batch in loader:
            x = batch["timeseries"].to(device)           # [B, C, L]
            input_mask = batch["input_mask"].to(device)  # [B, L]

            # Forward pass: generates mask internally using the new Last-Patch logic
            out = model(x_enc=x, input_mask=input_mask)

            # Reconstruction Loss: [B, C, L]
            recon_loss = criterion(out.reconstruction, x)

            # Compute Target Mask
            # We want to train on: (Where model predicted) AND (Where data is real/observed)
            # out.pretrain_mask: 1=Context, 0=Target
            # So (1 - out.pretrain_mask) gives us the Target region.
            target_mask = (1 - out.pretrain_mask) * input_mask 
            target_mask = target_mask.unsqueeze(1) # Broadcast to channels [B, 1, L]

            # Compute Scalar Loss
            # Sum error only in target region / divide by number of target elements
            loss = (recon_loss * target_mask).sum() / (target_mask.sum() + 1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        print(f"Epoch {epoch}/20 - Avg Loss: {total_loss / max(steps, 1):.6f}")


if __name__ == "__main__":
    main()