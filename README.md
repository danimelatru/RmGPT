# RmGPT – Industrial Time-Series Pretraining (Simplified Pipeline)

This repository contains a **functional implementation** of an RmGPT-style self-supervised pretraining pipeline for industrial time-series data.

The goal is **not** to reproduce the original RmGPT codebase line by line, but to:
- Faithfully replicate the **learning objective** described in the paper
- Keep the codebase **simple, inspectable, and extensible**

---

## 1. What this project does

This project implements **Next Signal Token (Patch) Prediction** for industrial time-series data:

- A long signal is split into **non-overlapping windows**
- Each window is split into **fixed-length patches**
- The **last patch is masked**
- The model is trained to **reconstruct only that last patch** using the preceding patches as context

This matches the self-supervised objective described in the RmGPT paper.

---

## 2. Current project structure

```text
RmGPT/
├── src/
│   ├── data/
│   │   └── industrial_dataset.py
│   │
│   └── mymoment/
│       └── model/
│           ├── moment.py
│           ├── layers.py
│           ├── masking.py
│           └── outputs.py
│
├── training/
│   └── pretrain.py
│
├── README.md
```

---

## 3. Dataset support

### Supported Data
The pipeline uses `phmd` to access validated industrial datasets. The loader automatically handles missing datasets by substituting them with equivalent open-source alternatives:

- **Bearing Data:** CWRU, XJTU-SY, MFPT (substitute for SLIET)
- **Gear Data:** KAUG17 (substitute for SMU), HSG18 (substitute for QPZZ)

### Using your own data
The file `industrial_dataset.py` is extensible. It expects data in the format:
- **Timeseries:** `[Channels, Seq_Len]` (Float)
- **Input Mask:** `[Seq_Len]` (Long, 1=observed, 0=padding)

---

## 4. Model overview (MOMENT)

The model in `moment.py` consists of:

1. **RevIN normalization**
 - Per-sample normalization to handle distribution shift
2. **Patch-based tokenizer**
 - Splits the signal into patches of length `P`
3. **Patch embedding**
 - Linear projection + optional positional encoding
4. **Transformer encoder**
 - HuggingFace T5 encoder (encoder-only)
5. **Pretraining head**
 - Projects embeddings back to time-domain patches

The model only supports **pretraining** (by design).

---

## 5. Masking strategy (critical)

Implemented in `masking.py`.

This is **not random masking**.

Instead:
- All patches are visible **except the last observed patch**
- The last patch is the **only prediction target**
- Padding (if any) is automatically excluded

This exactly matches the RmGPT objective:
> Predict the next signal token using the previous tokens as context.

---

## 6. Training objective

## 6. Training objective

During training:

- Loss is computed using **MSE**.
- Loss is applied **only to the masked region** (the target patch).
- Context patches do not contribute to the loss.

Mathematically:
```python
loss = (MSE(reconstruction, input) * target_mask).sum() / target_mask.sum()
```
Where `target_mask` identifies only the last patch of the sequence.

---

## 7. Pretraining configuration

Hyperparameters are aligned with RmGPT Table III:

| Parameter            | Value | Description |
|---------------------|-------|-------------|
| Patch length (P)    | 256   | Length of each token |
| Stride (S)          | 256   | Non-overlapping patches |
| Context patches     | 7     | Visible history |
| Target patches      | 1     | Masked future |
| Sequence length     | 2048  | Total window size (8 * 256) |
| Epochs              | 20    | Pretraining duration |

---

## 8. What this implementation does NOT include (by design)

This repository intentionally excludes:

- Fine-tuning heads (classification, RUL, fault diagnosis)
- Evaluation scripts
- Logging frameworks (wandb, tensorboard)
- Multi-task objectives
- Distributed training boilerplate

The focus is **clarity and correctness**, not feature completeness.

---

## 9. Dependencies

Minimal required dependencies:
- Python ≥ 3.9
- PyTorch
- transformers
- phmd
- numpy

Optional:
- CUDA for GPU training