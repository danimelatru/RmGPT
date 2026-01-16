# src/mymoment/model/moment.py
from __future__ import annotations

import warnings
from typing import Any, Optional, Union
from dataclasses import dataclass

import torch
from torch import nn

from .layers import PatchEmbedding, Patching, RevIN
from .masking import Masking
from .outputs import TimeseriesOutputs

SUPPORTED_HUGGINGFACE_MODELS = [
    "t5-small",
    "t5-base",
    "t5-large",
    "google/flan-t5-small",
    "google/flan-t5-base",
    "google/flan-t5-large",
]

@dataclass
class ClassificationOutput:
    logits: torch.Tensor
    input_mask: torch.Tensor

class PretrainHead(nn.Module):
    """Maps patch embeddings back to time domain patches."""
    def __init__(self, d_model: int, patch_len: int, dropout: float = 0.1, orth_gain: float = 1.41):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, patch_len)
        if orth_gain is not None:
            torch.nn.init.orthogonal_(self.linear.weight, gain=orth_gain)
            self.linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, n_patches, d_model]
        x = self.linear(self.dropout(x))
        return x.flatten(start_dim=2, end_dim=3)

class ClassificationHead(nn.Module):
    """Maps patch embeddings to class logits."""
    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, n_patches, d_model]
        # Global Mean Pooling: Average over Channels and Patches
        # Result: [B, d_model]
        x = x.mean(dim=(1, 2))
        return self.linear(self.dropout(x))

def _get(cfg: Union[dict, Any], key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)

def _set(cfg: Union[dict, Any], key: str, value: Any) -> None:
    if isinstance(cfg, dict):
        cfg[key] = value
    else:
        setattr(cfg, key, value)

def _infer_d_model(hf_name: str) -> int:
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(hf_name)
    if hasattr(c, "d_model"): return int(c.d_model)
    if hasattr(c, "hidden_size"): return int(c.hidden_size)
    raise ValueError(f"Cannot infer d_model from config for {hf_name}")

class MOMENT(nn.Module):
    """
    MOMENT model supporting both Pretraining and Classification.
    """
    def __init__(self, configs: Union[dict, Any]):
        super().__init__()
        self.cfg = configs

        # Core config
        self.task_name = _get(configs, "task_name", "pre-training")
        self.n_class = int(_get(configs, "n_class", 2))
        self.seq_len = int(_get(configs, "seq_len", 512))
        self.patch_len = int(_get(configs, "patch_len", 8))
        self.stride = int(_get(configs, "patch_stride_len", self.patch_len))
        self.mask_ratio = float(_get(configs, "mask_ratio", 0.3))
        self.n_channels = int(_get(configs, "n_channels", 1))

        transformer_backbone = _get(configs, "transformer_backbone", "google/flan-t5-large")
        if transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS:
            raise ValueError(f"Unsupported backbone: {transformer_backbone}")

        # Infer d_model
        if _get(configs, "d_model", None) is None:
            _set(configs, "d_model", _infer_d_model(transformer_backbone))

        self.d_model = int(_get(configs, "d_model"))
        dropout = float(_get(configs, "dropout", 0.1))
        orth_gain = float(_get(configs, "orth_gain", 1.41))
        revin_affine = bool(_get(configs, "revin_affine", False))

        # Shared Blocks
        self.normalizer = RevIN(num_features=self.n_channels, affine=revin_affine)
        self.tokenizer = Patching(patch_len=self.patch_len, stride=self.stride)
        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model, seq_len=self.seq_len, patch_len=self.patch_len,
            stride=self.stride, dropout=dropout,
            add_positional_embedding=bool(_get(configs, "add_positional_embedding", True)),
            orth_gain=orth_gain,
        )
        self.mask_generator = Masking(mask_ratio=self.mask_ratio, patch_len=self.patch_len, stride=self.stride)
        self.encoder = self._build_hf_encoder(transformer_backbone)

        # Task Specific Head
        if self.task_name == "classification":
            self.head = ClassificationHead(d_model=self.d_model, n_classes=self.n_class, dropout=dropout)
        else:
            self.head = PretrainHead(d_model=self.d_model, patch_len=self.patch_len, dropout=dropout, orth_gain=orth_gain)

    def _build_hf_encoder(self, name: str) -> nn.Module:
        from transformers import T5EncoderModel
        return T5EncoderModel.from_pretrained(name)

    def forward(self, x_enc: torch.Tensor, input_mask: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        if self.task_name == "classification":
            return self.classification(x_enc, input_mask)
        return self.pretraining(x_enc, input_mask, mask)

    def pretraining(self, x_enc, input_mask=None, mask=None):
        B, C, L = x_enc.shape
        if input_mask is None: input_mask = torch.ones((B, L), device=x_enc.device, dtype=torch.long)
        if mask is None: mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask).to(x_enc.device, dtype=torch.long)

        # Encode
        h, _ = self._encode(x_enc, input_mask, mask)
        
        # Reconstruct
        recon = self.head(h)
        recon = self.normalizer(x=recon, mode="denorm")

        return TimeseriesOutputs(input_mask=input_mask, reconstruction=recon, pretrain_mask=mask)

    def classification(self, x_enc, input_mask=None):
        B, C, L = x_enc.shape
        if input_mask is None: input_mask = torch.ones((B, L), device=x_enc.device, dtype=torch.long)
        
        # No masking for classification (use all data)
        mask = torch.ones_like(input_mask)

        # Encode
        h, _ = self._encode(x_enc, input_mask, mask)
        
        # Classify
        logits = self.head(h) # [B, n_classes]

        return ClassificationOutput(logits=logits, input_mask=input_mask)

    def _encode(self, x_enc, input_mask, mask):
        B, C, L = x_enc.shape
        
        # Normalize
        x_norm = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0.0)

        # Tokenize & Embed
        x_tok = self.tokenizer(x=x_norm)
        enc_in = self.patch_embedding(x_tok, mask=mask) # [B, C, n_patches, d_model]

        # Reshape for HF Encoder
        n_patches = enc_in.shape[2]
        enc_in_flat = enc_in.reshape(B * C, n_patches, self.d_model)

        # Attention Mask
        attn_mask = Masking.convert_seq_to_patch_view(input_mask, self.patch_len, self.stride).to(x_enc.device, dtype=torch.long)
        attn_mask = attn_mask.repeat_interleave(C, dim=0)

        # Transformer Pass
        out = self.encoder(inputs_embeds=enc_in_flat, attention_mask=attn_mask)
        h_flat = out.last_hidden_state
        
        # Reshape back
        h = h_flat.reshape(B, C, n_patches, self.d_model)
        
        return h, mask