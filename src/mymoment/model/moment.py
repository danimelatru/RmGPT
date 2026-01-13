# src/mymoment/model/moment.py
from __future__ import annotations

import warnings
from typing import Any, Optional, Union

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
        # x: [B, C, n_patches, d_model] -> [B, C, n_patches, patch_len] -> [B, C, n_patches*patch_len]
        x = self.linear(self.dropout(x))
        return x.flatten(start_dim=2, end_dim=3)


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
    # Requires transformers
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(hf_name)
    if hasattr(c, "d_model"):
        return int(c.d_model)
    if hasattr(c, "hidden_size"):
        return int(c.hidden_size)
    raise ValueError(f"Cannot infer d_model from config for {hf_name}")


class MOMENT(nn.Module):
    """
    Minimal MOMENT for pretraining:
      Input:
        x_enc      [B, C, L]
        input_mask [B, L]  (1 observed, 0 padding)
      Output:
        reconstruction [B, C, L]
        pretrain_mask  [B, L] (1 keep, 0 masked)
    """

    def __init__(self, configs: Union[dict, Any]):
        super().__init__()
        self.cfg = configs

        # Core config
        self.seq_len = int(_get(configs, "seq_len", 512))
        self.patch_len = int(_get(configs, "patch_len", 8))
        self.stride = int(_get(configs, "patch_stride_len", self.patch_len))
        self.mask_ratio = float(_get(configs, "mask_ratio", 0.3))

        # Channels (important for RevIN if affine=True)
        self.n_channels = int(_get(configs, "n_channels", 1))

        transformer_backbone = _get(configs, "transformer_backbone", "google/flan-t5-large")
        transformer_type = _get(configs, "transformer_type", "encoder_only")

        if transformer_backbone not in SUPPORTED_HUGGINGFACE_MODELS:
            raise ValueError(
                f"Unsupported backbone: {transformer_backbone}. "
                f"Choose one of {SUPPORTED_HUGGINGFACE_MODELS}"
            )

        if transformer_type != "encoder_only":
            warnings.warn("This minimal MOMENT only supports encoder_only. Forcing encoder_only.")
            transformer_type = "encoder_only"

        # Infer d_model if missing
        if _get(configs, "d_model", None) is None:
            _set(configs, "d_model", _infer_d_model(transformer_backbone))

        self.d_model = int(_get(configs, "d_model"))
        dropout = float(_get(configs, "dropout", 0.1))
        orth_gain = float(_get(configs, "orth_gain", 1.41))
        revin_affine = bool(_get(configs, "revin_affine", False))

        # Model blocks
        self.normalizer = RevIN(num_features=self.n_channels, affine=revin_affine)
        self.tokenizer = Patching(patch_len=self.patch_len, stride=self.stride)

        self.patch_embedding = PatchEmbedding(
            d_model=self.d_model,
            seq_len=self.seq_len,
            patch_len=self.patch_len,
            stride=self.stride,
            dropout=dropout,
            add_positional_embedding=bool(_get(configs, "add_positional_embedding", True)),
            value_embedding_bias=bool(_get(configs, "value_embedding_bias", False)),
            orth_gain=orth_gain,
        )

        self.mask_generator = Masking(mask_ratio=self.mask_ratio, patch_len=self.patch_len, stride=self.stride)

        # HF encoder backbone
        self.encoder = self._build_hf_encoder(transformer_backbone)

        # Pretrain head
        self.head = PretrainHead(d_model=self.d_model, patch_len=self.patch_len, dropout=dropout, orth_gain=orth_gain)

    def _build_hf_encoder(self, name: str) -> nn.Module:
        # Requires transformers
        from transformers import T5EncoderModel

        # T5EncoderModel is already the encoder.
        # We'll call it directly as: encoder(inputs_embeds=..., attention_mask=...)
        return T5EncoderModel.from_pretrained(name)

    def pretraining(
        self,
        x_enc: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> TimeseriesOutputs:
        """
        x_enc:      [B, C, L]
        input_mask: [B, L] 1 observed, 0 padding
        mask:       [B, L] 1 keep, 0 masked (if None, generated)
        """
        B, C, L = x_enc.shape

        # Basic safety checks (keeps the pipeline predictable)
        if C != self.n_channels:
            # You can relax this later, but it avoids silent wrong RevIN affine shapes.
            raise ValueError(f"Expected C={self.n_channels} channels but got C={C}.")

        if L != self.seq_len:
            raise ValueError(f"Expected sequence length L={self.seq_len} but got L={L}. "
                             f"Pad/crop your series to seq_len before calling MOMENT.")

        if input_mask is None:
            input_mask = torch.ones((B, L), device=x_enc.device, dtype=torch.long)
        else:
            input_mask = input_mask.to(device=x_enc.device, dtype=torch.long)

        if mask is None:
            mask = self.mask_generator.generate_mask(x=x_enc, input_mask=input_mask).to(x_enc.device, dtype=torch.long)
        else:
            mask = mask.to(device=x_enc.device, dtype=torch.long)

        # Normalize on observed & kept
        x_norm = self.normalizer(x=x_enc, mask=mask * input_mask, mode="norm")
        x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # Tokenize + embed
        x_tok = self.tokenizer(x=x_norm)                     # [B, C, n_patches, patch_len]
        enc_in = self.patch_embedding(x_tok, mask=mask)      # [B, C, n_patches, d_model]

        n_patches = enc_in.shape[2]
        enc_in = enc_in.reshape(B * C, n_patches, self.d_model)  # [B*C, n_patches, d_model]

        # Attention mask in patch view: [B, n_patches] -> [B*C, n_patches]
        attn_mask = Masking.convert_seq_to_patch_view(
            input_mask, patch_len=self.patch_len, stride=self.stride
        ).to(device=x_enc.device, dtype=torch.long)

        attn_mask = attn_mask.repeat_interleave(C, dim=0)  # [B*C, n_patches]

        # Encode (T5EncoderModel returns BaseModelOutput)
        out = self.encoder(inputs_embeds=enc_in, attention_mask=attn_mask)
        h = out.last_hidden_state                          # [B*C, n_patches, d_model]
        h = h.reshape(B, C, n_patches, self.d_model)       # [B, C, n_patches, d_model]

        # Decode to reconstruction
        recon = self.head(h)                               # [B, C, n_patches*patch_len]

        # Ensure it matches seq_len (in MOMENT setup it should)
        if recon.shape[-1] != L:
            raise RuntimeError(
                f"Reconstruction length {recon.shape[-1]} != input length {L}. "
                f"Check (seq_len, patch_len, stride)."
            )

        recon = self.normalizer(x=recon, mode="denorm")

        return TimeseriesOutputs(
            input_mask=input_mask,
            reconstruction=recon,
            pretrain_mask=mask,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        input_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> TimeseriesOutputs:
        return self.pretraining(x_enc=x_enc, input_mask=input_mask, mask=mask)
