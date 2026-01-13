from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    Normalizes per-sample, per-channel over the time dimension.
    """

    def __init__(self, num_features: int = 1, affine: bool = False, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if affine:
            self.gamma = nn.Parameter(torch.ones(1, num_features, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_features, 1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self._cached_mean: Optional[torch.Tensor] = None
        self._cached_std: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, mode: str = "norm"):
        """
        x: [B, C, L]
        mask: [B, L] with 1 observed, 0 padded (optional)
        """
        if mode == "norm":
            mean, std = self._compute_stats(x, mask)
            self._cached_mean = mean
            self._cached_std = std
            x_norm = (x - mean) / (std + self.eps)
            if self.affine:
                x_norm = x_norm * self.gamma + self.beta
            return x_norm

        if mode == "denorm":
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError("RevIN denorm called before norm.")
            x_den = x
            if self.affine:
                x_den = (x_den - self.beta) / (self.gamma + self.eps)
            x_den = x_den * (self._cached_std + self.eps) + self._cached_mean
            return x_den

        raise ValueError(f"Unknown mode: {mode}")

    def _compute_stats(self, x: torch.Tensor, mask: Optional[torch.Tensor]):
        if mask is None:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True, unbiased=False)
            return mean, std

        # mask: [B, L] -> [B, 1, L]
        m = mask.unsqueeze(1).to(x.dtype)
        denom = m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=-1, keepdim=True) / denom
        var = ((x - mean) ** 2 * m).sum(dim=-1, keepdim=True) / denom
        std = torch.sqrt(var + self.eps)
        return mean, std


class Patching(nn.Module):
    """
    Converts [B, C, L] -> [B, C, N, patch_len]
    """

    def __init__(self, patch_len: int = 8, stride: int = 8):
        super().__init__()
        self.patch_len = int(patch_len)
        self.stride = int(stride)

    def forward(self, x: torch.Tensor):
        # x: [B, C, L]
        B, C, L = x.shape
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, N, patch_len]
        return patches


class PatchEmbedding(nn.Module):
    """
    Embeds patches into d_model.
    Input:  x_patches [B, C, N, patch_len]
    Output: embeds    [B, C, N, d_model]
    """

    def __init__(
        self,
        d_model: int,
        seq_len: int,
        patch_len: int,
        stride: int,
        dropout: float = 0.1,
        add_positional_embedding: bool = False,
        value_embedding_bias: bool = False,
        orth_gain: Optional[float] = 1.41,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.seq_len = int(seq_len)
        self.patch_len = int(patch_len)
        self.stride = int(stride)

        self.value_embedding = nn.Linear(self.patch_len, self.d_model, bias=value_embedding_bias)
        if orth_gain is not None:
            nn.init.orthogonal_(self.value_embedding.weight, gain=orth_gain)
            if self.value_embedding.bias is not None:
                self.value_embedding.bias.data.zero_()

        self.dropout = nn.Dropout(dropout)
        self.add_positional_embedding = bool(add_positional_embedding)

        # number of patches for positional embeddings
        n_patches = (max(self.seq_len, self.patch_len) - self.patch_len) // self.stride + 1
        if self.add_positional_embedding:
            self.pos_embedding = nn.Parameter(torch.zeros(1, 1, n_patches, self.d_model))
        else:
            self.register_parameter("pos_embedding", None)

    def forward(self, x_patches: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x_patches: [B, C, N, patch_len]
        mask: [B, L] or [B, N] (optional). In this simplified implementation we do not zero patches here.
        """
        emb = self.value_embedding(x_patches)  # [B, C, N, d_model]

        if self.add_positional_embedding and self.pos_embedding is not None:
            emb = emb + self.pos_embedding[:, :, : emb.shape[2], :]

        emb = self.dropout(emb)
        return emb
