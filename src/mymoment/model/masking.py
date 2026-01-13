from typing import Optional
import torch


class Masking:
    """
    Mask convention:
      1 = observed/keep (visible context)
      0 = masked/remove (target to predict)

    RmGPT-style objective:
      Mask the LAST *observed* patch and predict it from preceding patches.
    """

    def __init__(self, mask_ratio: float = 0.125, patch_len: int = 256, stride: Optional[int] = None):
        self.mask_ratio = float(mask_ratio)  # Kept for compatibility; ignored in 'last patch' logic
        self.patch_len = int(patch_len)
        self.stride = int(patch_len if stride is None else stride)

    @staticmethod
    def convert_seq_to_patch_view(mask: torch.Tensor, patch_len: int = 256, stride: Optional[int] = None):
        """
        Input:
            mask : [B, L] (0/1)
        Output:
            mask : [B, N] (0/1), where a patch is 1 only if ALL points are observed
        """
        stride = patch_len if stride is None else stride
        m = mask.unfold(dimension=-1, size=patch_len, step=stride)  # [B, N, patch_len]
        return (m.sum(dim=-1) == patch_len).long()

    @staticmethod
    def convert_patch_to_seq_view(mask: torch.Tensor, patch_len: int = 256):
        """
        Input:
            mask : [B, N]
        Output:
            mask : [B, N*patch_len]
        """
        return mask.repeat_interleave(patch_len, dim=-1)

    def generate_mask(self, x: torch.Tensor, input_mask: Optional[torch.Tensor] = None):
        """
        Returns:
          seq mask [B, L] where 1 = keep/context, 0 = target region (last observed patch)
        """
        if x.ndim == 3:  # [B, C, L]
            B, C, L = x.shape
            # Derive N from input_mask if available; otherwise assume full length
            if input_mask is None:
                input_mask = torch.ones((B, L), device=x.device, dtype=torch.long)

            patch_view = self.convert_seq_to_patch_view(input_mask, self.patch_len, self.stride)  # [B, N]
            keep_patch = self._mask_last_observed_patch(patch_view)  # [B, N]
            return self.convert_patch_to_seq_view(keep_patch, self.patch_len).long()

        elif x.ndim == 4:  # [B, C, N, patch_len]
            B, C, N, P = x.shape
            if input_mask is None:
                # Assume all observed
                patch_view = torch.ones((B, N), device=x.device, dtype=torch.long)
            else:
                patch_view = self.convert_seq_to_patch_view(input_mask, self.patch_len, self.stride).to(x.device)

            keep_patch = self._mask_last_observed_patch(patch_view)
            return self.convert_patch_to_seq_view(keep_patch, self.patch_len).long()

        else:
            raise ValueError(f"Unsupported x.ndim={x.ndim}")

    @staticmethod
    def _mask_last_observed_patch(input_mask_patch: torch.Tensor) -> torch.Tensor:
        """
        input_mask_patch: [B, N] (1 observed patch, 0 padded patch)
        returns keep_mask_patch: [B, N] (1 keep/context, 0 target)
        """
        B, N = input_mask_patch.shape
        keep = torch.ones((B, N), device=input_mask_patch.device, dtype=torch.long)

        # For each sample, find last observed patch index
        for b in range(B):
            observed = torch.where(input_mask_patch[b] == 1)[0]
            if observed.numel() > 0:
                last_idx = int(observed[-1].item())
                keep[b, last_idx] = 0  # Mask the target patch

        # Note: Padded patches (input_mask=0) remain '1' in keep mask here, 
        # but they will be zeroed out later by the loss calculation (target_mask = (1-keep) * input_mask).
        return keep