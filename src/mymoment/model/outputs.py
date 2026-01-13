from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class TimeseriesOutputs:
    """
    Standard output container used by MOMENT.
    Only fields you actually need for pretraining are included, plus a few optional ones.
    """
    input_mask: Optional[torch.Tensor] = None

    # Pretraining / reconstruction
    reconstruction: Optional[torch.Tensor] = None
    pretrain_mask: Optional[torch.Tensor] = None

    # Embeddings use-case (optional)
    embeddings: Optional[torch.Tensor] = None

    # Forecasting / anomaly (optional, included for compatibility)
    forecast: Optional[torch.Tensor] = None
    anomaly_scores: Optional[torch.Tensor] = None

    # Debug / metadata
    illegal_output: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
