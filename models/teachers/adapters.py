# models/teachers/adapters.py

import torch
import torch.nn as nn
from typing import Optional

class DistillationAdapter(nn.Module):
    """Simple MLP used to refine teacher features for distillation."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        if cfg is not None:
            hidden_dim = cfg.get("distill_hidden_dim", hidden_dim)
            out_dim = cfg.get("distill_out_dim", out_dim)

        # allow 0, negative, or None to trigger automatic dimension selection
        if hidden_dim is None or hidden_dim <= 0:
            hidden_dim = max(1, in_dim // 2)
        if out_dim is None or out_dim <= 0:
            out_dim = max(1, in_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
