# models/teachers/adapters.py

import torch
import torch.nn as nn
from typing import Optional

class DistillationAdapter(nn.Module):
    """
    [Teacher-side] 채널 / 공간 해상도 보정을 위한 작은 어댑터.

    학생의 feature dimension에 맞추기 위한 용도로만 사용하며,
    학습 루프에서는 ``train_distill_adapter_only`` 옵션이 활성화될 때에만
    파라미터를 업데이트한다.
    """

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
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
