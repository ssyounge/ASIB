# models/ib/proj_head.py

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class StudentProj(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        normalize: bool = True,
        use_bn: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []

        if hidden_dim:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_dim))

        self.net = nn.Sequential(*layers)
        self._normalize = normalize

    def forward(self, x):
        x = x.flatten(1)
        z = self.net(x)
        # 기본은 L2‑정규화(dim=1). ‑‑normalize False 면 그대로 반환
        return F.normalize(z, dim=1) if self._normalize else z
