# models/ib/proj_head.py

import torch.nn as nn
import torch.nn.functional as F

class StudentProj(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, normalize: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, z_dim)
        self.normalize = normalize

    def forward(self, x):
        x = x.flatten(1)
        if self.normalize:
            x = F.normalize(x, dim=1)
        return self.proj(x)
