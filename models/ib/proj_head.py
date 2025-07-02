# models/ib/proj_head.py

import torch.nn as nn
import torch.nn.functional as F

class StudentProj(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, normalize: bool = True, use_bn: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_dim, z_dim)
        self.normalize = normalize
        self.bn = nn.BatchNorm1d(z_dim, affine=False) if use_bn else nn.Identity()

    def forward(self, x):
        x = x.flatten(1)
        if self.normalize:
            x = F.normalize(x, dim=1)
        x = self.proj(x)
        x = self.bn(x)
        return x
