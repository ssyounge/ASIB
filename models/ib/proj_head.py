import torch.nn as nn

class StudentProj(nn.Module):
    def __init__(self, in_dim: int, z_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, z_dim)

    def forward(self, x):
        return self.proj(x.flatten(1))
