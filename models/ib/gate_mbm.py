import torch
import torch.nn as nn
import torch.nn.functional as F

class GateMBM(nn.Module):
    def __init__(self, c_in1: int, c_in2: int, z_dim: int = 512, n_cls: int = 100, beta: float = 1e-3):
        super().__init__()
        c = min(c_in1, c_in2)
        self.proj1 = nn.Conv2d(c_in1, c, 1)
        self.proj2 = nn.Conv2d(c_in2, c, 1)
        self.gate = nn.Parameter(torch.zeros(1, c, 1, 1))  # 0 -> 0.5 after sigmoid
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mu = nn.Linear(c, z_dim)
        self.log = nn.Linear(c, z_dim)
        self.head = nn.Linear(z_dim, n_cls)
        self.beta = beta

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, log_kl: bool = False):
        f1 = self.proj1(f1)
        f2 = self.proj2(f2)
        g = torch.sigmoid(self.gate)
        fused = g * f1 + (1 - g) * f2
        v = self.pool(fused).flatten(1)
        mu = self.mu(v)
        log = self.log(v)
        z = mu + torch.randn_like(mu) * (0.5 * log).exp()
        kl = -0.5 * (1 + log - mu.pow(2) - log.exp()).mean()
        out = self.head(z)
        return z, out, kl, None
