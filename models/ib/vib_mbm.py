import torch
import torch.nn as nn
import torch.nn.functional as F

class VIB_MBM(nn.Module):
    def __init__(self, in_dim1: int, in_dim2: int, z_dim: int, n_cls: int):
        super().__init__()
        self.fc_mu = nn.Linear(in_dim1 + in_dim2, z_dim)
        self.fc_log = nn.Linear(in_dim1 + in_dim2, z_dim)
        self.cls = nn.Linear(z_dim, n_cls)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor):
        h = torch.cat([f1.flatten(1), f2.flatten(1)], dim=1)
        mu = self.fc_mu(h)
        log = self.fc_log(h).clamp(-5, 5)
        std = torch.exp(0.5 * log)
        eps = torch.randn_like(std)
        z = mu + eps * std

        logits = self.cls(z)

        kl_z = -0.5 * torch.sum(1 + log - mu.pow(2) - log.exp(), dim=1)

        return z, logits, kl_z, mu
