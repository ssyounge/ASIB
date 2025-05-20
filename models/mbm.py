"""
models/mbm.py

Manifold Bridging Module (MBM):
 - Bridges two Teacher feature vectors (feat1, feat2) into a "synergy" embedding.
 - Optionally includes a small head (SynergyHead) for producing final logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldBridgingModule(nn.Module):
    """
    Example MBM that fuses teacher1_feat + teacher2_feat by simple MLP.
    in_dim = feat1_dim + feat2_dim
    hidden_dim, out_dim: user-chosen
    """
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat1, feat2):
        """
        feat1, feat2: shape [N, d]
        We'll concat them channel-wise => shape [N, d1 + d2]
        Then pass through MLP => synergy embedding [N, out_dim]
        """
        x = torch.cat([feat1, feat2], dim=1)  # [N, in_dim]
        out = self.mlp(x)
        return out


class SynergyHead(nn.Module):
    """
    Optional synergy head to map synergy embedding -> final logit.
    If we want to produce logits for classification from MBM output,
    we can attach a small linear layer here.
    """
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """
        x: synergy embedding [N, in_dim]
        Returns: [N, num_classes]
        """
        return self.fc(x)
