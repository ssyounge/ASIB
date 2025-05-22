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
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)  # synergy embedding
        )

    def forward(self, feat1, feat2):
        # feat1, feat2: [N, d]  -> concat -> [N, d1+d2]
        x = torch.cat([feat1, feat2], dim=1)
        synergy_emb = self.mlp(x)  # [N, out_dim]

        # 'dict' 형태로 반환
        #  - synergy_emb만 저장 (필요하면 여기에 'logit' 등도 가능)
        synergy_dict = {
            "feat_2d": synergy_emb
        }
        return synergy_dict


class SynergyHead(nn.Module):
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, synergy_emb):
        # synergy_emb: [N, in_dim]
        logit = self.fc(synergy_emb)
        # dict로 반환할 수도, 바로 tensor로 반환해도 됨
        return logit
