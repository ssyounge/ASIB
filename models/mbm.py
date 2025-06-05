# models/mbm.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldBridgingModule(nn.Module):
    """
    2D 입력([N,d]) => MLP 융합 => (N, out_dim) 텐서
    (4D 기능은 제거)
    """
    def __init__(
        self,                 # ← 반드시 첫 번째 파라미터로 self 추가
        in_dim:      int,
        hidden_dim:  int,
        out_dim:     int,
        # in_ch_4d:  Optional[int] = None,  # 4D 경로를 쓰게 되면 추가
        # out_ch_4d: Optional[int] = None
    ):
        super().__init__()

        # ----- 2D MLP -----
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, feat1, feat2):
        # 항상 2D로 가정
        x = torch.cat([feat1, feat2], dim=1)  # (N, d1 + d2)
        synergy_2d = self.mlp(x)             # (N, out_dim)
        return synergy_2d

class SynergyHead(nn.Module):
    """
    2D 전용 Head: (N, in_dim) -> (N, num_classes)
    """
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, synergy_emb):
        # synergy_emb: (N, in_dim)
        return self.fc(synergy_emb)
