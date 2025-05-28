# models/mbm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldBridgingModule(nn.Module):
    """
    - 2D 입력([N,d]) => MLP 융합 => (N, out_dim_2d) 텐서 반환
    - 4D 입력([N,C,H,W]) => 1x1 Conv 융합 => (N, out_channels_4d, H, W) 텐서 반환
    - 딕셔너리를 반환하지 않고, 순수 텐서만 반환
    """

    def __init__(
        # (A) 2D MLP 설정
        in_dim_2d=None,
        hidden_dim_2d=None,
        out_dim_2d=None,
        # (B) 4D Conv 설정
        c1_4d=None,
        c2_4d=None,
        out_channels_4d=None
    ):
        super().__init__()

        # ----- 2D MLP -----
        self.use_mlp = False
        if (in_dim_2d is not None) and (hidden_dim_2d is not None) and (out_dim_2d is not None):
            self.use_mlp = True
            self.mlp = nn.Sequential(
                nn.Linear(in_dim_2d, hidden_dim_2d),
                nn.ReLU(),
                nn.Linear(hidden_dim_2d, hidden_dim_2d),
                nn.ReLU(),
                nn.Linear(hidden_dim_2d, out_dim_2d)
            )

        # ----- 4D Conv -----
        self.use_conv = False
        if (c1_4d is not None) and (c2_4d is not None) and (out_channels_4d is not None):
            self.use_conv = True
            in_ch = c1_4d + c2_4d
            self.conv1 = nn.Conv2d(in_ch, out_channels_4d, kernel_size=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_channels_4d)
            self.relu  = nn.ReLU(inplace=True)

    def forward(self, feat1, feat2):
        """
        feat1, feat2: 이미 텐서여야 함 (2D or 4D).
        => 2D path: MLP
        => 4D path: Conv
        => 반환도 텐서.
        """

        # 2D 입력
        if feat1.dim() == 2 and feat2.dim() == 2:
            if not self.use_mlp:
                raise ValueError("MBM: got 2D input but MLP not configured!")
            x = torch.cat([feat1, feat2], dim=1)  # => (N, d1+d2)
            synergy_2d = self.mlp(x)             # => (N, out_dim_2d)
            return synergy_2d  # \textbf{순수 텐서} 반환

        # 4D 입력
        elif feat1.dim() == 4 and feat2.dim() == 4:
            if not self.use_conv:
                raise ValueError("MBM: got 4D input but Conv not configured!")
            x = torch.cat([feat1, feat2], dim=1)  # => (N, c1+c2, H, W)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            # => (N, out_channels_4d, H, W)
            return x  # \textbf{순수 텐서} 반환

        else:
            raise ValueError(
                f"MBM forward mismatch: "
                f"feat1.shape={feat1.shape}, feat2.shape={feat2.shape}"
            )

class SynergyHead(nn.Module):
    """
    2D 전용 Head: (N, in_dim) -> (N, num_classes)
    (4D 전용이라면 별도 클래스를 만들거나, 여기서 분기)
    """
    def __init__(self, in_dim, num_classes=100):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, synergy_emb):
        # synergy_emb: (N, in_dim)
        return self.fc(synergy_emb)
