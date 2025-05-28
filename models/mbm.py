# models/mbm.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ManifoldBridgingModule(nn.Module):
    """
    통합 MBM 예시:
      - (A) 2D 입력([N,d]) => MLP 융합
      - (B) 4D 입력([N,C,H,W]) => 1x1 Conv 융합
    """

    def __init__(
        self,
        # 2D MLP 설정
        in_dim_2d=None,
        hidden_dim_2d=None,
        out_dim_2d=None,

        # 4D Conv 설정
        c1_4d=None,
        c2_4d=None,
        out_channels_4d=None
    ):
        super().__init__()

        # ----- 2D용 MLP 구성 -----
        # 예: in_dim_2d = d1+d2, hidden_dim_2d=128, out_dim_2d=64
        self.use_mlp = False
        if in_dim_2d is not None and hidden_dim_2d is not None and out_dim_2d is not None:
            self.use_mlp = True
            self.mlp = nn.Sequential(
                nn.Linear(in_dim_2d, hidden_dim_2d),
                nn.ReLU(),
                nn.Linear(hidden_dim_2d, hidden_dim_2d),
                nn.ReLU(),
                nn.Linear(hidden_dim_2d, out_dim_2d)  # synergy embedding
            )

        # ----- 4D용 Conv 구성 -----
        # 예: c1_4d=채널1, c2_4d=채널2, out_channels_4d=512
        self.use_conv = False
        if c1_4d is not None and c2_4d is not None and out_channels_4d is not None:
            self.use_conv = True
            in_ch = c1_4d + c2_4d
            self.conv1 = nn.Conv2d(in_ch, out_channels_4d, kernel_size=1, bias=False)
            self.bn1   = nn.BatchNorm2d(out_channels_4d)
            self.relu  = nn.ReLU(inplace=True)

        # 혹은 더 많은 레이어(Residual 등)를 추가할 수 있음.

    def forward(self, feat1, feat2):
        """
        feat1과 feat2가 2D인지 4D인지에 따라 처리 경로 분기
        """
        if feat1.dim() == 2 and feat2.dim() == 2:
            # ------------- 2D path -------------
            # (N, d1), (N, d2) -> concat (N, d1+d2)
            if not self.use_mlp:
                raise ValueError("MBM: 2D input detected but MLP is not configured!")
            x = torch.cat([feat1, feat2], dim=1)
            synergy_emb = self.mlp(x)  # [N, out_dim_2d]
            return {"feat_2d": synergy_emb}

        elif feat1.dim() == 4 and feat2.dim() == 4:
            # ------------- 4D path -------------
            # (N, c1, H, W), (N, c2, H, W) -> concat channel
            if not self.use_conv:
                raise ValueError("MBM: 4D input detected but Conv is not configured!")
            x = torch.cat([feat1, feat2], dim=1)  # => [N, c1+c2, H, W]
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            # 시너지 4D 맵
            return {"feat_4d": x}

        else:
            raise ValueError(
                "MBM forward: feat1/feat2 dimension mismatch or unsupported shape. "
                f"Got feat1.shape={feat1.shape}, feat2.shape={feat2.shape}"
            )

class SynergyHead(nn.Module):
    def __init__(self, in_dim, num_classes=100):
        """
        2D 시너지 Embedding => 로짓
        주의: 만약 4D를 바로 로짓으로 연결하고 싶으면 또 다른 설계가 필요!
        """
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, synergy_emb):
        # synergy_emb: [N, in_dim]
        logit = self.fc(synergy_emb)
        return logit
