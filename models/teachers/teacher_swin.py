# models/teachers/teacher_swin.py

import torch
import torch.nn as nn
from typing import Optional
from .adapters import DistillationAdapter
from torchvision.models import swin_t, Swin_T_Weights

class TeacherSwinWrapper(nn.Module):
    """
    Teacher 모델(Swin Tiny) forward:
     => dict 반환 {"feat_4d", "feat_2d", "logit", "ce_loss"}
    feature_dict 예시:
      {
        "feat_4d": [N, C, H, W],  # backbone.forward_features(x) or unsqueezed
        "feat_2d": [N, C],       # global pooled or direct features
      }
    """
    def __init__(self, backbone: nn.Module, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # Swin Tiny의 head.in_features => 768 (기본)
        #   (모델마다 다를 수 있음)
        self.feat_dim = self.backbone.head.in_features
        self.feat_channels = self.feat_dim

        # distillation adapter
        cfg = cfg or {}
        hidden_dim = cfg.get("distill_hidden_dim")
        out_dim = cfg.get("distill_out_dim")
        self.distillation_adapter = DistillationAdapter(
            self.feat_dim, hidden_dim=hidden_dim, out_dim=out_dim
        )
        self.distill_dim = self.distillation_adapter.out_dim

    
    def forward(self, x, y=None):
        """Standard forward pass for a Swin Transformer teacher."""

        # 1. Swin 모델의 백본 특징 추출 모듈을 직접 호출합니다.
        #    이것이 torchvision Swin 모델의 권장 방식입니다.
        x_features = self.backbone.features(x)

        # 2. 3D 텐서를 2D 벡터로 올바르게 변환합니다.
        #    norm 레이어를 거친 뒤 [N, L, C] -> [N, C, L] 형태로 바꾼 뒤
        #    avgpool을 적용해 최종 벡터를 얻습니다.
        x_features = self.backbone.norm(x_features)
        x_features = x_features.permute(0, 2, 1)
        f2d = self.backbone.avgpool(x_features)
        f2d = torch.flatten(f2d, 1)

        # 3. 어댑터와 헤드에 전달합니다.
        distill_feat = self.distillation_adapter(f2d)
        logit = self.backbone.head(f2d)

        # (선택적) CE 손실 계산
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # 호환성 유지를 위한 더미 4D 특징맵 생성
        dummy_4d = f2d.unsqueeze(-1).unsqueeze(-1)
        return {
            "feat_4d": dummy_4d,
            "feat_2d": f2d,
            "distill_feat": distill_feat,
            "logit": logit,
            "ce_loss": ce_loss,
            "feat_4d_layer1": dummy_4d,
            "feat_4d_layer2": dummy_4d,
            "feat_4d_layer3": dummy_4d,
        }
        
    def get_feat_dim(self):
        """
        Swin Tiny => usually 768
        """
        return self.feat_dim

    def get_feat_channels(self):
        """Channel dimension of the 4D feature."""
        return self.feat_channels

def create_swin_t(
    num_classes=100,
    pretrained=True,
    cfg: Optional[dict] = None,
):
    """
    Swin Tiny 로드 후, head 교체 => TeacherSwinWrapper
    => (feature_dict, logit, ce_loss)
    """
    if pretrained:
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    else:
        model = swin_t(weights=None)

    in_ch = model.head.in_features
    model.head = nn.Linear(in_ch, num_classes)

    teacher_model = TeacherSwinWrapper(model, cfg=cfg)
    return teacher_model
