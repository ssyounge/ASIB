# models/teachers/teacher_efficientnet.py

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .adapters import DistillationAdapter
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class TeacherEfficientNetWrapper(nn.Module):
    """
    Teacher 모델(EfficientNet-B2) forward:
     => dict 반환 {"feat_4d", "feat_2d", "logit", "ce_loss"}
    feature_dict 예시:
      {
        "feat_4d": [N, 1408, H, W],   # backbone.features(x)
        "feat_2d": [N, 1408],         # global pooled
      }
    """
    def __init__(self, backbone, cfg: Optional[dict] = None):
        super().__init__()
        self.backbone = backbone
        self.criterion_ce = nn.CrossEntropyLoss()

        # 추가: EffNet-B2의 글로벌 피처 차원(1408)
        self.feat_dim = 1408
        self.feat_channels = 1408

        # distillation adapter
        cfg = cfg or {}
        hidden_dim = cfg.get("distill_hidden_dim")
        out_dim = cfg.get("distill_out_dim")
        self.distillation_adapter = DistillationAdapter(
            self.feat_dim, hidden_dim=hidden_dim, out_dim=out_dim
        )
        self.distill_dim = self.distillation_adapter.out_dim
    
    def forward(self, x, y=None):
        # 1) compute intermediate 4D features
        feat_layer1 = None
        feat_layer2 = None
        feat_layer3 = None
        out = x
        for idx, block in enumerate(self.backbone.features):
            out = block(out)
            if idx == 2:
                feat_layer1 = out
            elif idx == 4:
                feat_layer2 = out
            elif idx == 6:
                feat_layer3 = out
        f4d = out  # final feature map [N, 1408, h, w]

        # 2) final logits
        fpool = F.adaptive_avg_pool2d(f4d, (1,1)).flatten(1)  # [N, 1408]
        logit = self.backbone.classifier(fpool)
        # 3) feat_2d from pooled feature

        # distillation adapter feature
        distill_feat = self.distillation_adapter(fpool)

        # (optional) CE loss
        ce_loss = None
        if y is not None:
            ce_loss = self.criterion_ce(logit, y)

        # Dict로 묶어서 반환
        return {
            "feat_4d": f4d,       # [N, 1408, h, w]
            "feat_2d": fpool,     # [N, 1408]
            "distill_feat": distill_feat,
            "logit": logit,
            "ce_loss": ce_loss,
            "feat_4d_layer1": feat_layer1,
            "feat_4d_layer2": feat_layer2,
            "feat_4d_layer3": feat_layer3,
        }

    def get_feat_dim(self):
        """
        Returns the dimension of the 2D feature (feat_2d).
        EffNet-B2 => 1408
        """
        return self.feat_dim

    def get_feat_channels(self):
        """Channel dimension of the 4D feature."""
        return self.feat_channels

def create_efficientnet_b2(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    dropout_p: float = 0.3,
    cfg: Optional[dict] = None,
):
    """
    EfficientNet-B2를 로드한 뒤, (in_feats->num_classes) 교체
    small_input=True 시, CIFAR-100과 같은 작은 이미지에 맞게 stem stride를 1로 수정
    => TeacherEfficientNetWrapper로 감싸서 반환
    """
    if pretrained:
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b2(weights=None)

    if small_input:
        # 32x32 등 작은 입력에 맞게 conv_stem stride를 1로 수정
        # 기존 pretrained weight를 보존하기 위해 새 layer 생성 후 weight copy
        old_conv = model.features[0][0]
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        # pretrained weights 복사
        new_conv.weight.data.copy_(old_conv.weight.data)
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)

        model.features[0][0] = new_conv

    in_feats = model.classifier[1].in_features

    # classifier[0] is Dropout layer in torchvision implementation
    model.classifier[0] = nn.Dropout(p=dropout_p)
    model.classifier[1] = nn.Linear(in_feats, num_classes)

    teacher_model = TeacherEfficientNetWrapper(model, cfg=cfg)
    return teacher_model
