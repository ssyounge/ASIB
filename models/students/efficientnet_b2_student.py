import torch
import torch.nn as nn
import timm

from models.common.base_wrapper import BaseKDModel, register


@register("efficientnet_b2_student")
class EfficientNetB2Student(BaseKDModel):
    """EfficientNet-B2 student with 1x1 conv adapter."""

    def __init__(
        self,
        *,
        num_classes: int = 100,
        pretrained: bool = True,
        small_input: bool = False,
        cfg: dict | None = None,
    ) -> None:
        backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.2,
        )

        if small_input:
            stem = backbone.conv_stem
            stem.stride = (1, 1)
            backbone.maxpool = nn.Identity()

        super().__init__(backbone, num_classes, role="student", cfg=cfg or {})

        self.adapter = nn.Conv2d(backbone.num_features, backbone.num_features, 1, bias=False)

    # ------------------------------------------------------------------
    def extract_feats(self, x):
        feat_4d = self.backbone.forward_features(x)
        feat_4d = self.adapter(feat_4d)
        feat_2d = feat_4d.mean([-2, -1])
        return feat_4d, feat_2d

