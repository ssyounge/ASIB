# models/teachers/efficientnet_l2_teacher.py

import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import timm
from models.common.base_wrapper import BaseKDModel, register


@register("efficientnet_l2_teacher")
class EfficientNetL2Teacher(BaseKDModel):
    """EfficientNet-L2 (Noisy Student) Teacher."""

    def __init__(
        self, backbone: nn.Module, *, num_classes: int, cfg: Optional[dict] = None
    ):
        super().__init__(backbone, num_classes, role="teacher", cfg=cfg or {})

    def extract_feats(self, x):
        f4d = self.backbone.forward_features(x)
        f2d = F.adaptive_avg_pool2d(f4d, 1).flatten(1)
        return f4d, f2d


def create_efficientnet_l2(
    num_classes: int = 100,
    pretrained: bool = True,
    small_input: bool = False,
    dropout_p: float = 0.4,
    use_checkpointing: bool = False,
    cfg: Optional[dict] = None,
):
    """Create an EfficientNet-L2 (Noisy Student) teacher via timm."""
    model_name = "tf_efficientnet_l2_ns"
    backbone = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout_p,
    )

    # ------------------------------------------------------------
    # (선택) gradient-checkpointing 활성화
    #  - timm 0.9.x  : timm.models.helpers.checkpoint_seq
    #  - timm 1.x    : timm.layers.helpers.checkpoint_seq
    #  - timm ≥0.9.12 일부 릴리스에서 경로가 바뀌므로 다단계 fallback
    # ------------------------------------------------------------
    if use_checkpointing:
        import importlib, inspect

        checkpoint_seq = None
        CANDIDATES = (
            "timm.models.helpers",   # timm 0.9.x / 1.x (모듈당-checkpoint 버전)
            "timm.layers.helpers",   # timm 1.x 일부 릴리스
            "timm.layers",           # timm 1.x 경로 변형
        )

        for _path in CANDIDATES:
            try:
                mod = importlib.import_module(_path)
                fn = getattr(mod, "checkpoint_seq", None)
                if fn is None:
                    continue

                # ── 시그니처 검사 ──
                sig = inspect.signature(fn)
                # 필수 positional-only / positional-or-keyword 인자
                req = [
                    p for p in sig.parameters.values()
                    if p.default is p.empty
                    and p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                if len(req) == 1:  # <modules> 하나만 필수 → 사용 가능
                    checkpoint_seq = fn
                    break
            except ImportError:
                continue

        if checkpoint_seq is not None:
            backbone.blocks = checkpoint_seq(backbone.blocks)   # ✅
        else:
            import warnings
            warnings.warn(
                "[Eff-L2] ‘checkpoint_seq’ 시그니처를 만족하는 버전을 찾지 못했습니다. "
                "gradient-checkpointing을 비활성화합니다.",
                RuntimeWarning,
            )

    if small_input:
        stem = backbone.conv_stem
        new_stem = nn.Conv2d(
            stem.in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=1,
            padding=stem.padding,
            bias=stem.bias is not None,
        )
        new_stem.weight.data.copy_(stem.weight.data)
        if stem.bias is not None:
            new_stem.bias.data.copy_(stem.bias.data)
        backbone.conv_stem = new_stem

    return EfficientNetL2Teacher(backbone, num_classes=num_classes, cfg=cfg)
