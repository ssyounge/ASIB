# models/common/base_wrapper.py

import torch
import torch.nn as nn
# typing …
from typing import Dict, Tuple, Any, Optional, Union

# ------------------------------------------------------------------
# ❶ registry 얇게 바인딩 ‑ 순환 임포트 회피용
#     (scan 은 나중에 필요할 때 ensure_scanned() 로 수행)
# ------------------------------------------------------------------
# registry 객체는 즉시 쓰지만 **스캔은 하지 않는다**.
from models.common import registry as _reg
register       = _reg.register
MODEL_REGISTRY = _reg.MODEL_REGISTRY

# ---------------------------   BaseKDModel  ---------------------------------
class BaseKDModel(nn.Module):
    """Teacher/Student 공통 래퍼 - forward 규격 단일화."""

    def __init__(self, backbone: nn.Module, num_classes: int, *, role: str, cfg: Dict):
        super().__init__()

        self.backbone = backbone
        self.role = role
        self.cfg = cfg or {}
        self.ce = nn.CrossEntropyLoss()

        # ------------------------------------------------------------
        # Robust feature-dim inference (tv-ResNet, timm, custom …)
        # ------------------------------------------------------------
        feat_dim = (
            getattr(backbone, "num_features", None)
            or getattr(backbone, "feat_dim", None)
            or (backbone.fc.in_features if hasattr(backbone, "fc") else None)
            or (backbone.classifier[1].in_features if hasattr(backbone, "classifier") and hasattr(backbone.classifier, "__getitem__") else None)
        )
        if feat_dim is None:
            raise ValueError(
                "Unable to infer feature dimension – "
                "backbone must expose 'num_features', 'feat_dim', '.fc.in_features' or '.classifier[1].in_features'."
            )
        self.feat_dim = feat_dim

        if self.cfg.get("use_distillation_adapter", False):
            hid = self.cfg.get("distill_hidden_dim", feat_dim // 2)
            out = self.cfg.get("distill_out_dim", feat_dim // 4)
            self.distillation_adapter = nn.Sequential(
                nn.Linear(feat_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, out),
            )
            # legacy alias (외부 코드 호환)
            self.distill_adapter = self.distillation_adapter
            self.distill_dim = out
        else:
            self.distillation_adapter = None
            self.distill_adapter = None
            self.distill_dim = feat_dim

        self.classifier = nn.Linear(feat_dim, num_classes)

    def extract_feats(self, x) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """return (feat_4d or None, feat_2d)"""
        # Simple implementation for testing
        feat_2d = self.backbone(x)
        if isinstance(feat_2d, torch.Tensor):
            return None, feat_2d
        else:
            # If backbone returns a tuple/list, take the last element as feat_2d
            return None, feat_2d[-1] if isinstance(feat_2d, (tuple, list)) else feat_2d

    def forward(self, x, y: Optional[torch.Tensor] = None) -> Tuple[Dict, torch.Tensor, Dict[str, Any]]:
        f4d, f2d = self.extract_feats(x)
        distill = self.distillation_adapter(f2d) if self.distillation_adapter else None
        logit = self.classifier(f2d)

        aux: Dict[str, Any] = {}
        if y is not None and self.role == "student":
            aux["ce_loss"] = self.ce(logit, y)

        feat_dict = {
            "feat_4d": f4d,
            "feat_2d": f2d,
            "distill_feat": distill,
            "logit": logit,
        }
        return feat_dict, logit, aux

    def get_feat_dim(self):
        return self.feat_dim

    def get_feat_channels(self):
        return self.feat_dim


# ------------------------------------------------------------------
# 레지스트리 등록은 registry 모듈 import 시 구성 파일을 통해 수행됩니다.
# ------------------------------------------------------------------

