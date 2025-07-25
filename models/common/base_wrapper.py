import torch
import torch.nn as nn
from typing import Dict, Tuple, Any

# -------------------------------- Registry ----------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {}

def register(name: str):
    def _wrap(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return _wrap

# ---------------------------   BaseKDModel  ---------------------------------
class BaseKDModel(nn.Module):
    """Teacher/Student 공통 래퍼 - forward 규격 단일화."""

    def __init__(self, backbone: nn.Module, num_classes: int, *, role: str, cfg: Dict):
        super().__init__()

        self.backbone = backbone
        self.role = role
        self.cfg = cfg or {}
        self.ce = nn.CrossEntropyLoss()

        feat_dim = getattr(backbone, "num_features", None) or getattr(backbone, "feat_dim", None)
        if feat_dim is None:
            raise ValueError("Backbone must expose 'num_features' or 'feat_dim'")
        self.feat_dim = feat_dim

        if self.cfg.get("use_distillation_adapter", False):
            hid = self.cfg.get("distill_hidden_dim", feat_dim // 2)
            out = self.cfg.get("distill_out_dim", feat_dim // 4)
            self.distill_adapter = nn.Sequential(
                nn.Linear(feat_dim, hid),
                nn.ReLU(inplace=True),
                nn.Linear(hid, out),
            )
            self.distill_dim = out
        else:
            self.distill_adapter = None
            self.distill_dim = feat_dim

        self.classifier = nn.Linear(feat_dim, num_classes)

    def extract_feats(self, x) -> Tuple[torch.Tensor | None, torch.Tensor]:
        """return (feat_4d or None, feat_2d)"""
        raise NotImplementedError

    def forward(self, x, y: torch.Tensor | None = None) -> Tuple[Dict, torch.Tensor, Dict[str, Any]]:
        f4d, f2d = self.extract_feats(x)
        distill = self.distill_adapter(f2d) if self.distill_adapter else None
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
