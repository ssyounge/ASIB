from __future__ import annotations
import torch
import torch.nn as nn
# NOTE: circular-import 해결을 위해
#       create_teacher_by_name 는 __init__ 안에서 늦게 import 합니다.

class SnapshotTeacher(nn.Module):
    def __init__(self,
                 ckpt_paths: list[str],
                 backbone_name: str = "resnet50",
                 n_cls: int = 100):
        super().__init__()
        from utils.model_factory import create_teacher_by_name  # delayed import

        self.models = nn.ModuleList()
        for p in ckpt_paths:
            m = create_teacher_by_name(
                backbone_name,
                num_classes=n_cls,
                pretrained=False,
            )
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            for param in m.parameters():
                param.requires_grad_(False)
            self.models.append(m)

    @torch.no_grad()
    def forward(self, x, return_feat: bool = False):
        """Return averaged logits and, optionally, features."""
        logits_all, feat_all = [], []
        for m in self.models:
            out = m(x)
            feat = out[0]["feat_2d"] if isinstance(out, tuple) else None
            log = out[1] if isinstance(out, tuple) else out
            logits_all.append(log)
            feat_all.append(feat)

        logit_avg = torch.stack(logits_all).mean(0)
        feat_avg = (
            torch.stack(feat_all).mean(0) if feat_all[0] is not None else None
        )
        return {"feat_2d": feat_avg}, logit_avg

    @property
    def backbone(self):
        """Expose backbone of the first teacher for hook registration."""
        return self.models[0].backbone

    def get_feat_dim(self) -> int:
        return self.models[0].get_feat_dim()
