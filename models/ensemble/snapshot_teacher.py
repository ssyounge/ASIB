from __future__ import annotations
import torch, torch.nn as nn
from utils.model_factory import create_teacher_by_name

class SnapshotTeacher(nn.Module):
    def __init__(self,
                 ckpt_paths: list[str],
                 backbone_name: str = "resnet50",
                 n_cls: int = 100):
        super().__init__()
        self.models = nn.ModuleList()
        for p in ckpt_paths:
            m = create_teacher_by_name(backbone_name,
                                       num_classes=n_cls,
                                       pretrained=False)
            m.load_state_dict(torch.load(p, map_location="cpu"))
            m.eval()
            for param in m.parameters():
                param.requires_grad_(False)
            self.models.append(m)

    @torch.no_grad()
    def forward(self, x):
        """교사‑API 호환: (feat_dict, logits) tuple 반환"""
        logits_all, feat_all = [], []
        for m in self.models:
            feat_dict, logit = m(x)
            logits_all.append(logit)
            feat_all.append(feat_dict["feat_2d"])

        logit_avg = torch.stack(logits_all).mean(0)
        feat_avg = torch.stack(feat_all).mean(0)
        return {"feat_2d": feat_avg}, logit_avg
