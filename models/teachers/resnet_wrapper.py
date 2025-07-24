import torch.nn as nn

class ResNetWrapper(nn.Module):
    def __init__(self, backbone, num_classes, cfg):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.out_features, num_classes)
        # NEW: distillation adapter
        from models.teachers.adapters import DistillationAdapter
        self.distillation_adapter = DistillationAdapter(
            in_dim=backbone.out_features,
            out_dim=cfg.get("t_feat_dim", 512),
            cfg=cfg,
        )
        print(
            f"[ResNetWrapper] - add distillation_adapter "
            f"(in={backbone.out_features}, out={self.distillation_adapter.out_dim})"
        )

    def forward(self, x):
        feat = self.backbone(x)
        distill_feat = self.distillation_adapter(feat)
        logit = self.fc(feat)
        return {
            "feat_2d": feat,
            "distill_feat": distill_feat,
            "logit": logit,
        }
