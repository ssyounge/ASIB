# models/mbm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .la_mbm import LightweightAttnMBM
from .ib import VIB_MBM

class ManifoldBridgingModule(nn.Module):
    """Fuses teacher features using optional MLP, convolutional and attention paths."""

    def __init__(
        self,
        feat_dims: List[int],
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        use_4d: bool = False,
        in_ch_4d: Optional[int] = None,
        out_ch_4d: Optional[int] = None,
        attn_heads: int = 0,
    ) -> None:
        super().__init__()
        self.use_2d = True  # always available
        self.use_4d = use_4d and in_ch_4d is not None and out_ch_4d is not None
        self.use_attn = attn_heads > 0

        in_dim = sum(feat_dims)

        if self.use_2d:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

        if self.use_4d:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch_4d, out_ch_4d, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch_4d, out_ch_4d, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            self.conv_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.conv_fc = nn.Linear(out_ch_4d, out_dim)

        if self.use_attn:
            self.attn_proj = nn.ModuleList([nn.Linear(d, out_dim) for d in feat_dims])
            self.attn = nn.MultiheadAttention(out_dim, attn_heads, batch_first=True)

    def forward(
        self,
        feats_2d: List[torch.Tensor],
        feats_4d: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        outputs = []

        if self.use_2d:
            cat = torch.cat(feats_2d, dim=1)
            outputs.append(self.mlp(cat))

        if self.use_4d and feats_4d is not None:
            cat4d = torch.cat(feats_4d, dim=1)
            x = self.conv(cat4d)
            x = self.conv_pool(x).flatten(1)
            outputs.append(self.conv_fc(x))

        if self.use_attn:
            tokens = [proj(f).unsqueeze(1) for f, proj in zip(feats_2d, self.attn_proj)]
            tokens = torch.cat(tokens, dim=1)
            attn_out, _ = self.attn(tokens, tokens, tokens)
            outputs.append(attn_out.mean(dim=1))

        assert len(outputs) > 0, "No active MBM paths"
        if len(outputs) == 1:
            return outputs[0]
        return sum(outputs) / len(outputs)

class SynergyHead(nn.Sequential):
    """FC-ReLU-(Dropout)-FC mapping from synergy embedding to logits."""

    def __init__(self, in_dim: int, num_classes: int = 100, p: float = 0.0) -> None:
        super().__init__(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(in_dim, num_classes),
        )

def build_from_teachers(
    teachers: List[nn.Module],
    cfg: dict,
    query_dim: Optional[int] = None,
) -> Tuple[ManifoldBridgingModule, SynergyHead]:
    use_da = bool(cfg.get("use_distillation_adapter", False))
    feat_dims = []
    for t in teachers:
        if use_da and hasattr(t, "distill_dim"):
            feat_dims.append(getattr(t, "distill_dim"))
        else:
            feat_dims.append(t.get_feat_dim())
    in_dim = sum(feat_dims)

    use_4d = bool(cfg.get("mbm_use_4d", False))
    if use_4d:
        channels = [t.get_feat_channels() for t in teachers]
        in_ch_4d = sum(channels)
        out_ch_4d = cfg.get("mbm_out_ch_4d", cfg.get("mbm_out_dim", 256))
    else:
        in_ch_4d = out_ch_4d = None

    mbm_type = cfg.get("mbm_type", "MLP")
    if mbm_type == "LA":
        qdim = cfg.get("mbm_query_dim")
        if qdim is None or qdim <= 0:
            qdim = query_dim
        if qdim is not None and qdim <= 0:
            qdim = None
        mbm = LightweightAttnMBM(
            feat_dims=feat_dims,
            out_dim=cfg.get("mbm_out_dim", 512),
            r=cfg.get("mbm_r", 4),
            n_head=cfg.get("mbm_n_head", 1),
            learnable_q=cfg.get("mbm_learnable_q", False),
            query_dim=qdim,
        )
        head = SynergyHead(
            in_dim=cfg.get("mbm_out_dim", 512),
            num_classes=cfg.get("num_classes", 100),
            p=cfg.get("synergy_head_dropout", cfg.get("mbm_dropout", 0.0)),
        )
    elif mbm_type == "VIB":
        in1 = teachers[0].get_feat_dim()
        in2 = teachers[1].get_feat_dim()
        zdim = cfg.get("z_dim", 256)
        mbm = VIB_MBM(in1, in2, zdim, cfg.get("num_classes", 100))
        head = nn.Identity()
    else:
        mbm = ManifoldBridgingModule(
            feat_dims=feat_dims,
            hidden_dim=cfg.get("mbm_hidden_dim", 512),
            out_dim=cfg.get("mbm_out_dim", 512),
            dropout=cfg.get("mbm_dropout", 0.0),
            use_4d=use_4d,
            in_ch_4d=in_ch_4d,
            out_ch_4d=out_ch_4d,
            attn_heads=int(cfg.get("mbm_attn_heads", 0)),
        )
        head = SynergyHead(
            in_dim=cfg.get("mbm_out_dim", 512),
            num_classes=cfg.get("num_classes", 100),
            p=cfg.get("synergy_head_dropout", cfg.get("mbm_dropout", 0.0)),
        )
    return mbm, head
