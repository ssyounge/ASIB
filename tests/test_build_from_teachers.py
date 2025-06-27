import pytest

torch = pytest.importorskip("torch")

from models.mbm import build_from_teachers, SynergyHead
from models.la_mbm import LightweightAttnMBM


class DummyTeacher(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def get_feat_dim(self):
        return self.dim

    def get_feat_channels(self):
        return self.dim


def test_build_from_teachers_la_auto_query_dim():
    teachers = [DummyTeacher(4), DummyTeacher(4)]
    cfg = {
        "mbm_type": "LA",
        "mbm_query_dim": 0,
        "mbm_out_dim": 8,
        "num_classes": 10,
        "mbm_learnable_q": False,
    }
    mbm, head = build_from_teachers(teachers, cfg, query_dim=4)
    assert isinstance(mbm, LightweightAttnMBM)
    assert isinstance(head, SynergyHead)

    q = torch.randn(2, 4)
    feats = [torch.randn(2, 4), torch.randn(2, 4)]
    out, _, _, _ = mbm(q, feats)
    logits = head(out)
    assert out.shape == (2, 8)
    assert logits.shape == (2, 10)
    assert mbm.q_proj.in_features == q.size(1)


def test_build_from_teachers_with_distillation_adapter():
    class ADummyTeacher(torch.nn.Module):
        def __init__(self, feat_dim, distill_dim):
            super().__init__()
            self.dim = feat_dim
            self.distill_dim = distill_dim

        def get_feat_dim(self):
            return self.dim

        def get_feat_channels(self):
            return self.dim

    teachers = [ADummyTeacher(8, 2), ADummyTeacher(8, 2)]
    cfg = {
        "mbm_type": "LA",
        "mbm_query_dim": 4,
        "mbm_out_dim": 8,
        "num_classes": 10,
        "mbm_learnable_q": False,
        "use_distillation_adapter": True,
    }
    mbm, _ = build_from_teachers(teachers, cfg, query_dim=4)
    assert mbm.k_proj[0].in_features == teachers[0].distill_dim
