import pytest

torch = pytest.importorskip("torch")

from eval import SynergyEnsemble
from models.la_mbm import LightweightAttnMBM
from models.mbm import SynergyHead


class DummyTeacher(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return {"feat_2d": torch.zeros(x.size(0), self.dim), "feat_4d": None}

    def get_feat_dim(self):
        return self.dim

    def get_feat_channels(self):
        return self.dim


class DummyStudent(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        feat = torch.zeros(x.size(0), self.dim)
        logit = torch.zeros(x.size(0), 10)
        return {"feat_2d": feat}, logit, None

    def get_feat_dim(self):
        return self.dim


def test_synergy_ensemble_forward_la():
    t1 = DummyTeacher(4)
    t2 = DummyTeacher(4)
    student = DummyStudent(4)
    mbm = LightweightAttnMBM([4, 4], out_dim=8, r=2, learnable_q=False, query_dim=4)
    head = SynergyHead(in_dim=8, num_classes=10)

    ensemble = SynergyEnsemble(
        t1,
        t2,
        mbm,
        head,
        student=student,
        cfg={"feat_kd_key": "feat_2d", "mbm_type": "LA"},
    )

    x = torch.randn(2, 3)
    out = ensemble(x)
    assert out.shape == (2, 10)
