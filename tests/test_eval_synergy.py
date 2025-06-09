import pytest

torch = pytest.importorskip("torch")

from modules.trainer_teacher import eval_synergy
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


def test_eval_synergy_la_mode_runs():
    t1 = DummyTeacher(4)
    t2 = DummyTeacher(4)
    student = DummyStudent(4)
    mbm = LightweightAttnMBM([4, 4], out_dim=8, r=2, query_dim=4)
    head = SynergyHead(in_dim=8, num_classes=10)
    loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]
    cfg = {"feat_kd_key": "feat_2d", "mbm_type": "LA"}
    acc = eval_synergy([t1, t2], mbm, head, loader, device="cpu", cfg=cfg, student_model=student)
    assert isinstance(acc, float)
