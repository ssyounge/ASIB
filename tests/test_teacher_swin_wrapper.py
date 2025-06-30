import pytest

torch = pytest.importorskip("torch")
from models.teachers.teacher_swin import TeacherSwinWrapper

class DummySwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)
        self.norm = torch.nn.Identity()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)

    # mimic torchvision Swin interface
    def features(self, x):
        b, c, h, w = x.shape
        return x.view(b, h * w, 1)

def test_forward_basic():
    backbone = DummySwin()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    x_feat = x.view(2, -1, 1)
    x_feat = backbone.norm(x_feat)
    x_feat = x_feat.permute(0, 2, 1)
    expected_f2d = backbone.avgpool(x_feat).flatten(1)
    assert torch.allclose(out["feat_2d"], expected_f2d)
    assert out["feat_4d"].shape == (*expected_f2d.shape, 1, 1)
