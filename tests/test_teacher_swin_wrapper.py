import pytest

torch = pytest.importorskip("torch")
from models.teachers.teacher_swin import TeacherSwinWrapper

class _Add(torch.nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return x + self.val


class DummySwin(torch.nn.Module):
    """Minimal Swin-like backbone exposing the required modules."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)

        # modules used in the official forward sequence
        self.features = _Add(1)
        self.norm = _Add(2)
        self.permute = _Add(3)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.flatten = torch.nn.Flatten(1)

def test_forward_basic():
    backbone = DummySwin()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    expected_f2d = backbone.flatten(backbone.avgpool(backbone.permute(backbone.norm(backbone.features(x)))))
    assert torch.allclose(out["feat_2d"], expected_f2d)
    # features -> norm -> permute produces x + 6 with this dummy
    expected_f4d = backbone.permute(backbone.norm(backbone.features(x)))
    assert torch.allclose(out["feat_4d"], expected_f4d)
