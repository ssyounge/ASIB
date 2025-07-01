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
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

        # identity operations mimicking the real backbone methods
        self._norm = torch.nn.Identity()
        self._permute = torch.nn.Identity()
        self._flatten = torch.nn.Flatten(1)

    # simple wrappers so ``TeacherSwinWrapper`` can call them like the real model
    def norm(self, x):
        return self._norm(x)

    def permute(self, x):
        return self._permute(x)

    def flatten(self, x):
        return self._flatten(x)

def test_forward_basic():
    backbone = DummySwin()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    expected_f2d = backbone.flatten(backbone.avgpool(backbone.permute(backbone.norm(backbone.features(x)))))
    assert torch.allclose(out["feat_2d"], expected_f2d)
    # norm and permute are identities so this reduces to ``features(x)``
    expected_f4d = backbone.permute(backbone.norm(backbone.features(x)))
    assert torch.allclose(out["feat_4d"], expected_f4d)
