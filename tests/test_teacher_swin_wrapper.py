import pytest

torch = pytest.importorskip("torch")
from models.teachers.teacher_swin import TeacherSwinWrapper

class DummySwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)

    # mimic torchvision Swin interface returning a 4D feature map
    def features(self, x):
        return x

def test_forward_basic():
    backbone = DummySwin()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    expected_f2d = backbone.avgpool(x).flatten(1)
    assert torch.allclose(out["feat_2d"], expected_f2d)
    assert torch.allclose(out["feat_4d"], x)
