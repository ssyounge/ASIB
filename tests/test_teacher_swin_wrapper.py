import pytest

torch = pytest.importorskip("torch")
from models.teachers.teacher_swin import TeacherSwinWrapper

class DummySwin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)
        self.norm = torch.nn.Identity()

    # mimic torchvision Swin interface
    def features(self, x):
        b, c, h, w = x.shape
        return x.view(b, h * w, 1)

def test_forward_basic():
    backbone = DummySwin()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    expected_f2d = x.view(2, -1, 1).mean(dim=1)
    assert torch.allclose(out["feat_2d"], expected_f2d)
    assert out["feat_4d"].shape == (*expected_f2d.shape, 1, 1)
