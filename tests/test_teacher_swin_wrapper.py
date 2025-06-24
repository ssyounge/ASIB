import pytest

torch = pytest.importorskip("torch")

from models.teachers.teacher_swin import TeacherSwinWrapper


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # simple linear head expecting pooled feature of dim 1
        self.head = torch.nn.Linear(1, 2)

    def features(self, x):
        # return input as 4D feature
        return x


def test_forward_outputs_feature_keys():
    backbone = DummyBackbone()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    assert "feat_4d" in out
    assert "feat_2d" in out

