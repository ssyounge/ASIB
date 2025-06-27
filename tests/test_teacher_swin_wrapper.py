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


class TokenBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # expects pooled dim of 3
        self.head = torch.nn.Linear(3, 2)

    def forward_features(self, x):
        # return token tensor [N, L, C]
        return x


def test_forward_outputs_feature_keys():
    backbone = DummyBackbone()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    assert "feat_4d" in out
    assert "feat_2d" in out


def test_forward_accepts_token_tensor():
    token_backbone = TokenBackbone()
    wrapper = TeacherSwinWrapper(token_backbone)
    x = torch.randn(2, 4, 3)  # [N, L, C]

    out = wrapper(x)

    assert out["feat_2d"].shape[1] == 3

