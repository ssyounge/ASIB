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


class Dummy2DBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)

    def forward_features(self, x):
        # return a pooled 2D feature
        return x.mean(dim=(2, 3))


class Dummy3DBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)

    def forward_features(self, x):
        b, c, h, w = x.shape
        return x.view(b, -1, 1)


class Dummy3DBackboneChannelFirst(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(1, 2)

    def forward_features(self, x):
        b, c, h, w = x.shape
        return x.view(b, 1, -1)


def test_forward_outputs_feature_keys():
    backbone = DummyBackbone()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    assert "feat_4d" in out
    assert "feat_2d" in out


def test_forward_accepts_2d_output():
    backbone = Dummy2DBackbone()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    assert out["feat_4d"].dim() == 4
    expected_f2d = x.mean(dim=(2, 3))
    assert torch.allclose(out["feat_2d"], expected_f2d)
    assert out["feat_4d"].shape == (*expected_f2d.shape, 1, 1)


def test_forward_accepts_3d_output():
    backbone = Dummy3DBackbone()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    expected_f2d = x.view(2, -1, 1).mean(dim=1)
    assert torch.allclose(out["feat_2d"], expected_f2d)
    assert out["feat_4d"].shape == (*expected_f2d.shape, 1, 1)


def test_forward_accepts_3d_channel_first_output():
    backbone = Dummy3DBackboneChannelFirst()
    wrapper = TeacherSwinWrapper(backbone)
    x = torch.randn(2, 1, 2, 2)

    out = wrapper(x)

    expected_f2d = x.view(2, 1, -1).mean(dim=2)
    assert torch.allclose(out["feat_2d"], expected_f2d)
    assert out["feat_4d"].shape == (*expected_f2d.shape, 1, 1)

