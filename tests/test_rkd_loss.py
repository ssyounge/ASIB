import pytest

torch = pytest.importorskip("torch")

from modules.losses import rkd_distance_loss, rkd_angle_loss


def test_rkd_distance_zero_when_relative_same():
    s = torch.tensor([[0.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    t = torch.tensor([[2.0, 1.0], [3.0, 1.0]], dtype=torch.float32)
    loss = rkd_distance_loss(s, t)
    assert loss.item() == pytest.approx(0.0)


def test_rkd_distance_zero_when_identical():
    s = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    t = torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32)
    loss = rkd_distance_loss(s, t)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_rkd_distance_nonzero():
    s = torch.tensor([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=torch.float32)
    t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    loss = rkd_distance_loss(s, t)
    assert loss.item() > 0


def test_rkd_distance_reduction_none_matches_mean():
    s = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=torch.float32)
    t = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)
    vec = rkd_distance_loss(s, t, reduction="none")
    scalar = rkd_distance_loss(s, t, reduction="mean")
    assert vec.shape == (3,)
    assert vec.mean().item() == pytest.approx(scalar.item())


def test_rkd_angle_zero_when_same():
    s = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    t = torch.tensor([[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]], dtype=torch.float32)
    loss = rkd_angle_loss(s, t)
    assert loss.item() == pytest.approx(0.0)


def test_rkd_angle_nonzero():
    s = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    t = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    loss = rkd_angle_loss(s, t)
    assert loss.item() > 0
