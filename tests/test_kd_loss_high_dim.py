import pytest

torch = pytest.importorskip("torch")

from modules.losses import kd_loss_fn


def test_kd_loss_fn_handles_high_dim_logits():
    student = torch.randn(2, 5, 4, 4)
    teacher = torch.randn(2, 5, 4, 4)
    out = kd_loss_fn(student, teacher, T=2.0, reduction="none")
    assert out.shape == (2, 5)

