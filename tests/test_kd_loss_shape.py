import pytest

torch = pytest.importorskip("torch")

from modules.losses import kd_loss_fn


def test_kd_loss_mismatched_shapes_raise():
    student = torch.zeros(1, 3)
    teacher = torch.zeros(1, 4)
    with pytest.raises(ValueError, match="Mismatched logit shapes"):
        kd_loss_fn(student, teacher)
