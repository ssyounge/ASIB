import pytest

torch = pytest.importorskip("torch")

from modules.cutmix_finetune_teacher import cutmix_criterion


def test_cutmix_criterion_clamps_labels():
    pred = torch.randn(2, 5, requires_grad=True)
    y_a = torch.tensor([-1, 6])  # intentionally out of range
    y_b = torch.tensor([5, -2])
    criterion = torch.nn.CrossEntropyLoss()
    lam = 0.5

    loss = cutmix_criterion(criterion, pred, y_a, y_b, lam)

    y_a_clamped = torch.clamp(y_a, 0, pred.size(1) - 1)
    y_b_clamped = torch.clamp(y_b, 0, pred.size(1) - 1)
    expected = lam * criterion(pred, y_a_clamped) + (1 - lam) * criterion(pred, y_b_clamped)

    assert torch.allclose(loss, expected)
