import pytest

torch = pytest.importorskip("torch")

from modules.cutmix_finetune_teacher import cutmix_criterion


def test_cutmix_criterion_handles_high_dim_logits():
    pred = torch.randn(2, 5, 4, 4, requires_grad=True)
    y_a = torch.tensor([1, 2])
    y_b = torch.tensor([3, 0])
    criterion = torch.nn.CrossEntropyLoss()
    lam = 0.7
    loss = cutmix_criterion(criterion, pred, y_a, y_b, lam)
    assert loss.dim() == 0
