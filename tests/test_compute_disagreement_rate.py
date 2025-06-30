import pytest

torch = pytest.importorskip("torch")

from modules.disagreement import compute_disagreement_rate


class ConstTeacher(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.record_training = None

    def forward(self, x):
        self.record_training = self.training
        b = x.size(0)
        return {"logit": self.logits.expand(b, -1)}


def get_loader():
    # four samples split into two batches
    return [
        (torch.zeros(2, 1), torch.tensor([2, 0])),
        (torch.zeros(2, 1), torch.tensor([1, 2])),
    ]


def test_disagreement_rate_pred_and_both_wrong():
    t1 = ConstTeacher([[2.0, 0.0, 0.0]])  # predicts class 0
    t2 = ConstTeacher([[0.0, 2.0, 0.0]])  # predicts class 1
    loader = get_loader()

    pred_rate = compute_disagreement_rate(t1, t2, loader, device="cpu", mode="pred")
    both_wrong_rate = compute_disagreement_rate(
        t1, t2, loader, device="cpu", mode="both_wrong"
    )

    assert pred_rate == pytest.approx(100.0)
    assert both_wrong_rate == pytest.approx(50.0)


def test_disagreement_rate_preserves_training_state():
    t1 = ConstTeacher([[2.0, 0.0, 0.0]])
    t2 = ConstTeacher([[0.0, 2.0, 0.0]])
    loader = get_loader()

    # teachers start in train mode
    t1.train()
    t2.train()

    compute_disagreement_rate(t1, t2, loader, device="cpu")

    assert t1.record_training is False
    assert t2.record_training is False
    assert t1.training
    assert t2.training

    # teachers start in eval mode
    t1.eval()
    t2.eval()

    compute_disagreement_rate(t1, t2, loader, device="cpu")

    assert t1.record_training is False
    assert t2.record_training is False
    assert not t1.training
    assert not t2.training
