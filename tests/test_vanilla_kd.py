import pytest

torch = pytest.importorskip("torch")

from methods.vanilla_kd import VanillaKDDistiller
from modules.losses import ce_loss_fn, kd_loss_fn


class ConstTeacher(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return {"logit": self.logits.expand(x.size(0), -1)}


class ConstStudent(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = torch.tensor(logits, dtype=torch.float32)

    def forward(self, x):
        return {}, self.logits.expand(x.size(0), -1), None


def test_forward_uses_label_smoothing():
    teacher = ConstTeacher([[1.0, -1.0]])
    student = ConstStudent([[-1.0, 1.0]])

    cfg = {"label_smoothing": 0.2}
    distiller = VanillaKDDistiller(teacher, student, alpha=0.5, temperature=1.0, config=cfg)

    x = torch.zeros(1, 3)
    y = torch.tensor([0])

    loss, _ = distiller.forward(x, y)

    ce = ce_loss_fn(student.logits, y, label_smoothing=0.2)
    kd = kd_loss_fn(student.logits, teacher.logits, T=1.0)
    expected = 0.5 * ce + 0.5 * kd
    assert torch.allclose(loss, expected)
