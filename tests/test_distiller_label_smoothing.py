import pytest

torch = pytest.importorskip("torch")

from methods.at import ATDistiller
from methods.fitnet import FitNetDistiller
from methods.dkd import DKDDistiller
from methods.crd import CRDDistiller
from modules.losses import ce_loss_fn


class ConstTeacher(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        logits = torch.tensor([[1.0, -1.0]], dtype=torch.float32).expand(b, -1)
        return {
            "feat_4d_layer3": torch.ones(b, 1, 1, 1),
            "feat_4d_layer2": torch.ones(b, 1, 1, 1),
            "feat_2d": torch.ones(b, 1),
            "logit": logits,
        }


class ConstStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.tensor([[-1.0, 1.0]], dtype=torch.float32)

    def forward(self, x):
        b = x.size(0)
        return {
            "feat_4d_layer3": torch.ones(b, 1, 1, 1),
            "feat_4d_layer2": torch.ones(b, 1, 1, 1),
            "feat_2d": torch.ones(b, 1),
        }, self.logits.expand(b, -1), None


@pytest.mark.parametrize(
    "distiller_cls, kwargs",
    [
        (ATDistiller, {"alpha": 0.0}),
        (FitNetDistiller, {"alpha_hint": 0.0, "alpha_ce": 1.0}),
        (DKDDistiller, {"alpha": 0.0, "beta": 0.0, "warmup": 1}),
        (CRDDistiller, {"alpha": 0.0}),
    ],
)
def test_forward_label_smoothing_applied(distiller_cls, kwargs):
    teacher = ConstTeacher()
    student = ConstStudent()
    distiller = distiller_cls(teacher, student, label_smoothing=0.2, **kwargs)
    x = torch.zeros(1, 3)
    y = torch.tensor([0])
    if distiller_cls is DKDDistiller:
        loss, _ = distiller.forward(x, y, epoch=1)
    else:
        loss, _ = distiller.forward(x, y)
    expected = ce_loss_fn(student.logits, y, label_smoothing=0.2)
    assert torch.allclose(loss, expected)
