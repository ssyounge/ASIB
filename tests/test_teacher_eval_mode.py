import pytest

torch = pytest.importorskip("torch")

from methods.at import ATDistiller
from methods.fitnet import FitNetDistiller
from methods.dkd import DKDDistiller
from methods.crd import CRDDistiller


class DummyTeacher(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        return {
            "feat_4d_layer3": torch.zeros(b, 1, 1, 1),
            "feat_4d_layer2": torch.zeros(b, 1, 1, 1),
            "feat_2d": torch.zeros(b, 1),
            "logit": torch.zeros(b, 2),
        }


class DummyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = torch.nn.Linear(3, 1)
        self.cls = torch.nn.Linear(3, 2)

    def forward(self, x):
        f = self.feat(x)
        f4d = f.view(x.size(0), 1, 1, 1)
        f2d = f
        logit = self.cls(x)
        return {"feat_4d_layer3": f4d, "feat_4d_layer2": f4d, "feat_2d": f2d}, logit, None


def get_loader():
    x = torch.randn(2, 3)
    y = torch.tensor([0, 1])
    return [(x, y)]


@pytest.mark.parametrize(
    "distiller_cls, kwargs",
    [
        (ATDistiller, {}),
        (FitNetDistiller, {"hint_key": "feat_4d_layer3", "guided_key": "feat_4d_layer3"}),
        (DKDDistiller, {}),
        (CRDDistiller, {}),
    ],
)
def test_train_distillation_teacher_eval(distiller_cls, kwargs):
    teacher = DummyTeacher()
    student = DummyStudent()
    teacher.train()
    distiller = distiller_cls(teacher, student, **kwargs)
    loader = get_loader()
    distiller.train_distillation(loader, test_loader=None, epochs=1, device="cpu")
    assert not teacher.training
