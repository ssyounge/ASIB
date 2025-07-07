import pytest; pytest.importorskip("torch")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from methods import ATDistiller, FitNetDistiller


class DummyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        pooled = self.gap(f3)
        feat_2d = pooled.view(pooled.size(0), -1)
        logit = self.fc(feat_2d)
        ce_loss = self.criterion(logit, y) if y is not None else None
        feat_dict = {
            "feat_4d_layer1": f1,
            "feat_4d_layer2": f2,
            "feat_4d_layer3": f3,
            "feat_4d": f3,
            "feat_2d": feat_2d,
        }
        return feat_dict, logit, ce_loss


def _make_loader():
    x = torch.randn(4, 3, 8, 8)
    y = torch.randint(0, 10, (4,))
    return DataLoader(TensorDataset(x, y), batch_size=2)


def test_at_train_distillation_updates_student():
    teacher = DummyNet()
    student = DummyNet()
    distiller = ATDistiller(teacher, student)
    loader = _make_loader()
    initial = student.conv1.weight.detach().clone()
    distiller.train_distillation(loader, epochs=1, lr=0.01, device="cpu")
    updated = student.conv1.weight
    assert not torch.allclose(initial, updated)


def test_fitnet_train_distillation_updates_student_and_regressor():
    torch.manual_seed(0)
    teacher = DummyNet()
    student = DummyNet()
    distiller = FitNetDistiller(teacher, student)
    # warm-up forward to create regressor
    batch_loader = _make_loader()
    x, y = next(iter(batch_loader))
    with torch.no_grad():
        distiller(x, y)
    reg_init = [p.clone() for p in distiller.regressor.parameters()]
    student_init = student.conv1.weight.clone()

    train_loader = _make_loader()
    distiller.train_distillation(train_loader, epochs=1, lr=0.01, device="cpu")

    assert distiller.regressor is not None
    assert not torch.allclose(student_init, student.conv1.weight)
    for p0, p1 in zip(reg_init, distiller.regressor.parameters()):
        assert not torch.allclose(p0, p1)


@pytest.mark.parametrize("distiller_cls", [ATDistiller, FitNetDistiller])
def test_train_distillation_cfg_overrides_adam_betas(monkeypatch, distiller_cls):
    teacher = DummyNet()
    student = DummyNet()
    distiller = distiller_cls(teacher, student)
    loader = _make_loader()

    captured = {}

    class DummyOpt:
        def __init__(self, params, lr=0.0, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
            captured["betas"] = betas
            self.param_groups = [dict(lr=lr)]

        def zero_grad(self):
            pass

        def step(self):
            pass

        def add_param_group(self, group):
            self.param_groups.append(group)

    class DummyScheduler:
        def __init__(self, optimizer, *args, **kwargs):
            self.optimizer = optimizer
            self.base_lrs = [0.0]

        def step(self):
            pass

    monkeypatch.setattr(torch.optim, "AdamW", DummyOpt)
    monkeypatch.setattr(torch.optim.lr_scheduler, "CosineAnnealingLR", DummyScheduler)

    cfg = {"adam_beta1": 0.5, "adam_beta2": 0.4}
    distiller.train_distillation(loader, epochs=0, device="cpu", cfg=cfg)

    assert captured["betas"] == (0.5, 0.4)
