import pytest; pytest.importorskip("torch")
import torch
import torch.nn as nn

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


@pytest.mark.parametrize("distiller_cls", [ATDistiller, FitNetDistiller])
def test_distiller_forward(distiller_cls):
    teacher = DummyNet()
    student = DummyNet()
    distiller = distiller_cls(teacher, student)
    x = torch.randn(2, 3, 8, 8)
    y = torch.randint(0, 10, (2,))
    loss, logits = distiller(x, y)
    assert torch.isfinite(loss).item()
    assert logits.shape[0] == x.size(0)
