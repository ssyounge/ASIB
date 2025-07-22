import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn
from modules.partial_freeze import apply_partial_freeze


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.fc = nn.Linear(3, 3)


def test_no_freeze_when_level_negative():
    m = _Toy()
    apply_partial_freeze(m, level=-1)
    assert all(p.requires_grad for p in m.parameters())


def test_head_only_freeze():
    m = _Toy()
    apply_partial_freeze(m, level=0)
    conv_grad = m.conv.weight.requires_grad
    fc_grad = m.fc.weight.requires_grad
    assert conv_grad is False and fc_grad is True
