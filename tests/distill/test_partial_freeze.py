import torch
import torch.nn as nn
from modules.partial_freeze import apply_partial_freeze

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.fc = nn.Linear(3, 3)


def test_apply_partial_freeze():
    m = Toy()
    apply_partial_freeze(m, level=0)
    assert not m.conv.weight.requires_grad
    assert m.fc.weight.requires_grad
