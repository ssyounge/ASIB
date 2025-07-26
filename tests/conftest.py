import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import pytest

class DummyTeacher(torch.nn.Module):
    def __init__(self, num_classes: int = 2, feat_dim: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        return {
            "feat_2d": torch.zeros(b, self.feat_dim),
            "feat_4d": torch.zeros(b, self.feat_dim, 1, 1),
            "logit": torch.zeros(b, self.num_classes),
        }

@pytest.fixture
def dummy_teachers():
    return DummyTeacher(), DummyTeacher()
