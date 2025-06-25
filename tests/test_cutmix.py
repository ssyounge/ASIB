import random
import torch
import pytest

from utils.misc import cutmix_data


def test_cutmix_disabled():
    x = torch.zeros(2, 3, 4, 4)
    y = torch.tensor([0, 1])
    out_x, ta, tb, lam = cutmix_data(x, y, alpha=0.0)
    assert torch.equal(out_x, x)
    assert torch.equal(ta, y)
    assert torch.equal(tb, y)
    assert lam == 1.0


def test_cutmix_shapes_and_range():
    random.seed(0)
    torch.manual_seed(0)
    x = torch.randn(2, 3, 8, 8)
    y = torch.tensor([0, 1])
    out_x, ta, tb, lam = cutmix_data(x, y, alpha=1.0)
    assert out_x.shape == x.shape
    assert ta.shape == y.shape
    assert tb.shape == y.shape
    assert 0.0 <= lam <= 1.0
