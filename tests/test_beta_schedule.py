import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.schedule import get_beta


def test_beta_increases_with_warmup():
    cfg = {"ib_beta": 0.1, "ib_beta_warmup_epochs": 5}
    betas = [get_beta(cfg, ep) for ep in range(6)]
    assert betas[0] == 0.0
    assert betas[5] == pytest.approx(0.1)
    assert all(x <= y for x, y in zip(betas, betas[1:]))
