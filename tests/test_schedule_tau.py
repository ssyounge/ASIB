
import pytest

from utils.schedule import get_tau


def test_polynomial_schedule_basic():
    cfg = {"tau_start": 4.0, "tau_end": 1.0, "tau_decay_power": 2.0, "T_max": 10}
    assert get_tau(cfg, 0) == pytest.approx(4.0)
    assert get_tau(cfg, 5) == pytest.approx(1.75)
    assert get_tau(cfg, 10) == pytest.approx(1.0)


def test_tau_no_decay_when_start_equals_end():
    cfg = {"tau_start": 3.0, "tau_end": 3.0, "tau_decay_power": 2.0, "T_max": 5}
    vals = [get_tau(cfg, ep) for ep in range(6)]
    assert all(v == pytest.approx(3.0) for v in vals)
