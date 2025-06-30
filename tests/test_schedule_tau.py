# tests/test_schedule_tau.py

import pytest

from utils.schedule import get_tau


def test_linear_decay_equivalent_to_linear():
    cfg = {
        "temperature_schedule": "linear",
        "tau_start": 10.0,
        "tau_end": 2.0,
        "tau_decay_epochs": 4,
    }
    cfg_decay = dict(cfg)
    cfg_decay["temperature_schedule"] = "linear_decay"

    for epoch in range(6):
        tau_linear = get_tau(cfg, epoch)
        tau_decay = get_tau(cfg_decay, epoch)
        assert tau_linear == pytest.approx(tau_decay)
