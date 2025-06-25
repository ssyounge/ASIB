# tests/test_amp.py

import contextlib
import pytest
torch = pytest.importorskip("torch")
from utils.misc import get_amp_components

def test_amp_disabled():
    ctx, scaler = get_amp_components({"use_amp": False})
    assert isinstance(ctx, contextlib.AbstractContextManager)
    assert scaler is None

def test_amp_enabled():
    cfg = {"use_amp": True, "amp_dtype": "float16", "device": "cuda"}
    ctx, scaler = get_amp_components(cfg)
    if torch.cuda.is_available():
        assert scaler is not None
    else:
        assert scaler is None


def test_amp_cpu_device():
    cfg = {"use_amp": True, "device": "cpu"}
    ctx, scaler = get_amp_components(cfg)
    assert isinstance(ctx, contextlib.AbstractContextManager)
    assert scaler is None
