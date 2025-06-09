import contextlib
import pytest
pytest.importorskip("torch")
from utils.misc import get_amp_components

def test_amp_disabled():
    ctx, scaler = get_amp_components({"use_amp": False})
    assert isinstance(ctx, contextlib.AbstractContextManager)
    assert scaler is None

def test_amp_enabled():
    cfg = {"use_amp": True, "amp_dtype": "float16"}
    ctx, scaler = get_amp_components(cfg)
    assert scaler is not None
