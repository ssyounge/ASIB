import sys
import pytest; pytest.importorskip("torch")
import torch

from utils.misc import get_amp_components
from utils.path_utils import to_writable


def test_get_amp_components_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.amp", None)
    monkeypatch.delattr(torch, "amp", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ctx, scaler = get_amp_components({"use_amp": True})
    from torch.cuda.amp import GradScaler, autocast
    assert isinstance(scaler, GradScaler)
    assert type(ctx) == type(autocast())


def test_to_writable_expands_env(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    result = to_writable("$HOME/testfile")
    assert result == str(tmp_path / "testfile")

