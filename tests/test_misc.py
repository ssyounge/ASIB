import sys
import pytest; pytest.importorskip("torch")
import torch

from utils.misc import get_amp_components


def test_get_amp_components_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch.amp", None)
    monkeypatch.delattr(torch, "amp", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    ctx, scaler = get_amp_components({"use_amp": True})
    from torch.cuda.amp import GradScaler, autocast
    assert isinstance(scaler, GradScaler)
    assert type(ctx) == type(autocast())

