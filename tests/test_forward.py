# tests/test_forward.py

import torch
from models.ib.gate_mbm import GateMBM

def test_forward():
    mbm = GateMBM(16, 16, 256, 100)
    f1 = torch.randn(4, 16, 4, 4)
    f2 = torch.randn(4, 16, 4, 4)
    z, logit, kl, _ = mbm(f1, f2)
    assert logit.shape == (4, 100)
