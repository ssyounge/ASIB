import torch
from models.ib.vib_mbm import VIB_MBM

def test_forward():
    vib = VIB_MBM(2048, 2048, 256, 100)
    z, logit, kl, mu = vib(torch.randn(4,2048), torch.randn(4,2048))
    assert logit.shape == (4,100)
