import pytest

torch = pytest.importorskip("torch")
from modules.ib_mbm import IB_MBM

def test_forward_shape():
    mbm = IB_MBM(q_dim=256, kv_dim=256, d_emb=128)
    q = torch.randn(4, 256)
    kv = torch.randn(4, 2, 256)
    z, mu, logvar = mbm(q, kv)
    assert z.shape == (4, 128)

def test_ib_loss_nonneg():
    mbm = IB_MBM(q_dim=256, kv_dim=256, d_emb=128, beta=0.01)
    decoder = torch.nn.Linear(128, 10)
    q = torch.randn(4, 256)
    kv = torch.randn(4, 2, 256)
    y = torch.randint(0, 10, (4,))
    z, mu, logvar = mbm(q, kv)
    loss = mbm.loss(z, mu, logvar, y, decoder)
    assert loss.item() >= 0
