import torch
from models import IB_MBM


def test_ib_mbm_output_shape_dtype():
    ib_mbm = IB_MBM(q_dim=8, kv_dim=8, d_emb=4)
    q = torch.randn(2, 8)
    kv = torch.randn(2, 8)  # 2D tensor instead of 3D
    z, mu, logvar = ib_mbm(q, kv)
    assert z.shape == (2, 4)
    assert mu.dtype == torch.float32
    assert logvar.shape == (2, 4)
