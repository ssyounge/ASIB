import pytest

torch = pytest.importorskip("torch")

from models.la_mbm import LightweightAttnMBM


def test_custom_query_dim_forward():
    mbm = LightweightAttnMBM([16, 16], out_dim=32, r=2, query_dim=8)
    q = torch.randn(4, 8)
    feats = [torch.randn(4, 16), torch.randn(4, 16)]
    out, attn = mbm(q, feats)
    assert out.shape == (4, 32)
    assert attn.shape[0] == 4

