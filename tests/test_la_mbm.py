import pytest

pytest.skip("LightweightAttnMBM removed", allow_module_level=True)

torch = pytest.importorskip("torch")

"""
Deprecated LA-MBM tests removed.
"""
"""

def test_custom_query_dim_forward():
    mbm = LightweightAttnMBM([16, 16], out_dim=32, r=2,
                              learnable_q=False, query_dim=8)
    q = torch.randn(4, 8)
    feats = [torch.randn(4, 16), torch.randn(4, 16)]
    out, attn, q_proj, attn_out = mbm(q, feats)
    assert out.shape == (4, 32)
    assert attn.shape[0] == 4


def test_zero_query_dim_uses_teacher_dim():
    mbm = LightweightAttnMBM([16, 16], out_dim=32, r=2,
                              learnable_q=False, query_dim=0)
    q = torch.randn(2, 32)  # sum of teacher dims = 32
    feats = [torch.randn(2, 16), torch.randn(2, 16)]
    out, attn, q_proj, attn_out = mbm(q, feats)
    assert out.shape == (2, 32)
    assert attn.shape[0] == 2


def test_query_dim_mismatch_raises():
    mbm = LightweightAttnMBM([16, 16], out_dim=32, r=2,
                              learnable_q=False, query_dim=8)
    q = torch.randn(1, 10)
    feats = [torch.randn(1, 16), torch.randn(1, 16)]
    with pytest.raises(ValueError, match="mbm_query_dim mismatch: expected 8, got 10"):
        mbm(q, feats)

"""
