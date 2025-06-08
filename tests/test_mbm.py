import torch
from models.mbm import ManifoldBridgingModule


def test_mbm_2d_only():
    N = 2
    feat_dims = [10, 20]
    mbm = ManifoldBridgingModule(feat_dims=feat_dims, hidden_dim=32, out_dim=16)
    feats_2d = [torch.randn(N, d) for d in feat_dims]
    out = mbm(feats_2d)
    assert out.shape == (N, 16)


def test_mbm_4d_only():
    N = 2
    mbm = ManifoldBridgingModule(
        feat_dims=[1],
        hidden_dim=8,
        out_dim=4,
        use_4d=True,
        in_ch_4d=3,
        out_ch_4d=5,
    )
    # disable 2D path to test 4D path alone
    mbm.use_2d = False
    feats_4d = [torch.randn(N, 3, 4, 4)]
    out = mbm([], feats_4d)
    assert out.shape == (N, 4)


def test_mbm_mixed_with_attention():
    N = 2
    feat_dims = [8, 8]
    mbm = ManifoldBridgingModule(
        feat_dims=feat_dims,
        hidden_dim=16,
        out_dim=12,
        use_4d=True,
        in_ch_4d=6,
        out_ch_4d=6,
        attn_heads=2,
    )
    feats_2d = [torch.randn(N, d) for d in feat_dims]
    feats_4d = [torch.randn(N, 3, 4, 4), torch.randn(N, 3, 4, 4)]
    out = mbm(feats_2d, feats_4d)
    assert out.shape == (N, 12)
