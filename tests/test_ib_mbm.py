import pytest

torch = pytest.importorskip("torch")
from models.mbm import IB_MBM, build_from_teachers

def test_forward_shape():
    mbm = IB_MBM(q_dim=256, kv_dim=256, d_emb=128)
    q = torch.randn(4, 256)
    kv = torch.randn(4, 2, 256)
    z, mu, logvar = mbm(q, kv)
    assert z.shape == (4, 128)


def test_logvar_clamping():
    mbm = IB_MBM(q_dim=16, kv_dim=16, d_emb=8)
    q = torch.randn(2, 16)
    kv = torch.randn(2, 3, 16)
    _, _, logvar = mbm(q, kv)
    assert torch.all(logvar <= 10.0)
    assert torch.all(logvar >= -10.0)

def test_ib_loss_nonneg():
    mbm = IB_MBM(q_dim=256, kv_dim=256, d_emb=128, beta=0.01)
    decoder = torch.nn.Linear(128, 10)
    q = torch.randn(4, 256)
    kv = torch.randn(4, 2, 256)
    y = torch.randint(0, 10, (4,))
    z, mu, logvar = mbm(q, kv)
    logvar = torch.clamp(logvar, -10.0, 10.0)
    q_dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
    p_dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    ce = torch.nn.CrossEntropyLoss()(decoder(z), y)
    kl = torch.distributions.kl_divergence(q_dist, p_dist).mean()
    loss = ce + mbm.beta * kl
    assert loss.item() >= 0


def test_loss_backward():
    mbm = IB_MBM(q_dim=16, kv_dim=16, d_emb=8)
    decoder = torch.nn.Linear(8, 4)
    q = torch.randn(2, 16, requires_grad=True)
    kv = torch.randn(2, 1, 16)
    y = torch.randint(0, 4, (2,))
    z, mu, logvar = mbm(q, kv)
    logvar = torch.clamp(logvar, -10.0, 10.0)
    q_dist = torch.distributions.Normal(mu, torch.exp(0.5 * logvar))
    p_dist = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
    ce = torch.nn.CrossEntropyLoss()(decoder(z), y)
    kl = torch.distributions.kl_divergence(q_dist, p_dist).mean()
    loss = ce + mbm.beta * kl
    loss.backward()
    assert q.grad is not None
    assert q.grad.abs().sum() > 0


@pytest.mark.parametrize("d_emb", [8, 32])
def test_output_dim_matches_d_emb(d_emb):
    mbm = IB_MBM(q_dim=16, kv_dim=16, d_emb=d_emb)
    q = torch.randn(2, 16)
    kv = torch.randn(2, 1, 16)
    z, _, _ = mbm(q, kv)
    assert z.size(1) == d_emb


def test_ib_loss_scales_with_beta():
    torch.manual_seed(0)
    mbm_small = IB_MBM(q_dim=16, kv_dim=16, d_emb=8, beta=0.1)
    torch.manual_seed(0)
    mbm_large = IB_MBM(q_dim=16, kv_dim=16, d_emb=8, beta=1.0)
    mbm_large.load_state_dict(mbm_small.state_dict())

    q = torch.randn(2, 16)
    kv = torch.randn(2, 1, 16)
    y = torch.randint(0, 4, (2,))
    decoder = torch.nn.Linear(8, 4)

    torch.manual_seed(0)
    z1, mu1, logvar1 = mbm_small(q, kv)
    torch.manual_seed(0)
    z2, mu2, logvar2 = mbm_large(q, kv)

    logvar1 = torch.clamp(logvar1, -10.0, 10.0)
    logvar2 = torch.clamp(logvar2, -10.0, 10.0)
    q1 = torch.distributions.Normal(mu1, torch.exp(0.5 * logvar1))
    p1 = torch.distributions.Normal(torch.zeros_like(mu1), torch.ones_like(mu1))
    q2 = torch.distributions.Normal(mu2, torch.exp(0.5 * logvar2))
    p2 = torch.distributions.Normal(torch.zeros_like(mu2), torch.ones_like(mu2))
    ce1 = torch.nn.CrossEntropyLoss()(decoder(z1), y)
    ce2 = torch.nn.CrossEntropyLoss()(decoder(z2), y)
    kl1 = torch.distributions.kl_divergence(q1, p1).mean()
    kl2 = torch.distributions.kl_divergence(q2, p2).mean()
    loss1 = ce1 + mbm_small.beta * kl1
    loss2 = ce2 + mbm_large.beta * kl2

    assert loss1.item() >= 0
    assert loss2.item() >= 0
    assert loss2.item() > loss1.item()


class DummyTeacher(torch.nn.Module):
    def __init__(self, feat_dim=32, distill_dim=4):
        super().__init__()
        self.feat_dim = feat_dim
        self.distill_dim = distill_dim

    def get_feat_dim(self):
        return self.feat_dim

    def get_feat_channels(self):
        return self.feat_dim


@pytest.mark.parametrize("use_da", [True, False])
def test_build_from_teachers_distill_dim(use_da):
    teacher = DummyTeacher()
    cfg = {
        "mbm_type": "ib_mbm",
        "mbm_query_dim": 16,
        "mbm_out_dim": 8,
        "use_distillation_adapter": use_da,
    }
    mbm, _ = build_from_teachers([teacher], cfg, query_dim=16)

    expected_dim = teacher.distill_dim if use_da else teacher.get_feat_dim()
    assert mbm.kv_proj.in_features == expected_dim
