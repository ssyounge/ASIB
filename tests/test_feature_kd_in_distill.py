import pytest

torch = pytest.importorskip("torch")

from models.la_mbm import LightweightAttnMBM
from modules.trainer_student import student_distillation_update
from modules.losses import ce_loss_fn, kd_loss_fn


class ConstTeacher(torch.nn.Module):
    def __init__(self, feat, logit):
        super().__init__()
        self.register_buffer("feat", torch.tensor(feat, dtype=torch.float32))
        self.register_buffer("logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, x):
        b = x.size(0)
        return {
            "feat_2d": self.feat.expand(b, -1),
            "logit": self.logit.expand(b, -1),
        }

    def get_feat_dim(self):
        return self.feat.size(1)

    def get_feat_channels(self):
        return self.feat.size(1)


class ConstStudent(torch.nn.Module):
    def __init__(self, feat, logit):
        super().__init__()
        self.register_buffer("feat", torch.tensor(feat, dtype=torch.float32))
        self.register_buffer("logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, x):
        b = x.size(0)
        return {"feat_2d": self.feat.expand(b, -1)}, self.logit.expand(b, -1), None

    def get_feat_dim(self):
        return self.feat.size(1)


class AvgMBM(torch.nn.Module):
    def forward(self, feats_2d, feats_4d=None):
        return sum(feats_2d) / len(feats_2d)


class ConstHead(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.register_buffer("logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, x):
        return self.logit.expand(x.size(0), -1)


class DummyLogger:
    def __init__(self):
        self.metrics = {}

    def info(self, msg: str):
        pass

    def update_metric(self, key, value):
        self.metrics[key] = value


def test_feature_kd_term_in_student_distill():
    t1 = ConstTeacher([[1.0, 2.0]], [[0.5, 0.5]])
    t2 = ConstTeacher([[3.0, 0.0]], [[0.1, 0.9]])
    student = ConstStudent([[0.0, 0.0]], [[0.0, 1.0]])
    mbm = AvgMBM()
    head = ConstHead([[0.5, -0.5]])

    loader = [(torch.zeros(1, 3), torch.tensor([1]))]

    cfg = {
        "device": "cpu",
        "ce_alpha": 1.0,
        "kd_alpha": 1.0,
        "feat_kd_alpha": 0.5,
        "student_iters": 1,
    }

    logger = DummyLogger()

    opt = torch.optim.Adam(student.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    student_distillation_update(
        [t1, t2],
        mbm,
        head,
        student,
        loader,
        loader,
        cfg,
        logger,
        optimizer=opt,
        scheduler=sched,
    )

    ep_loss = logger.metrics["student_ep1_loss"]

    ce = ce_loss_fn(student.logit, torch.tensor([1]))
    kd = kd_loss_fn(student.logit, head.logit, T=cfg.get("tau_start", 4.0))
    fsyn = (t1.feat + t2.feat) / 2
    mse = torch.nn.functional.mse_loss(student.feat, fsyn)
    expected = cfg["ce_alpha"] * ce + cfg["kd_alpha"] * kd + cfg["feat_kd_alpha"] * mse

    assert ep_loss == pytest.approx(expected.item())


@pytest.mark.parametrize("use_la", [False, True])
def test_student_distill_loss_components(use_la):
    t1 = ConstTeacher([[1.0, 2.0]], [[0.5, 0.5]])
    t2 = ConstTeacher([[3.0, 0.0]], [[0.1, 0.9]])
    student = ConstStudent([[0.0, 0.0]], [[0.0, 1.0]])

    if use_la:
        mbm = LightweightAttnMBM(
            [2, 2], out_dim=2, r=1, n_head=1, learnable_q=False, query_dim=2
        )
    else:
        mbm = AvgMBM()

    head = ConstHead([[0.5, -0.5]])

    loader = [(torch.zeros(1, 3), torch.tensor([1]))]

    cfg = {
        "device": "cpu",
        "ce_alpha": 1.0,
        "kd_alpha": 1.0,
        "feat_kd_alpha": 0.5,
        "student_iters": 1,
    }

    logger = DummyLogger()

    opt = torch.optim.Adam(student.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)

    student_distillation_update(
        [t1, t2],
        mbm,
        head,
        student,
        loader,
        loader,
        cfg,
        logger,
        optimizer=opt,
        scheduler=sched,
    )

    ep_loss = logger.metrics["student_ep1_loss"]
    feat_kd_logged = logger.metrics["ep1_feat_kd"]

    ce = ce_loss_fn(student.logit, torch.tensor([1]))
    kd = kd_loss_fn(student.logit, head.logit, T=cfg.get("tau_start", 4.0))

    with torch.no_grad():
        if use_la:
            _, _, s_q, t_attn = mbm(student.feat, [t1.feat, t2.feat])
            feat_kd = torch.nn.functional.mse_loss(s_q, t_attn)
        else:
            fsyn = mbm([t1.feat, t2.feat])
            feat_kd = torch.nn.functional.mse_loss(student.feat, fsyn)

    expected = (
        cfg["ce_alpha"] * ce
        + cfg["kd_alpha"] * kd
        + cfg["feat_kd_alpha"] * feat_kd
    )

    assert feat_kd_logged == pytest.approx(feat_kd.item())
    assert ep_loss == pytest.approx(expected.item())
