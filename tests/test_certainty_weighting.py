import pytest

torch = pytest.importorskip("torch")

from modules.trainer_student import student_distillation_update
from modules.trainer_teacher import teacher_adaptive_update
from modules.losses import certainty_weights


class ConstTeacher(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.register_buffer("logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, x):
        b = x.size(0)
        return {"feat_2d": torch.zeros(b, 1), "logit": self.logit.expand(b, -1), "feat_4d": None}

    def get_feat_dim(self):
        return 1

    def get_feat_channels(self):
        return 1


class ConstStudent(torch.nn.Module):
    def __init__(self, logit):
        super().__init__()
        self.register_buffer("logit", torch.tensor(logit, dtype=torch.float32))

    def forward(self, x):
        b = x.size(0)
        return {"feat_2d": torch.zeros(b, 1)}, self.logit.expand(b, -1), None

    def get_feat_dim(self):
        return 1


class DummyIBMBM(torch.nn.Module):
    def __init__(self, logvar):
        super().__init__()
        self.register_buffer("logvar", torch.tensor(logvar, dtype=torch.float32))

    def forward(self, q, feats):
        b = q.size(0)
        logvar = self.logvar.expand(b, -1)
        mu = torch.zeros_like(logvar)
        z = torch.zeros(b, self.logvar.size(0))
        return z, mu, logvar


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


def run_student(use_ib):
    t1 = ConstTeacher([[0.0, 1.0]])
    t2 = ConstTeacher([[0.0, 1.0]])
    student = ConstStudent([[0.0, 0.0]])
    mbm = DummyIBMBM([[2.0]])
    head = ConstHead([[1.0, 0.0]])
    loader = [(torch.zeros(1, 3), torch.tensor([0]))]
    cfg = {
        "device": "cpu",
        "ce_alpha": 1.0,
        "kd_alpha": 1.0,
        "student_iters": 1,
        "use_ib": use_ib,
        "ib_beta": 0.0,
    }
    logger = DummyLogger()
    opt = torch.optim.Adam(student.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    student_distillation_update([t1, t2], mbm, head, student, loader, loader, cfg, logger, optimizer=opt, scheduler=sched)
    return logger.metrics["student_ep1_loss"]


def run_teacher(use_ib):
    t1 = ConstTeacher([[0.0, 1.0]])
    t2 = ConstTeacher([[0.0, 1.0]])
    student = ConstStudent([[0.0, 0.0]])
    mbm = DummyIBMBM([[2.0]])
    head = ConstHead([[1.0, 0.0]])
    loader = [(torch.zeros(1, 3), torch.tensor([0]))]
    cfg = {
        "device": "cpu",
        "synergy_ce_alpha": 1.0,
        "teacher_iters": 1,
        "teacher_adapt_alpha_kd": 1.0,
        "use_ib": use_ib,
        "ib_beta": 0.0,
    }
    logger = DummyLogger()
    params = []
    opt = torch.optim.Adam(params, lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    teacher_adaptive_update([t1, t2], mbm, head, student, loader, None, cfg, logger, optimizer=opt, scheduler=sched, global_ep=0)
    return logger.metrics["teacher_ep1_loss"]


def test_certainty_weighting_changes_losses():
    w = certainty_weights(torch.tensor([[2.0]])).mean().item()
    loss_base = run_student(False)
    loss_weighted = run_student(True)
    assert loss_weighted == pytest.approx(loss_base * w)

    loss_base_t = run_teacher(False)
    loss_weighted_t = run_teacher(True)
    assert loss_weighted_t == pytest.approx(loss_base_t * w)
