import pytest

torch = pytest.importorskip("torch")

from modules.trainer_teacher import teacher_adaptive_update


class DummyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3)
        self.trainable = torch.nn.Linear(3, 3)
        for p in self.frozen.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.trainable(x)
        x = self.frozen(x)
        return {"feat_2d": x, "logit": x, "feat_4d": None}

    def get_feat_dim(self):
        return 3

    def get_feat_channels(self):
        return 3


class DummyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = torch.nn.Linear(3, 3)
        self.cls = torch.nn.Linear(3, 3)

    def forward(self, x):
        f = self.feat(x)
        logit = self.cls(x)
        return {"feat_2d": f}, logit, None

    def get_feat_dim(self):
        return 3


class AvgMBM(torch.nn.Module):
    def forward(self, feats_2d, feats_4d=None):
        return sum(feats_2d) / len(feats_2d)


class DummyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.fc(x)


class DummyLogger:
    def __init__(self):
        self.metrics = {}

    def info(self, msg: str):
        pass

    def update_metric(self, key, value):
        self.metrics[key] = value


def test_teacher_adaptive_update_preserves_freeze():
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    student = DummyStudent()
    mbm = AvgMBM()
    head = DummyHead()

    loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]

    cfg = {"device": "cpu", "synergy_ce_alpha": 0.1, "teacher_iters": 1}

    params = []
    for m in [t1, t2, mbm, head]:
        for p in m.parameters():
            if p.requires_grad:
                params.append(p)
    opt = torch.optim.SGD(params, lr=0.1)
    logger = DummyLogger()

    frozen_before = [p.requires_grad for p in t1.frozen.parameters()]

    teacher_adaptive_update(
        teacher_wrappers=[t1, t2],
        mbm=mbm,
        synergy_head=head,
        student_model=student,
        trainloader=loader,
        testloader=None,
        cfg=cfg,
        logger=logger,
        optimizer=opt,
        scheduler=None,
        global_ep=0,
    )

    frozen_after = [p.requires_grad for p in t1.frozen.parameters()]
    assert frozen_before == frozen_after == [False, False]
