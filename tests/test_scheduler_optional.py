import pytest

torch = pytest.importorskip("torch")

from modules.trainer_student import student_distillation_update
from modules.trainer_teacher import teacher_adaptive_update


class DummyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        feat = self.linear(x)
        return {"feat_2d": feat, "logit": feat, "feat_4d": None}

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
    def info(self, msg: str):
        pass

    def update_metric(self, key, value):
        pass


def test_student_update_scheduler_none():
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    student = DummyStudent()
    mbm = AvgMBM()
    head = DummyHead()

    loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]

    cfg = {"device": "cpu", "ce_alpha": 1.0, "kd_alpha": 1.0, "student_iters": 1}

    logger = DummyLogger()
    opt = torch.optim.SGD(student.parameters(), lr=0.1)

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
        scheduler=None,
    )


def test_teacher_update_scheduler_none():
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    student = DummyStudent()
    mbm = AvgMBM()
    head = DummyHead()

    loader = [(torch.randn(2, 3), torch.tensor([0, 1]))]

    cfg = {"device": "cpu", "synergy_ce_alpha": 0.1, "teacher_iters": 1}

    logger = DummyLogger()

    params = [p for p in t1.parameters() if p.requires_grad]
    params += [p for p in t2.parameters() if p.requires_grad]
    params += [p for p in mbm.parameters() if p.requires_grad]
    params += [p for p in head.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.1)

    teacher_adaptive_update(
        [t1, t2],
        mbm,
        head,
        student,
        trainloader=loader,
        testloader=None,
        cfg=cfg,
        logger=logger,
        optimizer=opt,
        scheduler=None,
        global_ep=0,
    )
