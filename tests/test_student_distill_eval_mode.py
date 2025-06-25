import pytest

torch = pytest.importorskip("torch")

from modules.trainer_student import student_distillation_update


class RecordTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.record_training = None

    def forward(self, x):
        self.record_training = self.training
        b = x.size(0)
        return {
            "feat_2d": torch.zeros(b, 1),
            "logit": torch.zeros(b, 2),
        }

    def get_feat_dim(self):
        return 1

    def get_feat_channels(self):
        return 1


class ConstStudent(torch.nn.Module):
    def forward(self, x):
        b = x.size(0)
        return {"feat_2d": torch.zeros(b, 1)}, torch.zeros(b, 2), None

    def get_feat_dim(self):
        return 1


class AvgMBM(torch.nn.Module):
    def forward(self, feats_2d, feats_4d=None):
        return sum(feats_2d) / len(feats_2d)


class ConstHead(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 2)


class DummyLogger:
    def info(self, msg):
        pass

    def update_metric(self, key, value):
        pass


def test_teachers_are_eval_during_distill():
    t1 = RecordTeacher()
    t2 = RecordTeacher()
    student = ConstStudent()
    mbm = AvgMBM()
    head = ConstHead()

    loader = [(torch.zeros(1, 3), torch.tensor([1]))]

    cfg = {"device": "cpu", "ce_alpha": 1.0, "kd_alpha": 1.0, "student_iters": 1}

    logger = DummyLogger()

    t1.train()
    t2.train()

    opt = torch.optim.SGD(student.parameters(), lr=0.1)
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

    assert t1.record_training is False
    assert t2.record_training is False
    assert t1.training
    assert t2.training
