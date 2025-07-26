import pytest
torch = pytest.importorskip("torch")
from modules.trainer_student import student_distillation_update
from models.mbm import IB_MBM

class DummyTeacher(torch.nn.Module):
    distill_dim = 1
    def forward(self, x):
        b = x.size(0)
        return {
            "feat_2d": torch.zeros(b, 2),
            "distill_feat": torch.ones(b, 1),
            "logit": torch.zeros(b, 2),
        }
    def get_feat_dim(self):
        return 2
    def get_feat_channels(self):
        return 2

class DummyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 1)
    def forward(self, x):
        b = x.size(0)
        feat = torch.zeros(b, 1)
        logit = torch.zeros(b, 2)
        return {"feat_2d": feat}, logit, None
    def get_feat_dim(self):
        return 1

class RecordMBM(IB_MBM):
    def __init__(self):
        super().__init__(q_dim=1, kv_dim=1, d_emb=1)
        self.record = None
    def forward(self, q, feats):
        self.record = [feats[:, i].size(-1) for i in range(feats.size(1))]
        return super().forward(q, feats)

class DummyHead(torch.nn.Module):
    def forward(self, x):
        return torch.zeros(x.size(0), 2)

class DummyLogger:
    def info(self, msg):
        pass
    def update_metric(self, key, value):
        pass

def test_student_distill_uses_distill_feat():
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    student = DummyStudent()
    mbm = RecordMBM()
    head = DummyHead()
    loader = [(torch.zeros(1,3), torch.tensor([0]))]
    cfg = {"device": "cpu", "ce_alpha": 1.0, "kd_alpha": 1.0, "student_iters": 1,
           "use_distillation_adapter": True}
    logger = DummyLogger()
    opt = torch.optim.Adam(student.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    student_distillation_update([t1, t2], mbm, head, student, loader, loader, cfg,
                                logger, optimizer=opt, scheduler=sched)
    assert mbm.record == [t1.distill_dim, t2.distill_dim]
