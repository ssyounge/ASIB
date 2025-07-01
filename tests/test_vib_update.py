import pytest

torch = pytest.importorskip("torch")

from modules.trainer_vib import teacher_vib_update, student_vib_update
from models.ib import VIB_MBM, StudentProj


class DummyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        feat = self.fc(x)
        return {"feat_2d": feat}

    def get_feat_dim(self):
        return 4


def get_loader():
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])
    return [(x, y)]


def copy_params(module):
    return [p.detach().clone() for p in module.parameters()]


def params_changed(before, module):
    after = list(module.parameters())
    return any(not torch.allclose(b, a) for b, a in zip(before, after))


class DummyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = torch.nn.Linear(4, 4)
        self.cls = torch.nn.Linear(4, 2)

    def forward(self, x):
        feat = self.feat(x)
        logit = self.cls(x)
        return {"feat_2d": feat}, logit, None

    def get_feat_dim(self):
        return 4


def test_teacher_vib_update_updates_mbm_only():
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    vib = VIB_MBM(4, 4, 2, n_cls=2)
    loader = get_loader()
    cfg = {"device": "cpu", "teacher_iters": 1}
    opt = torch.optim.SGD(vib.parameters(), lr=0.1)

    t1_before = copy_params(t1)
    t2_before = copy_params(t2)
    vib_before = copy_params(vib)

    teacher_vib_update(t1, t2, vib, loader, cfg, opt)

    assert params_changed(vib_before, vib)
    assert not params_changed(t1_before, t1)
    assert not params_changed(t2_before, t2)


def test_student_vib_update_updates_proj_only():
    t1 = DummyTeacher()
    t2 = DummyTeacher()
    vib = VIB_MBM(4, 4, 2, n_cls=2)
    student = DummyStudent()
    proj = StudentProj(4, 2)
    loader = get_loader()
    cfg = {"device": "cpu", "student_iters": 1, "alpha_kd": 0.7, "ce_alpha": 1.0}
    opt = torch.optim.SGD(list(student.parameters()) + list(proj.parameters()), lr=0.1)

    vib_before = copy_params(vib)
    t1_before = copy_params(t1)
    t2_before = copy_params(t2)
    proj_before = copy_params(proj)

    student_vib_update(t1, t2, student, vib, proj, loader, cfg, opt)

    assert params_changed(proj_before, proj)
    assert not params_changed(vib_before, vib)
    assert not params_changed(t1_before, t1)
    assert not params_changed(t2_before, t2)
