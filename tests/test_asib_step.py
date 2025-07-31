import torch
from methods.asib import ASIBDistiller
from models.mbm import IB_MBM, SynergyHead

class DummyStudent(torch.nn.Module):
    def __init__(self, feat_dim: int = 4, num_classes: int = 2):
        super().__init__()
        self.proj = torch.nn.Linear(3, feat_dim)
        self.cls = torch.nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.proj(x)
        logit = self.cls(x)
        return {"feat_2d": feat}, logit, None

def test_asmb_forward_backward(dummy_teachers):
    t1, t2 = dummy_teachers
    student = DummyStudent()
    mbm = IB_MBM(q_dim=4, kv_dim=4, d_emb=4)
    head = SynergyHead(4, num_classes=2)
    distiller = ASIBDistiller(t1, t2, student, mbm, head, device="cpu")
    x = torch.randn(2, 3)
    y = torch.tensor([0, 1])
    loss, _ = distiller.forward(x, y)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in student.parameters())
