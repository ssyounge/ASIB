# methods/at.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from distillers.losses import ce_loss_fn

def at_loss(teacher_feat, student_feat, p=2):
    """
    Attention Transfer (AT) loss:
     - teacher_feat, student_feat: [N, C, H, W]
     - p=2 (L2 norm)로 채널별 norm => attention map => MSE
    """
    # 1) teacher attention
    t_atten = teacher_feat.abs().pow(p).sum(dim=1)  # [N, H, W]
    t_atten = t_atten.pow(1.0/p)

    # 2) student attention
    s_atten = student_feat.abs().pow(p).sum(dim=1)
    s_atten = s_atten.pow(1.0/p)

    # 3) 채널 제외한 [H,W] flatten => normalize
    t_norm = F.normalize(t_atten.view(t_atten.size(0), -1), dim=1)
    s_norm = F.normalize(s_atten.view(s_atten.size(0), -1), dim=1)

    # 4) MSE
    return F.mse_loss(s_norm, t_norm)

class ATDistiller(nn.Module):
    """
    Distiller for Attention Transfer (Zagoruyko & Komodakis, ICLR2017).
    total_loss = alpha * AT_loss + CE
    """
    def __init__(self, teacher_model, student_model, alpha=1.0, p=2):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.p = p

    def forward(self, x, y):
        """
        1) teacher => (feat, logit, ...)
        2) student => (feat, logit, ...)
        3) AT_loss( teacher_feat, student_feat ) + CE(student_logit, y)
        """
        with torch.no_grad():
            t_feat, t_logit, _ = self.teacher(x)

        s_feat, s_logit, _ = self.student(x)

        # AT loss
        loss_at = at_loss(t_feat, s_feat, p=self.p)
        # CE
        ce_val = ce_loss_fn(s_logit, y)
        total_loss = self.alpha * loss_at + ce_val
        return total_loss, s_logit

    def train_distillation(
        self,
        train_loader,
        test_loader=None,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=10,
        device="cuda"
    ):
        self.to(device)
        # Student 파라미터만 업데이트 (Teacher는 고정)
        optimizer = optim.SGD(
            self.student.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay
        )
        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs+1):
            self.train()
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, _ = self.forward(x, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()*x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                print(f"[Epoch {epoch}] AT => loss={avg_loss:.4f}, testAcc={acc:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                print(f"[Epoch {epoch}] AT => loss={avg_loss:.4f}")

        if best_state is not None:
            self.student.load_state_dict(best_state["student"])
        return best_acc

    @torch.no_grad()
    def evaluate(self, loader, device="cuda"):
        self.eval()
        self.to(device)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # forward student
            _, s_logit, _ = self.student(x)
            pred = s_logit.argmax(dim=1)
            correct += (pred==y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
