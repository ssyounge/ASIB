# methods/fitnet.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from distillers.losses import ce_loss_fn

class FitNetDistiller(nn.Module):
    """
    FitNet Distiller: 
    - student intermediate feature vs teacher intermediate feature => MSE
    - plus CE on final logit
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        hint_alpha=1.0,
        ce_alpha=1.0
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.hint_alpha = hint_alpha
        self.ce_alpha = ce_alpha

    def forward(self, x, y):
        """
        1) teacher => (hint_feat, t_logit, ...)
        2) student => (guided_feat, s_logit, ...)
        3) MSE(hint vs guided) + CE
        """
        with torch.no_grad():
            t_hint, t_logit, _ = self.teacher(x)  # teacher's "hint layer"
        s_guided, s_logit, _ = self.student(x)   # student's guided layer

        # MSE: hint => guided
        hint_loss = F.mse_loss(s_guided, t_hint)
        # CE
        ce_loss = ce_loss_fn(s_logit, y)
        total_loss = self.hint_alpha * hint_loss + self.ce_alpha * ce_loss
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
        # (구조는 VanillaKD와 동일 패턴)
        self.to(device)
        optimizer = optim.SGD(self.student.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

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

                total_loss += loss.item() * x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                print(f"[Epoch {epoch}] FitNet => loss={avg_loss:.4f}, testAcc={acc:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                print(f"[Epoch {epoch}] FitNet => loss={avg_loss:.4f}")

        if best_state is not None:
            self.student.load_state_dict(best_state["student"])
        return best_acc

    @torch.no_grad()
    def evaluate(self, loader, device="cuda"):
        # student accuracy
        self.eval()
        self.to(device)
        correct, total = 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            # forward student
            _, s_logit, _ = self.student(x)
            pred = s_logit.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
