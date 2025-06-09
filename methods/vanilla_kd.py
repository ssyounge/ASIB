# methods/vanilla_kd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from modules.losses import kd_loss_fn, ce_loss_fn
from utils.schedule import get_tau

class VanillaKDDistiller(nn.Module):
    """
    Distiller for 'vanilla KD' (Hinton et al. 2015).
    The teacher forward returns a dict containing a "logit" entry and
    optional features. Only logits are used here.
    total_loss = alpha * CE + (1 - alpha) * KD
    """
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        alpha: float = 0.5,
        temperature: float = 4.0,
        config: dict | None = None
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = alpha
        self.temperature = temperature
        self.cfg = config if config is not None else {}

    def forward(self, x, y, tau=None):
        """
        1) teacher => dict_out (use "logit" field)
        2) student => (dict, s_logit, _)
        3) CE + KD
        """
        with torch.no_grad():
            t_out = self.teacher(x)
            t_logit = t_out["logit"]
        s_dict, s_logit, _ = self.student(x)     # we don't use s_dict either

        # CE
        ce_loss = ce_loss_fn(s_logit, y)
        # KD
        T_use = self.temperature if tau is None else tau
        kd_loss = kd_loss_fn(s_logit, t_logit, T=T_use)

        total_loss = self.alpha * ce_loss + (1 - self.alpha) * kd_loss
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
        """
        Basic KD training loop:
          - Student is updated, Teacher is fixed
          - Evaluate optionally each epoch
        """
        self.to(device)
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
            cur_tau = get_tau(self.cfg, epoch-1)
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss, _ = self.forward(x, y, tau=cur_tau)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                print(f"[Epoch {epoch}] VanillaKD => loss={avg_loss:.4f}, testAcc={acc:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                print(f"[Epoch {epoch}] VanillaKD => loss={avg_loss:.4f}")

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
            correct += (pred == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
