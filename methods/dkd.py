# methods/dkd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DKDDistiller(nn.Module):
    """
    Decoupled Knowledge Distillation (DKD) Distiller
    dict-based Teacher/Student:
      teacher(x)->dict_out (must contain "logit"),
      student(x)->(s_dict, s_logit, _)
    => total_loss = CE + warmup_factor * DKD
    """
    def __init__(self, 
                 teacher_model, 
                 student_model,
                 ce_weight=1.0,
                 alpha=1.0,
                 beta=1.0,
                 temperature=4.0,
                 warmup=5):
        """
        Args:
          ce_weight: CE 손실 가중치
          alpha, beta: DKD 로짓 분해 비중
          temperature: T
          warmup: 몇 epoch에 걸쳐 DKD 비중을 0->1로 서서히 증가
        """
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.ce_weight = ce_weight
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.warmup = warmup

        self.criterion_ce = nn.CrossEntropyLoss()

    def forward(self, x, y, epoch=1):
        """
        1) teacher => dict_out
        2) student => (s_dict, s_logit)
        3) CE + DKD
           - DKD는 warmup_factor= min(epoch/self.warmup, 1.0)
           - total_loss = ce_weight*CE + warmup_factor*(dkd_loss)
        """
        # teacher
        with torch.no_grad():
            t_out = self.teacher(x)
            t_dict = t_out
            t_logit = t_out["logit"]
        # student
        s_dict, s_logit, _ = self.student(x)

        # CE
        ce_loss = self.ce_weight * self.criterion_ce(s_logit, y)

        # DKD
        warmup_factor = min(float(epoch) / self.warmup, 1.0)
        dkd_val = dkd_loss(
            s_logit,
            t_logit,
            y,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature
        )
        dkd_val = dkd_val * warmup_factor

        total_loss = ce_loss + dkd_val
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
        Similar to other KD loops:
        - teacher fixed
        - only student is updated
        - pass 'epoch' to self.forward(...) for warmup
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

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss, total_num = 0.0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # pass epoch so forward can do warmup
                loss, _ = self.forward(x, y, epoch=epoch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            # optional evaluate
            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                print(f"[Epoch {epoch}] DKD => loss={avg_loss:.4f}, testAcc={acc:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                print(f"[Epoch {epoch}] DKD => loss={avg_loss:.4f}")

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
            # student forward
            _, s_logit, _ = self.student(x)
            pred = s_logit.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
