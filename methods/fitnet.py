# methods/fitnet.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from modules.losses import ce_loss_fn

class FitNetDistiller(nn.Module):
    """
    FitNet Distiller (dict-based):
      - Teacher/Student: forward(x)->(feat_dict, logit) 형태
      - 특정 hint_key(Teacher) vs. guided_key(Student) => MSE
      - + CE
      total_loss = alpha_hint*MSE + alpha_ce*CE
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        hint_key="feat_4d_layer2",    # teacher가 반환하는 key
        guided_key="feat_4d_layer2",  # student가 반환하는 key
        alpha_hint=1.0,
        alpha_ce=1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model

        # 어느 레이어(키)를 hint/guided로 쓸지
        self.hint_key = hint_key
        self.guided_key = guided_key

        self.alpha_hint = alpha_hint
        self.alpha_ce = alpha_ce
        self.label_smoothing = label_smoothing

    def forward(self, x, y):
        """
        1) teacher => dict_out
        2) student => (s_dict, s_logit)
        3) MSE( t_dict[hint_key], s_dict[guided_key] ) + CE
        """
        # Teacher (no_grad)
        with torch.no_grad():
            t_out = self.teacher(x)
            t_dict = t_out

        # Student
        s_dict, s_logit, _ = self.student(x)      # (feat_dict, logit, ce_loss(opt))

        # 1) hint/guided MSE
        t_feat = t_dict[self.hint_key]  # e.g. [N, C, H, W] 
        s_feat = s_dict[self.guided_key]
        hint_loss = F.mse_loss(s_feat, t_feat)

        # 2) CE
        ce_loss = ce_loss_fn(
            s_logit,
            y,
            label_smoothing=self.label_smoothing,
        )

        # total
        total_loss = self.alpha_hint * hint_loss + self.alpha_ce * ce_loss
        return total_loss, s_logit

    def train_distillation(
        self,
        train_loader,
        test_loader=None,
        lr=1e-3,
        weight_decay=1e-4,
        epochs=10,
        device="cuda",
        cfg=None,
    ):
        """
        FitNet Distillation:
          - Student 학습
          - Teacher 고정
          - MSE(hint) + CE
        """
        self.to(device)

        if cfg is not None:
            lr = cfg.get("student_lr", lr)
            weight_decay = cfg.get("student_weight_decay", weight_decay)
            lr_schedule = cfg.get("lr_schedule", "cosine")
            step_size = cfg.get("student_step_size", 10)
            gamma = cfg.get("student_gamma", 0.1)
        else:
            lr_schedule = "cosine"
            step_size = 10
            gamma = 0.1

        optimizer = optim.AdamW(
            self.student.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        if lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs+1):
            self.student.train()
            self.teacher.eval()
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

            scheduler.step()

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
