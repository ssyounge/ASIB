# methods/crd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from modules.losses import ce_loss_fn
from utils.misc import get_amp_components

class CRDLoss(nn.Module):
    """
    (위에서 이미 정의한 CRDLoss 그대로)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, s_feat, t_feat):
        """Compute InfoNCE loss between student and teacher features."""
        # flatten in case features have extra dimensions (e.g. N,C,H,W)
        s = s_feat.view(s_feat.size(0), -1)
        t = t_feat.view(t_feat.size(0), -1)

        # L2-normalize along the feature dimension
        s = F.normalize(s, dim=1)
        t = F.normalize(t, dim=1)

        # similarity matrix and labels for contrastive learning
        logits = torch.mm(s, t.t()) / self.temperature
        labels = torch.arange(s.size(0), device=s.device)

        loss = F.cross_entropy(logits, labels)
        return loss


class CRDDistiller(nn.Module):
    """
    CRD Distiller (dict-based):
      - teacher(x) -> dict_out (must contain the feature used)
      - student(x) -> (s_dict, s_logit, ...)
      - CRD = InfoNCE(s_dict[feat_key], t_dict[feat_key])
      - CE = cross entropy with s_logit
      - total_loss = alpha * CRD + (1 - alpha) * CE
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        feat_key="feat_2d",
        alpha=0.5,
        temperature=0.07,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.feat_key = feat_key   # 어떤 dict 키로부터 2D feat를 뽑아 CRD할지
        self.alpha = alpha
        self.crd_loss_fn = CRDLoss(temperature=temperature)
        self.label_smoothing = label_smoothing
        # optional runtime configuration for training loops
        self.cfg = config if config is not None else {}

        s_dim = student_model.get_feat_dim()
        t_dim = teacher_model.get_feat_dim()
        if s_dim != t_dim:
            self.projection = nn.Linear(s_dim, t_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, x, y):
        # 1) teacher
        with torch.no_grad():
            t_dict = self.teacher(x)
        # 2) student
        s_dict, s_logit, _ = self.student(x)

        # CRD
        t_feat = t_dict[self.feat_key]  # [N, D_t]
        s_feat = s_dict[self.feat_key]
        s_feat = self.projection(s_feat)  # match dims
        crd_loss_val = self.crd_loss_fn(s_feat, t_feat)

        # CE
        ce_val = ce_loss_fn(
            s_logit,
            y,
            label_smoothing=self.label_smoothing,
        )

        total_loss = self.alpha * crd_loss_val + (1 - self.alpha) * ce_val
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
        self.to(device)

        autocast_ctx, scaler = get_amp_components(cfg or self.cfg)

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
            betas=(
                self.cfg.get("adam_beta1", 0.9),
                self.cfg.get("adam_beta2", 0.999),
            ),
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
                with autocast_ctx:
                    loss, _ = self.forward(x, y)

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            
                total_loss += loss.item() * x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            scheduler.step()

            # optional: evaluate
            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                print(f"[Epoch {epoch}] CRD => loss={avg_loss:.4f}, testAcc={acc:.2f}")
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                print(f"[Epoch {epoch}] CRD => loss={avg_loss:.4f}")

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
            _, s_logit, _ = self.student(x)
            pred = s_logit.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
