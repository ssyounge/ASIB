# methods/dkd.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Optional
from modules.losses import ce_loss_fn, dkd_loss
from utils.common import get_amp_components

class DKDDistiller(nn.Module):
    """
    Decoupled Knowledge Distillation (DKD) Distiller
    dict-based Teacher/Student:
      teacher(x)->dict_out (must contain "logit"),
      student(x)->(s_dict, s_logit, _)
    => total_loss = CE + warmup_factor * DKD
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        ce_weight=1.0,
        alpha=1.0,
        beta=1.0,
        temperature=4.0,
        warmup=5,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ):
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
        cfg = config or {}
        self.ce_weight = cfg.get("dkd_ce_weight", ce_weight)
        self.alpha = cfg.get("dkd_alpha", alpha)
        self.beta = cfg.get("dkd_beta", beta)
        self.temperature = cfg.get("tau_start", temperature)
        self.warmup = cfg.get("dkd_warmup", warmup)
        self.label_smoothing = cfg.get("label_smoothing", label_smoothing)
        # optional runtime configuration for training loops
        self.cfg = cfg

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
            t_logit = t_out["logit"]
        # student
        s_dict, s_logit, _ = self.student(x)

        # CE
        ce_loss = self.ce_weight * ce_loss_fn(
            s_logit,
            y,
            label_smoothing=self.label_smoothing,
        )

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
        device="cuda",
        cfg=None,
    ):
        """
        Similar to other KD loops:
        - teacher fixed
        - only student is updated
        - pass 'epoch' to self.forward(...) for warmup
        """
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

        for epoch in range(1, epochs + 1):
            self.student.train()
            self.teacher.eval()
            total_loss, total_num = 0.0, 0

            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                with autocast_ctx:
                    # pass epoch so forward can do warmup
                    loss, _ = self.forward(x, y, epoch=epoch)

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

            # optional evaluate
            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                logging.info(
                    "[Epoch %s] DKD => loss=%.4f, testAcc=%.2f",
                    epoch,
                    avg_loss,
                    acc,
                )
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                logging.info("[Epoch %s] DKD => loss=%.4f", epoch, avg_loss)

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
