# methods/vanilla_kd.py

import torch
import torch.nn as nn
import torch.optim as optim
import logging
from typing import Optional

from modules.losses import kd_loss_fn, ce_loss_fn
from utils.training import get_tau
from utils.common import get_amp_components

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
        config: Optional[dict] = None
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        cfg = config or {}
        self.alpha = cfg.get("ce_alpha", alpha)
        self.temperature = cfg.get("tau_start", temperature)
        self.cfg = cfg

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
        label_smoothing = self.cfg.get("label_smoothing", 0.0)
        ce_loss = ce_loss_fn(
            s_logit,
            y,
            label_smoothing=label_smoothing,
        )
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
        device="cuda",
        cfg=None,
    ):
        """
        Basic KD training loop:
          - Student is updated, Teacher is fixed
          - Evaluate optionally each epoch
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

        for epoch in range(1, epochs+1):
            # ensure only the student is in training mode
            self.student.train()
            self.teacher.eval()
            cur_tau = get_tau(self.cfg, epoch-1)
            total_loss, total_num = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                with autocast_ctx:
                    loss, _ = self.forward(x, y, tau=cur_tau)

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

            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                logging.info(
                    "[Epoch %s] VanillaKD => loss=%.4f, testAcc=%.2f",
                    epoch,
                    avg_loss,
                    acc,
                )
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                logging.info(
                    "[Epoch %s] VanillaKD => loss=%.4f", epoch, avg_loss
                )

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
