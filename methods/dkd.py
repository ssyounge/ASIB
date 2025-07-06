# methods/dkd.py

"""Simplified Decoupled Knowledge Distillation."""

import torch
from typing import Optional
from tqdm.auto import tqdm

from modules.losses import dkd_loss
import torch.nn as nn  # forward 내부 투입 시 사용 가능
from torch.nn.functional import cross_entropy
from utils.eval import evaluate_acc
from utils.misc import get_amp_components
from utils.schedule import cosine_lr_scheduler


class DKDDistiller:
    """Minimal DKD distiller."""

    def __init__(
        self,
        teacher_model,
        student_model,
        ce_weight: float = 1.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        temperature: float = 4.0,
        warmup: int = 5,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ) -> None:
        self.teacher = teacher_model
        self.student = student_model
        self.ce_weight = float(ce_weight)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.warmup = int(warmup)
        self.label_smoothing = float(label_smoothing)
        self.cfg = config or {}

    # ───────── unified forward ─────────
    def _get_logits(self, out):
        if isinstance(out, tuple):
            return out[1]
        elif isinstance(out, dict):
            return out["logit"]
        else:
            return out

    def forward(self, x, y, warm):
        with torch.no_grad():
            t_logits = self._get_logits(self.teacher(x)).detach()
        s_logits = self._get_logits(self.student(x))

        ce = self.ce_weight * cross_entropy(
            s_logits, y, label_smoothing=self.label_smoothing
        )
        dkd = dkd_loss(
            s_logits,
            t_logits,
            y,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )
        loss = ce + dkd * warm      # warm‑up factor 적용
        return loss, s_logits

    def train_distillation(
        self,
        train_loader,
        test_loader,
        epochs: int = 1,
        lr: float = 5e-4,
        weight_decay: float = 5e-4,
        device: str = "cuda",
        cfg: Optional[dict] = None,
    ) -> float:
        cfg = {**self.cfg, **(cfg or {})}
        device = device or cfg.get("device", "cuda")
        self.teacher.to(device).eval()
        self.student.to(device)
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        scheduler = cosine_lr_scheduler(
            optimizer,
            epochs,
            warmup_epochs=cfg.get("student_warmup_epochs", 0),
            min_lr_ratio=cfg.get("min_lr_ratio_student", 0.05),
        )
        autocast_ctx, scaler = get_amp_components(cfg)  # criterion_ce 삭제
        for ep in range(epochs):
            self.student.train()
            running = 0.0
            correct = 0
            count = 0
            warm = min(1.0, (ep + 1) / max(1, self.warmup))
            for x, y in tqdm(
                train_loader,
                desc=f"[DKD] epoch {ep+1}",
                leave=False,
                disable=cfg.get("disable_tqdm", False),
            ):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with autocast_ctx:
                    loss, s_logits = self.forward(x, y, warm)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running += loss.item() * x.size(0)
                correct += (s_logits.argmax(1) == y).sum().item()
                count += x.size(0)
            scheduler.step()
            train_acc = 100.0 * correct / max(count, 1)
            test_acc = (
                evaluate_acc(self.student, test_loader, device=device)
                if test_loader is not None
                else 0.0
            )
            print(
                f"[DKD] ep {ep+1:03d}/{epochs} train_acc {train_acc:.2f}% test_acc {test_acc:.2f}%"
            )
        final_acc = (
            evaluate_acc(self.student, test_loader, device=device)
            if test_loader is not None
            else train_acc
        )
        # < 추가 >
        return float(final_acc)
