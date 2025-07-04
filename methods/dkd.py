"""Simplified Decoupled Knowledge Distillation."""

import torch
from typing import Optional
from tqdm.auto import tqdm

from modules.losses import dkd_loss
from utils.eval import evaluate_acc
from utils.misc import get_amp_components
from utils.schedule import cosine_lr_scheduler


class DKDDistiller:
    """Minimal DKD distiller."""

    def __init__(
        self,
        teacher_model,
        student_model,
        alpha: float = 1.0,
        beta: float = 8.0,
        temperature: float = 4.0,
        warmup: int = 5,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ) -> None:
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.temperature = float(temperature)
        self.warmup = int(warmup)
        self.label_smoothing = float(label_smoothing)
        self.cfg = config or {}

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
        self.teacher.eval()
        self.student.to(device)
        optimizer = torch.optim.AdamW(
            self.student.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        scheduler = cosine_lr_scheduler(optimizer, epochs)
        autocast_ctx, scaler = get_amp_components(cfg)
        criterion_ce = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        for ep in range(epochs):
            self.student.train()
            running = 0.0
            correct = 0
            count = 0
            progress = min(1.0, (ep + 1) / max(1, self.warmup))
            for x, y in tqdm(
                train_loader,
                desc=f"[DKD] epoch {ep+1}",
                leave=False,
                disable=cfg.get("disable_tqdm", False),
            ):
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    t_out = self.teacher(x)
                    if isinstance(t_out, tuple):
                        t_logits = t_out[1]
                    elif isinstance(t_out, dict):
                        t_logits = t_out["logit"]
                    else:
                        t_logits = t_out
                optimizer.zero_grad()
                with autocast_ctx:
                    s_out = self.student(x)
                    if isinstance(s_out, tuple):
                        s_logits = s_out[1]
                    elif isinstance(s_out, dict):
                        s_logits = s_out["logit"]
                    else:
                        s_logits = s_out
                    loss = dkd_loss(
                        s_logits,
                        t_logits.detach(),
                        y,
                        alpha=self.alpha * progress,
                        beta=self.beta,
                        temperature=self.temperature,
                    )
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
        return float(final_acc)
