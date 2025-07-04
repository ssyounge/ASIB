# methods/crd.py

"""Simple Contrastive Representation Distillation."""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm.auto import tqdm

# CRD = CE + α·InfoNCE  (원 논문 식)
from modules.losses import ce_loss_fn
from utils.eval import evaluate_acc
from utils.misc import get_amp_components
from utils.schedule import cosine_lr_scheduler


class CRDDistiller:
    """Minimal CRD distiller using feature MSE with KD loss."""

    def __init__(
        self,
        teacher_model,
        student_model,
        alpha: float = 0.5,
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ) -> None:
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = float(alpha)          # α = CRD 비중
        self.temperature = float(temperature)  # τ
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
        ce_criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        for ep in range(epochs):
            self.student.train()
            running = 0.0
            correct = 0
            count = 0
            for x, y in tqdm(
                train_loader,
                desc=f"[CRD] epoch {ep+1}",
                leave=False,
                disable=cfg.get("disable_tqdm", False),
            ):
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    t_out = self.teacher(x)
                    if isinstance(t_out, tuple):
                        t_feat = t_out[0]["feat_2d"]
                        t_logit = t_out[1]
                    elif isinstance(t_out, dict):
                        t_feat = t_out["feat_2d"]
                        t_logit = t_out["logit"]
                    else:
                        t_logit = t_out
                        t_feat = None
                optimizer.zero_grad()
                with autocast_ctx:
                    s_out = self.student(x)
                    if isinstance(s_out, tuple):
                        s_feat = s_out[0]["feat_2d"]
                        s_logit = s_out[1]
                    elif isinstance(s_out, dict):
                        s_feat = s_out["feat_2d"]
                        s_logit = s_out["logit"]
                    else:
                        s_logit = s_out
                        s_feat = None
                    ce = ce_criterion(s_logit, y)
                    # ─ InfoNCE( student vs teacher ) ─
                    if s_feat is not None and t_feat is not None:
                        s = F.normalize(s_feat.view(s_feat.size(0), -1), dim=1)
                        t = F.normalize(t_feat.view(t_feat.size(0), -1), dim=1)
                        logits_ct = torch.mm(s, t.t()) / self.temperature
                        labels_ct = torch.arange(s.size(0), device=s.device)
                        crd = F.cross_entropy(logits_ct, labels_ct)
                    else:
                        crd = 0.0

                    # total = (1-α)·CE + α·CRD
                    loss = (1 - self.alpha) * ce + self.alpha * crd
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running += loss.item() * x.size(0)
                correct += (s_logit.argmax(1) == y).sum().item()
                count += x.size(0)
            scheduler.step()
            train_acc = 100.0 * correct / max(count, 1)
            test_acc = (
                evaluate_acc(self.student, test_loader, device=device)
                if test_loader is not None
                else 0.0
            )
            print(
                f"[CRD] ep {ep+1:03d}/{epochs} train_acc {train_acc:.2f}% test_acc {test_acc:.2f}%"
            )
        final_acc = (
            evaluate_acc(self.student, test_loader, device=device)
            if test_loader is not None
            else train_acc
        )
        return float(final_acc)
