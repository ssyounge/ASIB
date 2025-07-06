# methods/feature_kd.py
"""Knowledge distillation with feature map imitation."""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm.auto import tqdm

from modules.losses import kd_loss_fn, ce_loss_fn
from utils.eval import evaluate_acc
from utils.misc import get_amp_components
from utils.schedule import cosine_lr_scheduler
from utils.feature_hook import FeatHook
from utils.distill_loss import feat_mse


class FeatureKD:
    """KD distiller that also matches intermediate feature maps."""

    def __init__(
        self,
        teacher_model,
        student_model,
        alpha: float = 1.0,
        temperature: float = 4.0,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ) -> None:
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.label_smoothing = float(label_smoothing)
        self.cfg = config or {}

    def _get_logits(self, out):
        if isinstance(out, tuple):
            return out[1]
        elif isinstance(out, dict):
            return out["logit"]
        return out

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
        autocast_ctx, scaler = get_amp_components(cfg)
        ce_criterion = torch.nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        layer_ids = cfg.get("feat_layers", [1, 2])
        layer_w = cfg.get("feat_weights", [0.5, 0.5])
        gamma_feat = cfg.get("feat_loss_weight", 1.0)

        hook_s = FeatHook(self.student.backbone, layer_ids)
        hook_t = FeatHook(self.teacher.backbone, layer_ids)

        for ep in range(epochs):
            self.student.train()
            running = 0.0
            correct = 0
            count = 0
            for x, y in tqdm(
                train_loader,
                desc=f"[FeatureKD] epoch {ep+1}",
                leave=False,
                disable=cfg.get("disable_tqdm", False),
            ):
                x, y = x.to(device), y.to(device)
                with torch.no_grad():
                    t_out = self.teacher(x)
                    t_logits = self._get_logits(t_out)
                optimizer.zero_grad()
                with autocast_ctx:
                    s_out = self.student(x)
                    s_logits = self._get_logits(s_out)
                    ce = ce_criterion(s_logits, y)
                    kd = kd_loss_fn(s_logits, t_logits.detach(), T=self.temperature)
                    feat_loss = feat_mse(
                        hook_s.features, hook_t.features, layer_ids, layer_w
                    )
                    loss = (1 - self.alpha) * ce + self.alpha * kd + gamma_feat * feat_loss
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                hook_s.clear(); hook_t.clear()
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
                f"[FeatureKD] ep {ep+1:03d}/{epochs} train_acc {train_acc:.2f}% test_acc {test_acc:.2f}%"
            )
        final_acc = (
            evaluate_acc(self.student, test_loader, device=device)
            if test_loader is not None
            else train_acc
        )
        hook_s.close(); hook_t.close()
        return float(final_acc)
