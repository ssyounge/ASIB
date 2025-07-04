# methods/crd.py

"""Simple Contrastive Representation Distillation."""

import torch
import torch.nn.functional as F
from typing import Optional
from tqdm.auto import tqdm
import torch.nn as nn

# CRD = CE + α·InfoNCE  (원 논문 식)
from modules.losses import ce_loss_fn
from utils.eval import evaluate_acc
from utils.misc import get_amp_components
from utils.schedule import cosine_lr_scheduler


class CRDDistiller(nn.Module):
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
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        self.alpha = float(alpha)          # α = CRD 비중
        self.temperature = float(temperature)  # τ
        self.label_smoothing = float(label_smoothing)
        self.cfg = config or {}

        # ─ Projection 헤드(고정 1회 생성) —
        s_dim = student_model.get_feat_dim()
        t_dim = teacher_model.get_feat_dim()
        self.proj = nn.Identity() if s_dim == t_dim else nn.Linear(s_dim, t_dim, bias=False)

    # ────────── unified forward API ──────────
    def _get_feat_logit(self, out):
        if isinstance(out, tuple):
            return out[0]["feat_2d"], out[1]
        elif isinstance(out, dict):
            return out["feat_2d"], out["logit"]
        else:  # only logits returned
            return None, out

    def _info_nce(self, s_feat, t_feat):
        s = F.normalize(s_feat.view(s_feat.size(0), -1), dim=1)
        t = F.normalize(t_feat.view(t_feat.size(0), -1), dim=1)
        logits = torch.mm(s, t.t()) / self.temperature
        return F.cross_entropy(logits, torch.arange(s.size(0), device=s.device))

    def forward(self, x, y):
        # ── teacher / student forward ──
        with torch.no_grad():
            t_feat, _ = self._get_feat_logit(self.teacher(x))
        s_feat, s_logit = self._get_feat_logit(self.student(x))

        # teacher·student 둘 다 “feat_2d” 를 제공하지 않는 경우 ⇒ CE 로만 학습
        if t_feat is None or s_feat is None:
            ce   = ce_loss_fn(s_logit, y, label_smoothing=self.label_smoothing)
            loss = ce                        # CRD 항은 계산하지 않음
            return loss, s_logit

        s_proj = self.proj(s_feat)              # ← 등록된 모듈 사용

        crd = self._info_nce(s_proj, t_feat)
        ce = ce_loss_fn(s_logit, y, label_smoothing=self.label_smoothing)
        loss = (1 - self.alpha) * ce + self.alpha * crd
        return loss, s_logit

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
        self.proj.to(device)
        optimizer = torch.optim.AdamW(
            list(self.student.parameters()) + list(self.proj.parameters()),
            lr=float(lr), weight_decay=float(weight_decay)
        )
        scheduler = cosine_lr_scheduler(optimizer, epochs)
        autocast_ctx, scaler = get_amp_components(cfg)  # ce_criterion 제거
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
                optimizer.zero_grad()
                with autocast_ctx:
                    loss, s_logit = self.forward(x, y)
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
        # < 추가 >
        return float(final_acc)
