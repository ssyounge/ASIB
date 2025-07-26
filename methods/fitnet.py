# methods/fitnet.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional
from modules.losses import ce_loss_fn
from utils.misc import get_amp_components

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
        s_channels=512,              # 학생 특징맵의 채널 수
        t_channels=88,               # 스승 특징맵의 채널 수
        alpha_hint=1.0,
        alpha_ce=1.0,
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model

        cfg = config or {}

        # 어느 레이어(키)를 hint/guided로 쓸지
        self.hint_key = cfg.get("hint_key", hint_key)
        self.guided_key = cfg.get("guided_key", guided_key)

        self.alpha_hint = cfg.get("fit_alpha_hint", alpha_hint)
        self.alpha_ce = cfg.get("fit_alpha_ce", alpha_ce)
        self.label_smoothing = cfg.get("label_smoothing", label_smoothing)
        # optional runtime configuration for training loops
        self.cfg = cfg

        # 학생 특징맵을 스승 특징맵 채널로 변환하는 1x1 convolution
        self.regressor = nn.Conv2d(s_channels, t_channels, kernel_size=1)

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
        t_feat = t_dict[self.hint_key]  # e.g. [N, C_t, H_t, W_t] 
        s_feat = s_dict[self.guided_key]
        s_feat_resized = F.interpolate(
            s_feat, size=t_feat.shape[-2:], mode="bilinear", align_corners=False
        )

        # 채널 불일치 시 1×1 Conv projection
        if s_feat_resized.shape[1] != t_feat.shape[1]:
            if not hasattr(self, "_proj_conv"):
                self._proj_conv = torch.nn.Conv2d(
                    in_channels=s_feat_resized.shape[1],
                    out_channels=t_feat.shape[1],
                    kernel_size=1,
                    bias=False,
                ).to(s_feat.device)
            s_feat_resized = self._proj_conv(s_feat_resized)

        hint_loss = F.mse_loss(s_feat_resized, t_feat)

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

            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                logging.info(
                    "[Epoch %s] FitNet => loss=%.4f, testAcc=%.2f",
                    epoch,
                    avg_loss,
                    acc,
                )
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                logging.info("[Epoch %s] FitNet => loss=%.4f", epoch, avg_loss)

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
