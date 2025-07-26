# methods/at.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional
from modules.losses import ce_loss_fn
from utils.misc import get_amp_components

def single_layer_at_loss(f_s, f_t, p=2):
    """
    f_s, f_t: [N, C, H, W]
    => (abs().pow(p).mean(dim=1)) -> flatten -> normalize -> MSE
    """
    if f_s.shape[2:] != f_t.shape[2:]:
        # 크기 불일치 시 adaptive pooling
        f_s = F.adaptive_avg_pool2d(f_s, (f_t.shape[2], f_t.shape[3]))

    # (1) channel-wise pow(p).mean => [N,H,W]
    sA = f_s.abs().pow(p).mean(dim=1)
    tA = f_t.abs().pow(p).mean(dim=1)

    # (2) flatten -> normalize
    sA = F.normalize(sA.view(sA.size(0), -1), dim=1)
    tA = F.normalize(tA.view(tA.size(0), -1), dim=1)

    return F.mse_loss(sA, tA)


def at_loss_dict(teacher_dict, student_dict, layer_key="feat_4d_layer3", p=2):
    """
    teacher_dict["feat_4d_layer3"], student_dict["feat_4d_layer3"] 사용
    => single_layer_at_loss 계산
    """
    f_t = teacher_dict[layer_key]  # 4D
    f_s = student_dict[layer_key]  # 4D
    return single_layer_at_loss(f_s, f_t, p=p)


class ATDistiller(nn.Module):
    """
    Example of AT Distiller using the dict-based teacher/student outputs.
    1) teacher(x)->dict_out
    2) student(x)->(s_dict, s_logits)
    3) at_loss_dict(...) => single_layer_at_loss
    4) total_loss = CE + alpha*AT
    """
    def __init__(
        self,
        teacher_model,
        student_model,
        alpha=1.0,
        p=2,
        layer_key="feat_4d_layer3",
        label_smoothing: float = 0.0,
        config: Optional[dict] = None,
    ):
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        cfg = config or {}
        self.alpha = cfg.get("at_alpha", alpha)
        self.p = cfg.get("at_p", p)
        self.layer_key = cfg.get("layer_key", layer_key)
        self.label_smoothing = cfg.get("label_smoothing", label_smoothing)
        # optional runtime configuration for training loops
        self.cfg = cfg

    def forward(self, x, y):
        # 1) teacher
        with torch.no_grad():
            t_dict = self.teacher(x)
        # 2) student
        s_dict, s_logit, _ = self.student(x)

        # 3) at_loss
        loss_at = at_loss_dict(t_dict, s_dict, layer_key=self.layer_key, p=self.p)
        # 4) CE
        loss_ce = ce_loss_fn(
            s_logit,
            y,
            label_smoothing=self.label_smoothing,
        )
        total_loss = loss_ce + self.alpha * loss_at

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

            
                total_loss += loss.item()*x.size(0)
                total_num  += x.size(0)

            avg_loss = total_loss / total_num

            scheduler.step()

            # evaluate
            if test_loader is not None:
                acc = self.evaluate(test_loader, device)
                logging.info(
                    "[Epoch %s] AT => loss=%.4f, testAcc=%.2f",
                    epoch,
                    avg_loss,
                    acc,
                )
                if acc > best_acc:
                    best_acc = acc
                    best_state = {"student": self.student.state_dict()}
            else:
                logging.info("[Epoch %s] AT => loss=%.4f", epoch, avg_loss)

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
            correct += (pred==y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total
