# methods/reviewkd.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from typing import Optional

from modules.losses import ce_loss_fn
from utils.training import get_tau
from utils.common import get_amp_components

class ReviewKDDistiller(nn.Module):
    """
    Distiller for 'ReviewKD' (Chen et al. 2020).
    ReviewKD: A Comprehensive Knowledge Distillation Framework for Object Detection
    Uses multi-level feature distillation with attention mechanism.
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
        self.feature_weight = cfg.get("feature_weight", 1.0)
        self.attention_weight = cfg.get("attention_weight", 0.5)
        self.cfg = cfg

    def forward(self, x, y, tau=None):
        """
        ReviewKD forward pass:
        1) teacher => multi-level features + logits
        2) student => multi-level features + logits  
        3) CE + Multi-level Feature KD + Attention KD
        """
        with torch.no_grad():
            t_out = self.teacher(x)
            if isinstance(t_out, tuple):
                t_out = t_out[0]
            t_logit = t_out["logit"]
            # Teacher multi-level features
            t_feats = {}
            for key in t_out.keys():
                if key.startswith("feat_"):
                    t_feats[key] = t_out[key]
        
        s_dict, s_logit, _ = self.student(x)
        # Student multi-level features
        s_feats = {}
        for key in s_dict.keys():
            if key.startswith("feat_"):
                s_feats[key] = s_dict[key]

        # CE Loss
        label_smoothing = self.cfg.get("label_smoothing", 0.0)
        ce_loss = ce_loss_fn(s_logit, y, label_smoothing=label_smoothing)

        # Logit KD Loss
        T_use = self.temperature if tau is None else tau
        logit_kd_loss = 0.0
        if T_use > 0:
            logit_kd_loss = nn.KLDivLoss(reduction='batchmean')(
                torch.log_softmax(s_logit / T_use, dim=1),
                torch.softmax(t_logit / T_use, dim=1)
            ) * (T_use ** 2)

        # Multi-level Feature KD Loss (ReviewKD 핵심)
        feature_kd_loss = 0.0
        attention_kd_loss = 0.0
        
        for feat_key in t_feats.keys():
            if feat_key in s_feats:
                t_feat = t_feats[feat_key]
                s_feat = s_feats[feat_key]
                
                # Feature alignment loss
                if t_feat.shape == s_feat.shape:
                    feature_kd_loss += F.mse_loss(s_feat, t_feat)
                    
                    # Attention mechanism (ReviewKD의 핵심)
                    if len(t_feat.shape) == 4:  # [B, C, H, W]
                        # Spatial attention
                        t_attn = torch.mean(t_feat, dim=1, keepdim=True)  # [B, 1, H, W]
                        s_attn = torch.mean(s_feat, dim=1, keepdim=True)  # [B, 1, H, W]
                        
                        # Attention distillation
                        attention_kd_loss += F.mse_loss(s_attn, t_attn)

        # Total Loss
        total_loss = (
            self.alpha * ce_loss + 
            (1 - self.alpha) * logit_kd_loss + 
            self.feature_weight * feature_kd_loss +
            self.attention_weight * attention_kd_loss
        )
        
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
        ReviewKD training loop:
          - Student is updated, Teacher is fixed
          - Multi-level feature + attention distillation
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
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-6
            )
        elif lr_schedule == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        else:
            scheduler = None

        for epoch in range(epochs):
            # Student only training; teacher fixed
            self.student.train()
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                with autocast_ctx:
                    loss, logits = self.forward(data, target)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx % 100 == 0:
                    logging.info(
                        f"Epoch {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] "
                        f"Loss: {loss.item():.4f} "
                        f"Acc: {100.*correct/total:.2f}%"
                    )
            
            if scheduler is not None:
                scheduler.step()
            
            # Evaluate on test set
            if test_loader is not None:
                test_acc = self.evaluate(test_loader, device)
                logging.info(f"Epoch {epoch+1} Test Acc: {test_acc:.2f}%")

    @torch.no_grad()
    def evaluate(self, loader, device="cuda"):
        """Evaluate the student model."""
        self.eval()
        correct = 0
        total = 0
        
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            _, logits = self.forward(data, target)
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return 100.0 * correct / total 