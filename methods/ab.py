# methods/ab.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
from typing import Optional

from modules.losses import ce_loss_fn
from utils.training import get_tau
from utils.common import get_amp_components

class ABDistiller(nn.Module):
    """
    Distiller for 'AB' (Heo et al. 2019).
    AB: Attention Branch Network
    Uses attention mechanism for knowledge distillation.
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
        self.attention_weight = cfg.get("attention_weight", 1.0)
        self.feature_weight = cfg.get("feature_weight", 0.5)
        self.cfg = cfg

    def forward(self, x, y, tau=None):
        """
        AB forward pass:
        1) teacher => attention maps + features + logits
        2) student => attention maps + features + logits  
        3) CE + Attention KD + Feature KD
        """
        with torch.no_grad():
            t_out = self.teacher(x)
            if isinstance(t_out, tuple):
                t_out = t_out[0]
            t_logit = t_out["logit"]
            t_feat = t_out.get("feat_2d", None)  # teacher features
            t_attn = t_out.get("attention", None)  # teacher attention
        
        s_dict, s_logit, _ = self.student(x)
        s_feat = s_dict.get("feat_2d", None)  # student features
        s_attn = s_dict.get("attention", None)  # student attention

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

        # Attention KD Loss (AB 핵심)
        attention_loss = 0.0
        if t_attn is not None and s_attn is not None:
            # Attention map distillation
            if len(t_attn.shape) == 4 and len(s_attn.shape) == 4:  # [B, C, H, W]
                # Adaptive pooling if shapes don't match
                if t_attn.shape == s_attn.shape:
                    # Spatial attention
                    t_spatial_attn = torch.mean(t_attn, dim=1, keepdim=True)  # [B, 1, H, W]
                    s_spatial_attn = torch.mean(s_attn, dim=1, keepdim=True)  # [B, 1, H, W]
                    
                    # Attention distillation loss
                    attention_loss += F.mse_loss(s_spatial_attn, t_spatial_attn)
                    
                    # Channel attention (handle different channel dimensions)
                    t_channel_attn = torch.mean(t_attn, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
                    s_channel_attn = torch.mean(s_attn, dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
                    
                    # Use smaller channel dimension
                    if t_channel_attn.shape[1] != s_channel_attn.shape[1]:
                        min_channels = min(t_channel_attn.shape[1], s_channel_attn.shape[1])
                        t_channel_attn = t_channel_attn[:, :min_channels]
                        s_channel_attn = s_channel_attn[:, :min_channels]
                    
                    attention_loss += F.mse_loss(s_channel_attn, t_channel_attn)
                else:
                    # Adaptive pooling to match sizes
                    target_size = t_attn.shape[2:]
                    s_attn_adapted = F.adaptive_avg_pool2d(s_attn, target_size)
                    
                    # Spatial attention
                    t_spatial_attn = torch.mean(t_attn, dim=1, keepdim=True)
                    s_spatial_attn = torch.mean(s_attn_adapted, dim=1, keepdim=True)
                    attention_loss += F.mse_loss(s_spatial_attn, t_spatial_attn)
                    
                    # Channel attention (handle different channel dimensions)
                    t_channel_attn = torch.mean(t_attn, dim=[2, 3], keepdim=True)
                    s_channel_attn = torch.mean(s_attn_adapted, dim=[2, 3], keepdim=True)
                    
                    # Use smaller channel dimension
                    if t_channel_attn.shape[1] != s_channel_attn.shape[1]:
                        min_channels = min(t_channel_attn.shape[1], s_channel_attn.shape[1])
                        t_channel_attn = t_channel_attn[:, :min_channels]
                        s_channel_attn = s_channel_attn[:, :min_channels]
                    
                    attention_loss += F.mse_loss(s_channel_attn, t_channel_attn)

        # Feature KD Loss
        feature_kd_loss = 0.0
        if t_feat is not None and s_feat is not None:
            # Adaptive pooling if shapes don't match
            if t_feat.shape == s_feat.shape:
                feature_kd_loss = F.mse_loss(s_feat, t_feat)
            else:
                # Adaptive average pooling to match sizes
                target_size = t_feat.shape[2:]
                s_feat_adapted = F.adaptive_avg_pool2d(s_feat, target_size)
                
                # Handle different channel dimensions
                if s_feat_adapted.shape[1] != t_feat.shape[1]:
                    min_channels = min(s_feat_adapted.shape[1], t_feat.shape[1])
                    s_feat_adapted = s_feat_adapted[:, :min_channels]
                    t_feat_adapted = t_feat[:, :min_channels]
                else:
                    t_feat_adapted = t_feat
                
                feature_kd_loss = F.mse_loss(s_feat_adapted, t_feat_adapted)

        # Total Loss
        total_loss = (
            self.alpha * ce_loss + 
            (1 - self.alpha) * logit_kd_loss + 
            self.attention_weight * attention_loss +
            self.feature_weight * feature_kd_loss
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
        AB training loop:
          - Student is updated, Teacher is fixed
          - Attention distillation + feature distillation
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