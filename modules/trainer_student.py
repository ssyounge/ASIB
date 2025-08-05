# modules/trainer_student.py

import torch
import copy
import logging
from utils.common import smart_tqdm, mixup_data, cutmix_data, mixup_criterion, get_amp_components
from utils.training import get_tau, get_beta, StageMeter
from models.mbm import IB_MBM

from modules.loss_safe import ce_safe, kl_safe
from modules.losses import (
    ib_loss, certainty_weights, feat_mse_loss
)
from modules.disagreement import sample_weights_from_disagreement


try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class StudentTrainer:
    """Student trainer for knowledge distillation."""
    
    def __init__(self, student_model, teacher_models, device="cuda", config=None):
        """
        Initialize student trainer.
        
        Parameters:
        -----------
        student_model : nn.Module
            Student model to train
        teacher_models : list
            List of teacher models
        device : str
            Device to train on
        config : dict
            Training configuration
        """
        self.student = student_model
        self.teachers = teacher_models
        self.device = device
        self.config = config or {}
        
    def train_step(self, batch, optimizer):
        """
        Single training step.
        
        Parameters:
        -----------
        batch : tuple
            (inputs, targets) batch
        optimizer : torch.optim.Optimizer
            Optimizer for student model
            
        Returns:
        --------
        dict
            Training metrics
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward pass through student
        student_outputs = self.student(inputs)
        student_logits = student_outputs["logit"] if isinstance(student_outputs, dict) else student_outputs
        
        # Forward pass through teachers (no grad)
        teacher_logits = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher.eval()
                teacher_outputs = teacher(inputs)
                teacher_logit = teacher_outputs["logit"] if isinstance(teacher_outputs, dict) else teacher_outputs
                teacher_logits.append(teacher_logit)
        
        # Compute losses
        ce_loss = torch.nn.functional.cross_entropy(student_logits, targets)
        
        # Knowledge distillation loss (average over teachers)
        kd_loss = 0.0
        temperature = self.config.get('temperature', 4.0)
        alpha = self.config.get('alpha', 0.5)
        
        for teacher_logit in teacher_logits:
            kd_loss += kl_safe(student_logits, teacher_logit, tau=temperature)
        kd_loss /= len(teacher_logits)
        
        # Total loss
        total_loss = alpha * ce_loss + (1 - alpha) * kd_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'kd_loss': kd_loss.item()
        }
    
    def evaluate(self, dataloader):
        """
        Evaluate student model.
        
        Parameters:
        -----------
        dataloader : DataLoader
            Evaluation dataloader
            
        Returns:
        --------
        float
            Accuracy
        """
        self.student.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.student(inputs)
                logits = outputs["logit"] if isinstance(outputs, dict) else outputs
                _, predicted = torch.max(logits.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        accuracy = 100.0 * correct / total
        return accuracy


def _get_cfg_val(cfg: dict, key: str, default):
    """cfg 또는 cfg["method"]·cfg["method"]["method"]에서 키를 순차 검색."""
    if key in cfg:
        return cfg[key]
    if "method" in cfg and key in cfg["method"]:
        return cfg["method"][key]
    if (
        "method" in cfg
        and "method" in cfg["method"]
        and key in cfg["method"]["method"]
    ):
        return cfg["method"]["method"][key]
    return default

def student_distillation_update(
    teacher_wrappers,
    mbm, synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler,
    global_ep: int = 0
):
    """Train the student model via knowledge distillation.

    The teachers and MBM are frozen to generate synergy logits while the
    student is optimized using a combination of cross-entropy and KD losses.
    If the MBM operates in query mode, the optional feature-level KD term
    aligns the student query with the teacher attention output in the MBM
    latent space.
    """
    # 1) freeze teacher + mbm
    teacher_reqgrad_states = []
    teacher_train_states = []
    for tw in teacher_wrappers:
        teacher_train_states.append(tw.training)
        states = []
        for p in tw.parameters():
            states.append(p.requires_grad)
            p.requires_grad = False
        teacher_reqgrad_states.append(states)
        tw.eval()

    mbm_reqgrad_states = []
    mbm_train_state = mbm.training
    for p in mbm.parameters():
        mbm_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    mbm.eval()

    syn_reqgrad_states = []
    syn_train_state = synergy_head.training
    for p in synergy_head.parameters():
        syn_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    synergy_head.eval()

    # ------------------------------------------------------------
    # 학생 Epoch 결정 로직 – schedule 우선, 없으면 student_iters 만 사용
    # ------------------------------------------------------------
    if "student_epochs_schedule" in cfg:
        cur_stage_idx = int(cfg.get("cur_stage", 1)) - 1   # 0-base
        try:
            student_epochs = int(cfg["student_epochs_schedule"][cur_stage_idx])
        except (IndexError, ValueError, TypeError):
            raise ValueError(
                "[trainer_student] student_epochs_schedule가 "
                f"stage {cur_stage_idx+1} 에 대해 정의돼 있지 않습니다."
            )
    else:
        student_epochs = int(cfg.get("student_iters", 1))   # 최후 fallback

    # ──────────────────────────────────────────────────────────
    stage_meter = StageMeter(cfg.get("cur_stage", 1), logger, cfg, student_model)
    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    logger.info(f"[StudentDistill] Using student_epochs={student_epochs}")

    autocast_ctx, scaler = get_amp_components(cfg)
    # ---------------------------------------------------------
    # IB‑MBM forward
    # ---------------------------------------------------------
    # no attention weights returned in simplified IB-MBM
    for ep in range(student_epochs):
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
        else:
            total_epochs = cfg.get("total_epochs", 1)
        cur_tau = get_tau(
            cfg,
            epoch=global_ep + ep,
            total_epochs=total_epochs,
        )
        distill_loss_sum = 0.0
        cnt = 0
        student_model.train()

        mix_mode = (
            "cutmix"
            if cfg.get("cutmix_alpha_distill", 0.0) > 0.0
            else "mixup" if cfg.get("mixup_alpha", 0.0) > 0.0
            else "none"
        )

        for step, (x, y) in enumerate(
            smart_tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]")
        ):
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            if mix_mode == "cutmix":
                x_mixed, y_a, y_b, lam = cutmix_data(
                    x, y, alpha=cfg["cutmix_alpha_distill"]
                )
            elif mix_mode == "mixup":
                x_mixed, y_a, y_b, lam = mixup_data(x, y, alpha=cfg["mixup_alpha"])
            else:
                x_mixed, y_a, y_b, lam = x, y, y, 1.0

            with autocast_ctx:
                # (A) Student forward (query)
                feat_dict, s_logit, _ = student_model(x_mixed)

                with torch.no_grad():
                    t1_out = teacher_wrappers[0](x_mixed)
                    t2_out = teacher_wrappers[1](x_mixed)
                    t1_dict = t1_out[0] if isinstance(t1_out, tuple) else t1_out
                    t2_dict = t2_out[0] if isinstance(t2_out, tuple) else t2_out

                    feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) \
                               else "feat_2d"
                    f1_2d = t1_dict[feat_key]
                    f2_2d = t2_dict[feat_key]

                s_feat = feat_dict[cfg.get("feat_kd_key", "feat_2d")]
                syn_feat, mu, logvar = mbm(
                    s_feat, torch.stack([f1_2d, f2_2d], dim=1)
                )
                if cfg.get("use_ib", False):
                    ib_beta = get_beta(cfg, global_ep + ep)
                    mu, logvar = mu.float(), logvar.float()
                    ib_loss_val = ib_loss(mu, logvar, beta=ib_beta)
                else:
                    ib_loss_val = torch.tensor(0.0, device=cfg["device"])
                fsyn = syn_feat
                zsyn = synergy_head(fsyn)

                # stable CE/KL calculations in float32
                loss_ce = ce_safe(
                    s_logit,
                    y,
                    ls_eps=cfg.get("label_smoothing", 0.0),
                )
                loss_kd = kl_safe(
                    s_logit,
                    zsyn,
                    tau=cur_tau,
                )

                # ---- DEBUG: 첫 batch 모양 확인 ----
                if ep == 0 and step == 0 and cfg.get("debug_verbose", False):
                    logging.debug(
                        "[DBG/student] x %s s_logit %s zsyn %s",
                        tuple(x.shape),
                        tuple(s_logit.shape),
                        tuple(zsyn.shape),
                    )

            # ── (B1) sample-weights (always same dtype as losses) ───────────
            if cfg.get("use_disagree_weight", False):
                weights = sample_weights_from_disagreement(
                    t1_dict["logit"],
                    t2_dict["logit"],
                    y,
                    mode=cfg.get("disagree_mode", "pred"),
                    lambda_high=cfg.get("disagree_lambda_high", 1.0),
                    lambda_low=cfg.get("disagree_lambda_low", 1.0),
                )
            else:
                weights = torch.ones_like(y, dtype=s_logit.dtype, device=y.device)

            # AMP / float16 환경에서도 안전
            weights = weights.to(s_logit.dtype)

            if cfg.get("use_ib", False) and isinstance(mbm, IB_MBM):
                cw = certainty_weights(logvar).mean(dim=1).to(s_logit.dtype)
                weights = weights * cw

            # apply sample weights to CE and KD losses computed above
            ce_loss_val = loss_ce
            kd_loss_val = loss_kd

            # --- μ‑MSE with certainty weight ---------------------------------
            feat_kd_val = torch.tensor(0.0, device=cfg["device"])
            if cfg.get("feat_kd_alpha", 0) > 0:
                diff = (s_feat - mu).pow(2).sum(dim=1)   # ‖zs - μφ‖²
                feat_kd_val = (cw * diff).mean()

            loss = (
                _get_cfg_val(cfg, "ce_alpha", 1.0) * ce_loss_val
                + cfg["kd_alpha"] * kd_loss_val
                + cfg.get("feat_kd_alpha", 0.0) * feat_kd_val
                + ib_loss_val
            )
            
            # 추가 안전장치: loss가 너무 크면 clipping
            loss = torch.clamp(loss, 0.0, 100.0)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), cfg["grad_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.get("debug_verbose", False):
                    logging.debug(
                        "[StudentDistill] batch loss=%.4f", loss.item()
                    )
                if cfg.get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), cfg["grad_clip_norm"]
                    )
                optimizer.step()

            bs = x.size(0)
            distill_loss_sum += loss.item() * bs
            cnt += bs
            stage_meter.step(bs)

        ep_loss = distill_loss_sum / cnt

        # (C) validate
        test_acc = eval_student(student_model, testloader, cfg["device"], cfg)

        logging.info(
            "[StudentDistill ep=%d] loss=%.4f testAcc=%.2f best=%.2f",
            ep + 1,
            ep_loss,
            test_acc,
            best_acc,
        )
        if wandb and wandb.run:
            wandb.log({
                "student/loss": ep_loss,
                "student/acc": test_acc,
                "student/epoch": global_ep + ep + 1,
            })

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"student_ep{ep+1}_acc", test_acc)
        logger.update_metric(f"student_ep{ep+1}_loss", ep_loss)
        logger.update_metric(f"ep{ep+1}_mix_mode", mix_mode)
        logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)

        if scheduler is not None:
            scheduler.step()

        # (E) best snapshot
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(student_model.state_dict())

    student_model.load_state_dict(best_state)

    # restore original requires_grad and training states
    for tw, states, train_flag in zip(teacher_wrappers, teacher_reqgrad_states, teacher_train_states):
        for p, rg in zip(tw.parameters(), states):
            p.requires_grad = rg
        tw.train(train_flag)
    for p, rg in zip(mbm.parameters(), mbm_reqgrad_states):
        p.requires_grad = rg
    mbm.train(mbm_train_state)
    for p, rg in zip(synergy_head.parameters(), syn_reqgrad_states):
        p.requires_grad = rg
    synergy_head.train(syn_train_state)

    stage_meter.finish(best_acc)
    logger.info(f"[StudentDistill] bestAcc={best_acc:.2f}")
    return best_acc

@torch.no_grad()
def eval_student(model, loader, device, cfg=None):
    autocast_ctx, _ = get_amp_components(cfg or {})
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            feat_dict, s_logit, _ = model(x)
        pred = s_logit.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return 100.*correct/total
