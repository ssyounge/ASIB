# modules/trainer_teacher.py

import torch
import copy
import logging
from torch.nn import functional as F
from utils.common import smart_tqdm, get_amp_components
from utils.training import get_tau, get_beta
from models import IB_MBM

from modules.losses import (
    kd_loss_fn,
    ce_loss_fn,
    ib_loss,
    certainty_weights,
    feat_mse_loss,
)


class TeacherTrainer:
    """Teacher trainer for knowledge distillation."""
    
    def __init__(self, teacher_models, device="cuda", config=None):
        """
        Initialize teacher trainer.
        
        Parameters:
        -----------
        teacher_models : list
            List of teacher models
        device : str
            Device to train on
        config : dict
            Training configuration
        """
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
            Optimizer for teacher models
            
        Returns:
        --------
        dict
            Training metrics
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        # Forward pass through teachers
        teacher_outputs = []
        for teacher in self.teachers:
            outputs = teacher(inputs)
            teacher_outputs.append(outputs)
        
        # Compute losses (simple cross-entropy for each teacher)
        total_loss = 0.0
        ce_losses = []
        
        for i, outputs in enumerate(teacher_outputs):
            logits = outputs["logit"] if isinstance(outputs, dict) else outputs
            ce_loss = F.cross_entropy(logits, targets)
            ce_losses.append(ce_loss.item())
            total_loss += ce_loss
        
        # Average loss over teachers
        total_loss /= len(self.teachers)
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'ce_losses': ce_losses,
            'avg_ce_loss': sum(ce_losses) / len(ce_losses)
        }
    
    def evaluate(self, dataloader):
        """
        Evaluate teacher models.
        
        Parameters:
        -----------
        dataloader : DataLoader
            Evaluation dataloader
            
        Returns:
        --------
        list
            List of accuracies for each teacher
        """
        accuracies = []
        
        for teacher in self.teachers:
            teacher.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = teacher(inputs)
                    logits = outputs["logit"] if isinstance(outputs, dict) else outputs
                    _, predicted = torch.max(logits.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            accuracy = 100.0 * correct / total
            accuracies.append(accuracy)
        
        return accuracies


def _cpu_state_dict(module: torch.nn.Module):
    """Return a copy of ``module.state_dict()`` on the CPU.

    Useful when saving snapshots to RAM to reduce GPU memory usage.
    """
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


@torch.no_grad()
def eval_synergy(
    teacher_wrappers,
    ib_mbm,
    synergy_head,
    loader,
    device="cuda",
    cfg=None,
    student_model=None,
):
    """Evaluate synergy accuracy.

    When the IB_MBM operates in query mode, ``student_model`` must be provided so
    that the student features can be used as the attention query.
    """

    autocast_ctx, _ = get_amp_components(cfg or {})
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            # ``BaseKDModel`` may return a tuple ``(feat_dict, logit, aux)``
            # for backward compatibility, extract the first element when needed.
            t1_out = teacher_wrappers[0](x)
            t2_out = teacher_wrappers[1](x)
            t1_dict = t1_out[0] if isinstance(t1_out, tuple) else t1_out
            t2_dict = t2_out[0] if isinstance(t2_out, tuple) else t2_out

            key = (
                "distill_feat"
                if (cfg or {}).get("use_distillation_adapter", False)
                else "feat_2d"
            )
            t1_feat = t1_dict[key]
            t2_feat = t2_dict[key]

            assert student_model is not None, "student_model required for IB_MBM"
            s_feat = student_model(x)[0][cfg.get("feat_kd_key", "feat_2d")]
            fsyn, _, _ = ib_mbm(s_feat, torch.stack([t1_feat, t2_feat], dim=1))
            zsyn = synergy_head(fsyn)

        pred = zsyn.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total if total > 0 else 0
    return acc


def teacher_adaptive_update(
    teacher_wrappers,
    ib_mbm,
    synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler,
    global_ep: int = 0,
):
    """
    - ``teacher_wrappers``: list containing ``teacher1`` and ``teacher2``.
    - ``ib_mbm`` and ``synergy_head``: assume partial freezing has been applied.
    - ``student_model``: kept fixed for knowledge distillation.
    - ``testloader``: optional loader used to evaluate synergy accuracy.
    """
    # 교사 백본 freeze 보장 (미세조정 OFF일 때)
    use_tf = bool(cfg.get("use_teacher_finetuning", False))
    only_da = cfg.get("train_distill_adapter_only", False)
    
    teacher_params = []
    for tw in teacher_wrappers:
        if not use_tf:
            # 교사 백본 고정 (기본값)
            for p in tw.parameters():
                p.requires_grad = False
            tw.eval()  # 교사를 eval 모드로 설정
        
        # optimizer에 추가할 파라미터 선택 (이미 optimizer에서 처리하지만 여기서도 확인)
        if use_tf:
            # 교사 전체 미세조정 허용
            for p in tw.parameters():
                if p.requires_grad:
                    teacher_params.append(p)
        elif only_da and hasattr(tw, "distillation_adapter"):
            # adapter만 학습
            for p in tw.distillation_adapter.parameters():
                p.requires_grad = True  # adapter는 학습 가능하게
                teacher_params.append(p)
        # else: 교사 백본은 학습하지 않음
    ib_mbm_params = [p for p in ib_mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    teacher_epochs = int(cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5)))

    best_synergy = -1
    best_state = {
        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
        "ib_mbm": _cpu_state_dict(ib_mbm),
        "syn_head": _cpu_state_dict(synergy_head),
    }

    # 추가 검증 로직: learning rate 조정을 위한 상태
    prev_obj = float("inf")
    backup_state = {
        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
        "ib_mbm": _cpu_state_dict(ib_mbm),
        "syn_head": _cpu_state_dict(synergy_head),
    }

    # A-Step 시작 로깅
    stage_info = f"[Stage {cfg.get('cur_stage', '?')}]" if 'cur_stage' in cfg else ""
    logger.info(f"{stage_info} A-Step (Teacher/IB) start - teacher_epochs={teacher_epochs}")
    
    # teacher_epochs=0인 경우 early return
    if teacher_epochs == 0:
        logger.info(f"{stage_info} A-Step (Teacher/IB) skipped - teacher_epochs=0")
        return 0.0  # 기본값 반환
    
    # student를 eval 모드로 고정하고 requires_grad=False
    if student_model is not None:
        student_model.eval()
        for p in student_model.parameters():
            p.requires_grad_(False)
    
    # ib_mbm, synergy_head만 train 모드로 (교사는 use_tf에 따라)
    if use_tf:
        # 교사 미세조정 모드에서만 train 모드로
        for tw in teacher_wrappers:
            tw.train()
    else:
        # 기본값: 교사는 eval 모드 유지
        for tw in teacher_wrappers:
            tw.eval()
    ib_mbm.train()
    synergy_head.train()

    autocast_ctx, scaler = get_amp_components(cfg)
    
    # teacher_epochs=0인 경우를 위한 기본값 설정
    teacher_loss_sum = 0.0
    count = 0
    
    for ep in range(teacher_epochs):
        # 교사 모델 모드 설정
        if use_tf:
            for tw in teacher_wrappers:
                tw.train()
        else:
            for tw in teacher_wrappers:
                tw.eval()
        ib_mbm.train()
        synergy_head.train()
        if student_model is not None:
            student_model.eval()
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
        else:
            total_epochs = cfg.get("total_epochs", 1)
        cur_tau = get_tau(
            cfg,
            epoch=global_ep + ep,
            total_epochs=total_epochs,
        )
        teacher_loss_sum = 0.0
        count = 0

        for step, batch in enumerate(
            smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]")
        ):
            x, y = batch
            x, y = x.to(cfg["device"], non_blocking=True), y.to(cfg["device"], non_blocking=True)
            
            # 채널-라스트 포맷 적용 (Conv 연산 가속)
            if cfg.get("use_channels_last", True) and x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)

            with autocast_ctx:
                # (A) Student features and logits (kept fixed)
                with torch.no_grad():
                    feat_dict, s_logit, _ = student_model(x)
                    key = cfg.get("feat_kd_key", "feat_2d")
                    s_feat = feat_dict[key]

                # (B) Teacher features
                feats_2d = []
                feat_key = (
                    "distill_feat"
                    if cfg.get("use_distillation_adapter", False)
                    else "feat_2d"
                )
                t1_dict = None
                for i, tw in enumerate(teacher_wrappers):
                    out = tw(x)
                    t_dict = out[0] if isinstance(out, tuple) else out
                    if i == 0:
                        t1_dict = t_dict
                    feat = t_dict[feat_key]
                    feats_2d.append(feat)
                    
                    # 차원 확인 로그 (첫 배치에서만)
                    if ep == 0 and step == 0:
                        logging.info(f"Teacher {i} {feat_key} shape: {feat.shape}")

                # (C) IB_MBM + synergy_head (IB_MBM only)
                # 차원 불일치 안전장치
                if len(feats_2d) > 0:
                    target_dim = feats_2d[0].size(1)
                    for i, feat in enumerate(feats_2d):
                        if feat.size(1) != target_dim:
                            logging.warning(f"Teacher {i} feature dim mismatch: {feat.size(1)} vs {target_dim}")
                            # 차원을 맞춰주기 위해 projection 추가
                            if not hasattr(self, f'feat_proj_{i}'):
                                setattr(self, f'feat_proj_{i}', 
                                       nn.Linear(feat.size(1), target_dim).to(feat.device))
                            proj_layer = getattr(self, f'feat_proj_{i}')
                            feats_2d[i] = proj_layer(feat)
                
                # 스택/검증 직전
                if len(feats_2d) < 2:
                    logging.error("[A-Step] need 2 teacher feats, got %d", len(feats_2d))
                    raise RuntimeError("Not enough teacher features")

                t1, t2 = feats_2d[0], feats_2d[1]
                if t1.shape[1] != t2.shape[1]:
                    logging.error("[A-Step] KV dim mismatch: t1=%s, t2=%s. Check use_distillation_adapter/distill_out_dim.",
                                  tuple(t1.shape), tuple(t2.shape))
                    raise RuntimeError("KV dim mismatch")

                # (선택) 쿼리 q_dim도 점검
                if getattr(ib_mbm.q_proj, "in_features", None) != s_feat.shape[1]:
                    logging.error("[A-Step] q_dim mismatch: q_in=%s, s_feat=%s",
                                  getattr(ib_mbm.q_proj, "in_features", None), s_feat.shape[1])
                    raise RuntimeError("Q dim mismatch")

                # A-Step에서 shape 체크 (첫 배치에서만)
                if ep == 0 and step == 0:
                    logging.info("[A-Step] t1=%s, t2=%s, s_feat=%s, q_in=%s",
                                 tuple(t1.shape), tuple(t2.shape), tuple(s_feat.shape), 
                                 getattr(ib_mbm.q_proj, "in_features", None))

                kv = torch.stack([t1, t2], dim=1)
                syn_feat, mu, logvar = ib_mbm(s_feat, kv)
                ib_loss_val = 0.0
                if cfg.get("use_ib", False):
                    mu, logvar = mu.float(), logvar.float()
                    ib_beta = get_beta(cfg, global_ep + ep)
                    ib_loss_val = ib_loss(mu, logvar, beta=ib_beta)
                fsyn = syn_feat
                zsyn = synergy_head(fsyn)

                # (D) compute loss (KL + synergyCE)
                if cfg.get("use_ib", False) and isinstance(ib_mbm, IB_MBM):
                    ce_vec = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                        reduction="none",
                    )
                    kd_vec = kd_loss_fn(zsyn, s_logit, T=cur_tau, reduction="none").sum(
                        dim=1
                    )

                    # ---- DEBUG: 첫 batch 모양 확인 ----
                    if ep == 0 and step == 0 and cfg.get("debug_verbose", False):
                        logging.debug(
                            "[DBG/teacher] t1_logit %s s_logit %s zsyn %s",
                            tuple(t1_dict["logit"].shape),
                            tuple(s_logit.shape),
                            tuple(zsyn.shape),
                        )
                    cw = certainty_weights(logvar).mean(dim=1).to(zsyn.dtype)
                    loss_ce = (cw * ce_vec).mean()
                    loss_kd = (cw * kd_vec).mean()
                else:
                    loss_kd = kd_loss_fn(zsyn, s_logit, T=cur_tau)
                    loss_ce = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                    )

                # ① 누락 시 기본값 0.6
                synergy_weight = cfg.get("synergy_ce_alpha", 0.6)
                synergy_ce_loss = synergy_weight * loss_ce

                # -----------------------------------------------
                #  Concave‑Convex surrogate 추가
                #   · 학생 로짓 z_S 는 gradient 를 막아 고정값으로 취급  (∇h(x^t))
                #   · KL( z_syn || z_S_detached )  ← convex in θ_T
                # -----------------------------------------------
                kd_cccp = 0.0
                if cfg.get("use_cccp", True):
                    with torch.no_grad():
                        s_out = student_model(x)[1] if isinstance(student_model(x), tuple) else student_model(x)
                    tau  = float(cfg.get("tau", 4.0))
                    kd_w = float(cfg.get("kd_alpha", 0.0))
                    kd_cccp = (
                        kd_w * tau * tau *
                        F.kl_div(
                            F.log_softmax(zsyn / tau, dim=1),
                            F.softmax(s_out.detach() / tau, dim=1),
                            reduction="batchmean",
                        )
                    )

                feat_kd_loss = torch.tensor(0.0, device=cfg["device"])
                if cfg.get("feat_kd_alpha", 0) > 0:
                    diff = (s_feat - mu).pow(2).sum(dim=1)
                    cw = certainty_weights(logvar).mean(dim=1).to(s_feat.dtype)
                    feat_kd_loss = (cw * diff).mean()

                # ---- (1) 전체 손실 구성 ----
                kd_weight = cfg.get("teacher_adapt_alpha_kd", cfg.get("kd_alpha", 1.0))

                # 정규화 항은 batch-별로 포함
                reg_loss = (
                    torch.stack([(p ** 2).mean() for p in teacher_params]).mean()
                    if teacher_params
                    else torch.tensor(0.0, device=cfg["device"])
                )
                ib_mbm_reg_params = ib_mbm_params + syn_params
                ib_mbm_reg_loss = (
                    torch.stack([(p ** 2).mean() for p in ib_mbm_reg_params]).mean()
                    if ib_mbm_reg_params
                    else torch.tensor(0.0, device=cfg["device"])
                )

                total_loss_step = (
                    kd_weight * loss_kd
                    + synergy_ce_loss
                    + cfg.get("feat_kd_alpha", 0) * feat_kd_loss
                    + ib_loss_val
                    + float(cfg.get("reg_lambda", 0.0)) * reg_loss
                    + float(cfg.get("ib_mbm_reg_lambda", 0.0)) * ib_mbm_reg_loss
                    + kd_cccp  # CCCP surrogate 추가
                )
                
                # 추가 안전장치: loss가 너무 크면 clipping
                # Loss 클리핑 완화 (A-Step 안정화를 위해)
            if cfg.get("use_loss_clamp", False):
                total_loss_step = torch.clamp(total_loss_step, 0.0, cfg.get("loss_clamp_max", 1000.0))
            else:
                # 기본적으로는 클리핑 해제
                pass

            # ---- (2) per-batch Optim ----
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss_step).backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    grad_params = teacher_params + ib_mbm_params + syn_params
                    if grad_params:
                        torch.nn.utils.clip_grad_norm_(
                            grad_params,
                            cfg["grad_clip_norm"],
                        )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss_step.backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    grad_params = teacher_params + ib_mbm_params + syn_params
                    if grad_params:
                        torch.nn.utils.clip_grad_norm_(
                            grad_params,
                            cfg["grad_clip_norm"],
                        )
                optimizer.step()

            teacher_loss_sum += total_loss_step.item() * x.size(0)
            count += x.size(0)

    ep_loss = teacher_loss_sum / count

    # synergy_eval
    if testloader is not None:
        synergy_test_acc = eval_synergy(
            teacher_wrappers,
            ib_mbm,
            synergy_head,
            loader=testloader,
            device=cfg["device"],
            cfg=cfg,
            student_model=student_model,
        )
    else:
        synergy_test_acc = -1

    logger.info(
        f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}"
    )

    # ── NEW: per-epoch logging ───────────────────────────────
    logger.update_metric(f"teacher_ep{ep+1}_loss", ep_loss)
    logger.update_metric(f"teacher_ep{ep+1}_synAcc", synergy_test_acc)
    logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)

    if scheduler is not None:
        scheduler.step()

    # best snapshot
    if synergy_test_acc > best_synergy:
        best_synergy = synergy_test_acc
        best_state["teacher_wraps"] = [
            copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers
        ]
        best_state["ib_mbm"] = copy.deepcopy(ib_mbm.state_dict())
        best_state["syn_head"] = copy.deepcopy(synergy_head.state_dict())

    # 추가 검증 로직: loss가 증가하면 learning rate 조정 및 상태 복원
    if ep_loss > prev_obj:
        logger.warning(f"[TeacherAdaptive] Loss increased from {prev_obj:.4f} to {ep_loss:.4f}, reducing LR and restoring state")
        for g in optimizer.param_groups:
            g['lr'] *= 0.5
        # 이전 상태로 복원
        for i, tw in enumerate(teacher_wrappers):
            tw.load_state_dict(backup_state["teacher_wraps"][i])
        ib_mbm.load_state_dict(backup_state["ib_mbm"])
        synergy_head.load_state_dict(backup_state["syn_head"])
    else:
        prev_obj = ep_loss
        # 현재 상태를 backup으로 저장
        backup_state["teacher_wraps"] = [
            copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers
        ]
        backup_state["ib_mbm"] = copy.deepcopy(ib_mbm.state_dict())
        backup_state["syn_head"] = copy.deepcopy(synergy_head.state_dict())

    # restore best
    for i, tw in enumerate(teacher_wrappers):
        tw.load_state_dict(best_state["teacher_wraps"][i])
    ib_mbm.load_state_dict(best_state["ib_mbm"])
    synergy_head.load_state_dict(best_state["syn_head"])

    # A-Step 종료 로깅
    stage_info = f"[Stage {cfg.get('cur_stage', '?')}]" if 'cur_stage' in cfg else ""
    logger.info(f"{stage_info} A-Step (Teacher/IB) end - best_synergy={best_synergy:.2f}%")
    
    return best_synergy
