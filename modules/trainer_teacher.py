# modules/trainer_teacher.py

import torch
import torch.nn as nn
import copy
import logging
from torch.nn import functional as F
from contextlib import nullcontext
from utils.common import smart_tqdm, get_amp_components
from utils.training import get_tau, get_beta
from models import IB_MBM

from modules.losses import (
    kd_loss_fn,
    ce_loss_fn,
    ib_loss,
    soft_clip_loss,
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
    update_logger: bool = False,
    logger=None,
):
    """Evaluate synergy accuracy (AMP off, float32).

    If ``update_logger`` is False, this function will not touch logger/cfg.
    When the IB_MBM operates in query mode, ``student_model`` must be provided so
    that the student features can be used as the attention query.
    """

    # Guard: need at least one teacher (K>=1). If K==1, build KV with a single teacher.
    try:
        if teacher_wrappers is None or len(teacher_wrappers) < 1:
            return -1.0
    except Exception:
        return -1.0
    # Force AMP off for numerically stable evaluation (explicit autocast disabled)
    try:
        autocast_ctx = torch.autocast(device_type="cuda", enabled=False)
    except Exception:
        autocast_ctx = nullcontext()
    correct, total = 0, 0
    first = True
    # Temporarily set eval() for teachers, IB_MBM, synergy_head, and student
    prev_teach_modes = []
    try:
        for tw in teacher_wrappers:
            prev_teach_modes.append(getattr(tw, 'training', False))
            tw.eval()
    except Exception:
        prev_teach_modes = []
    prev_ib_mode = getattr(ib_mbm, 'training', False)
    prev_syn_mode = getattr(synergy_head, 'training', False)
    prev_student_mode = getattr(student_model, 'training', False) if student_model is not None else False
    try:
        ib_mbm.eval(); synergy_head.eval()
        if student_model is not None:
            student_model.eval()
    except Exception:
        pass
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            # Collect teacher features (support K=1 or K>=2)
            feats = []
            key = (
                "distill_feat"
                if (cfg or {}).get("use_distillation_adapter", False)
                else "feat_2d"
            )
            for tw in teacher_wrappers:
                out = tw(x)
                t_dict = out[0] if isinstance(out, tuple) else out
                feats.append(t_dict[key].float())

            assert student_model is not None, "student_model required for IB_MBM"
            s_feat = student_model(x)[0][(cfg or {}).get("feat_kd_key", "feat_2d")].float()
            # eval 안정화: z = mu 모드 사용(sample=False)
            # Evaluate synergy using μ (mean) to avoid sample noise
            if len(feats) == 1:
                kv = torch.stack([feats[0]], dim=1)
            else:
                # Optional K>2 handling: use all teachers or select specific indices
                use_all = bool((cfg or {}).get("synergy_eval_use_all_teachers", False))
                idx_list = (cfg or {}).get("synergy_eval_teacher_indices", None)
                if isinstance(idx_list, (list, tuple)) and len(idx_list) > 0:
                    picked = []
                    for i in idx_list:
                        try:
                            if 0 <= int(i) < len(feats):
                                picked.append(feats[int(i)])
                        except Exception:
                            continue
                    if len(picked) == 0:
                        # Fallback to first two if indices invalid
                        if len(feats) >= 2:
                            picked = [feats[0], feats[1]]
                        else:
                            picked = [feats[0]]
                    kv = torch.stack(picked, dim=1)
                elif use_all and len(feats) > 2:
                    kv = torch.stack(feats, dim=1)
                else:
                    # Default: use first two
                    if len(feats) > 2 and first:
                        logging.info(
                            "[eval_synergy] K=%d teachers detected. Using first 2. Set synergy_eval_use_all_teachers=true or provide synergy_eval_teacher_indices to override.",
                            len(feats),
                        )
                    kv = torch.stack([feats[0], feats[1]], dim=1)
            # Deterministic evaluation: sample=False and float32 path
            fsyn, mu, _ = ib_mbm(s_feat, kv, sample=False)
            zsyn = synergy_head(mu).float()
            if first:
                if len(feats) >= 2:
                    logging.info(
                        "[eval_synergy] t1=%s t2=%s s=%s z=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
                        tuple(feats[0].shape), tuple(feats[1].shape), tuple(s_feat.shape), tuple(zsyn.shape),
                        zsyn.mean().item(), zsyn.std().item(), zsyn.min().item(), zsyn.max().item(),
                    )
                else:
                    logging.info(
                        "[eval_synergy] t1=%s s=%s z=%s mean=%.4f std=%.4f min=%.4f max=%.4f",
                        tuple(feats[0].shape), tuple(s_feat.shape), tuple(zsyn.shape),
                        zsyn.mean().item(), zsyn.std().item(), zsyn.min().item(), zsyn.max().item(),
                    )
                first = False

        pred = zsyn.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    acc = 100.0 * correct / total if total > 0 else 0
    # restore modes
    try:
        if prev_teach_modes:
            for tw, m in zip(teacher_wrappers, prev_teach_modes):
                tw.train(m)
        ib_mbm.train(prev_ib_mode)
        synergy_head.train(prev_syn_mode)
        if student_model is not None:
            student_model.train(prev_student_mode)
    except Exception:
        pass
    # 선택적으로 EMA 업데이트 (A-Step에서만)
    if update_logger and logger is not None:
        try:
            ema_alpha = float((cfg or {}).get("synergy_ema_alpha", 0.8))
        except Exception:
            ema_alpha = 0.8
        # 존재하지 않으면 음수(-1.0)로 간주하여 게이트가 자동으로 열리지 않도록 함
        prev = -1.0
        try:
            v = None
            if hasattr(logger, "get_metric"):
                v = logger.get_metric("last_synergy_acc", (cfg or {}).get("last_synergy_acc", None))
            elif cfg is not None:
                v = (cfg or {}).get("last_synergy_acc", None)
            if v is not None:
                prev = float(v)
        except Exception:
            prev = -1.0
        # prev가 음수(미측정)면 현재 측정값으로 초기화
        if prev < 0.0:
            ema = float(acc) / 100.0
        else:
            ema = ema_alpha * prev + (1.0 - ema_alpha) * (float(acc) / 100.0)
        logger.update_metric("last_synergy_acc", float(ema))
        logger.update_metric("last_synergy_acc_pct", float(ema * 100.0))
        if cfg is not None:
            cfg["last_synergy_acc"] = float(ema)
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
    # A-Step은 teacher_adapt_epochs>0이면 항상 실행
    if int(cfg.get("teacher_adapt_epochs", 0)) <= 0:
        logger.info("[A-Step] Skipped: teacher_adapt_epochs=0")
        return 0.0
    # 디버깅용 가드 상태 로그(참고용)
    try:
        use_ib = bool(cfg.get("use_ib", False))
        kd_mode = str((cfg or {}).get("kd_target", "avg")).lower()
        use_cccp_a = bool(cfg.get("use_cccp_in_a", cfg.get("use_cccp", False)))
        logging.info(f"[A-Step guard] use_ib={use_ib} kd_mode={kd_mode} use_cccp_in_a={use_cccp_a}")
    except Exception:
        pass

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
    # Ensure synergy_only_epochs ≤ teacher_epochs
    cfg_syn_only = int(cfg.get("synergy_only_epochs", 0))
    syn_only_epochs = min(cfg_syn_only, teacher_epochs)
    if cfg_syn_only > teacher_epochs:
        logging.info(
            "[A-Step] synergy_only_epochs=%d truncated to %d to fit teacher_adapt_epochs=%d",
            cfg_syn_only, syn_only_epochs, teacher_epochs,
        )

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
    # Optional: allow turning off AMP specifically for A‑Step via config
    if not bool(cfg.get("a_step_amp_enabled", True)):
        autocast_ctx = nullcontext()
        scaler = None
    
    # teacher_epochs=0인 경우를 위한 기본값 설정
    teacher_loss_sum = 0.0
    count = 0
    
    for ep in range(teacher_epochs):
        # A-step CCCP 사용 여부(학생 정답 고정 참조는 기본 OFF)
        use_cccp_a = bool(cfg.get("use_cccp_in_a", cfg.get("use_cccp", False)))
        # 초반 CE-only 구간 (ep는 0-based)
        synergy_only = (ep < syn_only_epochs)
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
        try:
            logger.update_metric(f"teacher_ep{ep+1}_tau", float(cur_tau))
        except Exception:
            pass
        teacher_loss_sum = 0.0
        count = 0

        for step, batch in enumerate(
            smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]")
        ):
            x, y = batch
            x, y = x.to(cfg["device"], non_blocking=True), y.to(cfg["device"], non_blocking=True)
            
            # CE-only window advisory (synergy_only_epochs): recommend synergy_ce_alpha=1.0
            if step == 0 and synergy_only:
                try:
                    _syn_alpha = float(cfg.get("synergy_ce_alpha", 0.6))
                    if _syn_alpha < 1.0:
                        logging.warning(
                            "[A-Step ep=%d] synergy_only_epochs active but synergy_ce_alpha=%.2f < 1.00. For CE-only warmup, consider setting synergy_ce_alpha=1.0 in YAML.",
                            ep + 1,
                            _syn_alpha,
                        )
                except Exception:
                    pass

            # 채널-라스트 포맷 적용 (Conv 연산 가속)
            if cfg.get("use_channels_last", True) and x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)

            with autocast_ctx:
                # (A) Student features and logits (kept fixed) – cache s_out for CCCP
                with torch.no_grad():
                    feat_dict, s_logit, s_out = student_model(x)
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
                            logging.warning(
                                "Teacher %d feature dim mismatch: %d vs %d",
                                i, feat.size(1), target_dim,
                            )
                            # 차원 맞춤은 호출자 측 adapter로 해결해야 함. 여기서는 오류만 보고.
                            raise RuntimeError("Teacher feature dim mismatch – check adapters/distill_out_dim")
                
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
                # Use deterministic μ for A-Step to remove sampling noise
                syn_feat, mu, logvar = ib_mbm(s_feat, kv, sample=False)
                ib_loss_val = 0.0
                if cfg.get("use_ib", False):
                    mu, logvar = mu.float(), logvar.float()
                    # μ/σ 안정화: logvar 클리핑
                    try:
                        clip_lv = float(cfg.get("ib_mbm_logvar_clip", 6))
                        logvar = logvar.clamp(-clip_lv, clip_lv)
                    except Exception:
                        pass
                    # raw KL (per-batch mean) for monitoring
                    raw_kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
                    ib_beta = get_beta(cfg, global_ep + ep)
                    ib_loss_val = ib_loss(mu, logvar, beta=ib_beta)
                    if step == 0:
                        try:
                            logger.update_metric(f"teacher_ep{ep+1}_ib_beta", float(ib_beta))
                            logger.update_metric(f"teacher_ep{ep+1}_ib_kld", float(raw_kld.item()))
                        except Exception:
                            pass
                # Feed μ to synergy head for stable logits
                fsyn = mu
                zsyn = synergy_head(fsyn)

                # (D) compute loss (KL + synergyCE)
                if cfg.get("use_ib", False) and isinstance(ib_mbm, IB_MBM):
                    ce_vec = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                        reduction="none",
                    )
                    # certainty weights from IB logvar
                    # ---- DEBUG: 첫 batch 모양 확인 ----
                    if ep == 0 and step == 0 and cfg.get("debug_verbose", False):
                        logging.debug(
                            "[DBG/teacher] t1_logit %s s_logit %s zsyn %s",
                            tuple(t1_dict["logit"].shape),
                            tuple(s_logit.shape),
                            tuple(zsyn.shape),
                        )
                    cw = torch.exp(-logvar).detach().mean(dim=1).to(zsyn.dtype)
                    # A-Step: 증폭 금지(상한 1.0), 하한은 cfg.min_cw 적용
                    try:
                        cw_min = float(cfg.get("min_cw", 0.1))
                    except Exception:
                        cw_min = 0.1
                    cw = cw.clamp(cw_min, 1.0)
                    # Optional monitoring on first step of each epoch
                    if step == 0:
                        try:
                            logger.update_metric(f"ep{ep+1}_cw_mean", float(cw.mean().item()))
                            logger.update_metric(f"ep{ep+1}_cw_min", float(cw.min().item()))
                            logger.update_metric(f"ep{ep+1}_cw_max", float(cw.max().item()))
                            logger.update_metric(f"ep{ep+1}_logvar_mean", float(logvar.mean().item()))
                        except Exception:
                            pass
                    loss_ce = (cw * ce_vec).mean()
                    # KD gate (pre-compute to skip KD work when closed)
                    thr = float(cfg.get("enable_kd_after_syn_acc", 0.0))
                    if thr > 1.5:
                        thr /= 100.0
                    try:
                        last_syn = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0)) or 0.0) if hasattr(logger, "get_metric") else float(cfg.get("last_synergy_acc", 0.0))
                    except Exception:
                        last_syn = float(cfg.get("last_synergy_acc", 0.0))
                    kd_weight_eff = float(cfg.get("teacher_adapt_alpha_kd", cfg.get("kd_alpha", 1.0)))
                    kd_gate_open = (not synergy_only) and (kd_weight_eff > 0.0) and (thr <= 0.0 or last_syn >= thr)
                    if kd_gate_open:
                        kd_raw = kd_loss_fn(zsyn, s_logit, T=cur_tau, reduction="none")
                        kd_vec = kd_raw.sum(dim=1) if kd_raw.dim() == 2 else kd_raw
                        loss_kd = (cw * kd_vec).mean()
                    else:
                        loss_kd = torch.zeros((), device=zsyn.device)
                    # CE-only 구간: cw 비활성화 및 IB-KL/KD OFF
                    if synergy_only:
                        loss_ce = ce_vec.mean()
                        loss_kd = torch.zeros((), device=zsyn.device)
                        ib_loss_val = 0.0
                else:
                    # No-IB path
                    # KD gate (pre-compute to skip KD work when closed)
                    thr = float(cfg.get("enable_kd_after_syn_acc", 0.0))
                    if thr > 1.5:
                        thr /= 100.0
                    try:
                        last_syn = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0)) or 0.0) if hasattr(logger, "get_metric") else float(cfg.get("last_synergy_acc", 0.0))
                    except Exception:
                        last_syn = float(cfg.get("last_synergy_acc", 0.0))
                    kd_weight_eff = float(cfg.get("teacher_adapt_alpha_kd", cfg.get("kd_alpha", 1.0)))
                    kd_gate_open = (not synergy_only) and (kd_weight_eff > 0.0) and (thr <= 0.0 or last_syn >= thr)
                    if kd_gate_open:
                        loss_kd = kd_loss_fn(zsyn, s_logit, T=cur_tau)
                    else:
                        loss_kd = torch.zeros((), device=zsyn.device)
                    loss_ce = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                    )
                    # CE-only 구간: KD OFF, IB는 사용 안 하므로 영향 없음
                    if synergy_only:
                        loss_kd = torch.zeros((), device=zsyn.device)

                # ① CE weight: force 1.0 during CE-only window
                synergy_weight_cfg = float(cfg.get("synergy_ce_alpha", 0.6))
                synergy_weight = 1.0 if synergy_only else synergy_weight_cfg
                synergy_ce_loss = synergy_weight * loss_ce

                # -----------------------------------------------
                #  Concave‑Convex surrogate (A-step에서 별도 플래그로 제어)
                #  - synergy_only_epochs 동안은 CCCP OFF
                #  - (선택) enable_kd_after_syn_acc 임계치 충족 시에만 ON
                # -----------------------------------------------
                kd_cccp = 0.0
                cccp_scale = 1.0
                if use_cccp_a and (not synergy_only) and (float(cfg.get("teacher_adapt_alpha_kd", cfg.get("kd_alpha", 1.0))) > 0.0):
                    syn_warm = int(cfg.get("synergy_only_epochs", 0))
                    if ep >= syn_warm:
                        thr = float(cfg.get("enable_kd_after_syn_acc", 0.0))
                        last_syn = 0.0
                        try:
                            if hasattr(logger, "get_metric"):
                                # logger 메트릭 우선, 없으면 cfg의 last_synergy_acc 사용
                                last_syn = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0)) or 0.0)
                            else:
                                last_syn = float(cfg.get("last_synergy_acc", 0.0))
                        except Exception:
                            last_syn = float(cfg.get("last_synergy_acc", 0.0))
                        if thr <= 0.0 or last_syn >= thr:
                            # Use cached s_out computed above
                            tau  = float(cfg.get("tau", 4.0))
                            kd_w = float(cfg.get("cccp_alpha", 0.0))
                            kd_cccp_raw = (
                                kd_w * tau * tau *
                                F.kl_div(
                                    F.log_softmax(zsyn / tau, dim=1),
                                    F.softmax((s_out if isinstance(s_out, torch.Tensor) else s_logit).detach() / tau, dim=1),
                                    reduction="batchmean",
                                )
                            )
                            # CCCP ramp-up after synergy_only window
                            try:
                                ramp = int(cfg.get("cccp_ramp_epochs", 2))
                            except Exception:
                                ramp = 2
                            t_lin = max(0, ep - syn_warm + 1) / max(1, ramp)
                            cccp_scale = float(min(1.0, t_lin))
                            kd_cccp = kd_cccp_raw * cccp_scale
                            # Optional CCCP loss soft clip
                            max_cccp = float(cfg.get("cccp_loss_max", 0.0))
                            if max_cccp and max_cccp > 0.0:
                                kd_cccp = soft_clip_loss(kd_cccp, max_cccp)
                            if ep == 0 and step == 0:
                                try:
                                    logging.info(
                                        "[A-Step ep=1/step=1][cccp] raw=%.4f scale=%.3f post=%.4f",
                                        float(kd_cccp_raw.detach().item()),
                                        float(cccp_scale),
                                        float(kd_cccp.detach().item()),
                                    )
                                except Exception:
                                    pass

                feat_kd_loss = torch.tensor(0.0, device=cfg["device"])
                feat_kd_on = (
                    float(cfg.get("feat_kd_alpha", 0.0)) > 0.0
                    or float(cfg.get("feat_kd_alpha_in_a", 0.0)) > 0.0
                )
                if feat_kd_on:
                    dims_match = s_feat.shape[1] == mu.shape[1]
                    first_batch = (ep == 0 and step == 0)
                    if dims_match:
                        diff = (s_feat - mu).pow(2).mean(dim=1)
                        cw = torch.exp(-logvar).detach().mean(dim=1).to(s_feat.dtype)
                        try:
                            cw_min = float(cfg.get("min_cw", 0.1))
                        except Exception:
                            cw_min = 0.1
                        cw = cw.clamp(cw_min, 1.0)
                        feat_kd_loss = (cw * diff).mean()
                    else:
                        if first_batch:
                            logging.warning(
                                "[A-Step] Feat-KD skipped (dim mismatch): s=%s, mu=%s",
                                tuple(s_feat.shape),
                                tuple(mu.shape),
                            )

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

                feat_kd_alpha_a = float(cfg.get("feat_kd_alpha_in_a", 0.0))
                total_loss_step = (
                    kd_weight * loss_kd
                    + synergy_ce_loss
                    + feat_kd_alpha_a * feat_kd_loss
                    + ib_loss_val
                    + float(cfg.get("reg_lambda", 0.0)) * reg_loss
                    + float(cfg.get("ib_mbm_reg_lambda", 0.0)) * ib_mbm_reg_loss
                    + kd_cccp  # CCCP surrogate 추가
                )
                total_loss_step_pre = None
                if ep == 0 and step == 0:
                    try:
                        total_loss_step_pre = float(total_loss_step.detach().item())
                    except Exception:
                        total_loss_step_pre = None

                # Optional: regularize learnable temperature (if enabled)
                try:
                    st_reg = float(cfg.get("synergy_temp_reg", 0.0))
                except Exception:
                    st_reg = 0.0
                if st_reg > 0 and hasattr(synergy_head, "log_temp"):
                    total_loss_step = total_loss_step + st_reg * (synergy_head.log_temp ** 2)

                # ep==1 첫 스텝에 튠용 요약 로그 한 줄 남기기 (KD 가중치는 실효값으로 표기)
                if ep == 0 and step == 0:
                    try:
                        thr = float(cfg.get("enable_kd_after_syn_acc", 0.0))
                        if thr > 1.5:
                            thr /= 100.0
                        last_syn = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0)) or 0.0) if hasattr(logger, "get_metric") else float(cfg.get("last_synergy_acc", 0.0))
                        kd_gate_on = int(thr <= 0.0 or last_syn >= thr)
                        kd_weight_eff = float(kd_weight)
                        if synergy_only or kd_weight_eff == 0.0 or (thr > 0.0 and last_syn < thr):
                            kd_weight_eff = 0.0
                        raw_kld_val = None
                        if cfg.get("use_ib", False):
                            raw_kld_val = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean().item()
                        logging.info(
                            "[A-Step ep=1/step=1] kd_gate_on=%d kd_weight=%.3f synergy_ce_alpha=%.3f cw_mean=%.4f cw_std=%.4f raw_kld=%s ib_beta=%.6f",
                            kd_gate_on,
                            float(kd_weight_eff),
                            float(synergy_weight),
                            float(cw.mean().item()) if 'cw' in locals() else float('nan'),
                            float(cw.std().item()) if 'cw' in locals() else float('nan'),
                            f"{raw_kld_val:.4f}" if raw_kld_val is not None else "NA",
                            float(get_beta(cfg, global_ep + ep)) if cfg.get("use_ib", False) else 0.0,
                        )
                    except Exception:
                        pass
                
                # 추가 안전장치: loss가 너무 크면 clipping
                # Loss 클리핑 완화 (A-Step 안정화를 위해)
            # A-step loss clamp (disabled by default; enable by setting disable_loss_clamp_in_a=false)
            disable_in_a = bool(cfg.get("disable_loss_clamp_in_a", True))
            if cfg.get("use_loss_clamp", False) and not disable_in_a:
                max_v = float(cfg.get("loss_clamp_max", 1000.0))
                mode = str(cfg.get("loss_clamp_mode", "soft"))
                lc_warm = int(cfg.get("loss_clamp_warmup_epochs", 0))
                if (global_ep + ep) < lc_warm:
                    pass  # 워밍업 동안 clamp 미적용
                else:
                    if mode == "soft":
                        from modules.losses import soft_clip_loss
                        total_loss_step = soft_clip_loss(total_loss_step, max_v)
                    else:
                        total_loss_step = torch.clamp(total_loss_step, 0.0, max_v)
            else:
                # 기본적으로는 클리핑 해제
                pass

            if ep == 0 and step == 0:
                try:
                    post_v = float(total_loss_step.detach().item())
                    pre_v = float(total_loss_step_pre) if total_loss_step_pre is not None else float("nan")
                    logging.info(
                        "[A-Step ep=1/step=1][loss] pre=%.4f post=%.4f cccp_scale=%.3f",
                        pre_v, post_v, float(cccp_scale)
                    )
                except Exception:
                    pass

            # ---- (2) per-batch Optim ----
            optimizer.zero_grad(set_to_none=True)
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

        # ==== epoch end ====
        ep_loss = teacher_loss_sum / max(1, count)

        # synergy_eval: 비용 절감을 위해 2ep 간격 평가 지원
        do_eval = (testloader is not None) and (((ep + 1) % int(cfg.get("teacher_eval_every", 2))) == 0)
        if do_eval:
            synergy_test_acc = eval_synergy(
                teacher_wrappers,
                ib_mbm,
                synergy_head,
                loader=testloader,
                device=cfg["device"],
                cfg=cfg,
                student_model=student_model,
                update_logger=True,
                logger=logger,
            )
        else:
            synergy_test_acc = logger.get_metric("last_synergy_acc_pct", -1) or -1

        logger.info(
            f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}"
        )

        # eval_synergy(update_logger=True)에서 EMA 갱신을 이미 처리함.
        # 여기서는 측정값 로깅만 수행
        try:
            if synergy_test_acc is not None and float(synergy_test_acc) >= 0:
                logger.update_metric(f"teacher_ep{ep+1}_syn_acc", float(synergy_test_acc))
        except Exception:
            pass

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"teacher_ep{ep+1}_loss", ep_loss)
        logger.update_metric(f"teacher_ep{ep+1}_syn_acc", synergy_test_acc)
        logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)

        # per-epoch scheduler step
        if scheduler is not None:
            scheduler.step()

        # best snapshot 업데이트
        if synergy_test_acc > best_synergy:
            best_synergy = synergy_test_acc
            # Save best snapshot to CPU to avoid VRAM fragmentation during long A‑Step
            best_state["teacher_wraps"] = [
                _cpu_state_dict(tw) for tw in teacher_wrappers
            ]
            best_state["ib_mbm"] = _cpu_state_dict(ib_mbm)
            best_state["syn_head"] = _cpu_state_dict(synergy_head)

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
            # 현재 상태를 backup으로 저장 (CPU로 저장하여 VRAM 파편화 방지)
            backup_state["teacher_wraps"] = [
                _cpu_state_dict(tw) for tw in teacher_wrappers
            ]
            backup_state["ib_mbm"] = _cpu_state_dict(ib_mbm)
            backup_state["syn_head"] = _cpu_state_dict(synergy_head)

    # restore best
    for i, tw in enumerate(teacher_wrappers):
        tw.load_state_dict(best_state["teacher_wraps"][i])
    ib_mbm.load_state_dict(best_state["ib_mbm"])
    synergy_head.load_state_dict(best_state["syn_head"])

    # A-Step 종료 로깅
    stage_info = f"[Stage {cfg.get('cur_stage', '?')}]" if 'cur_stage' in cfg else ""
    logger.info(f"{stage_info} A-Step (Teacher/IB) end - best_synergy={best_synergy:.2f}%")
    
    return best_synergy
