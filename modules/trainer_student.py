# modules/trainer_student.py

import torch
import torch.nn.functional as F
import copy
import logging
from utils.common import smart_tqdm, mixup_data, cutmix_data, mixup_criterion, get_amp_components
from utils.training import get_tau, get_beta, StageMeter
from models import IB_MBM

from modules.loss_safe import ce_safe, kl_safe
from modules.losses import (
    ib_loss, certainty_weights, feat_mse_loss, soft_clip_loss
)
from modules.disagreement import sample_weights_from_disagreement
from torch.amp import autocast


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


def ce_safe_vec(logits: torch.Tensor, target: torch.Tensor, ls_eps: float = 0.0) -> torch.Tensor:
    """Return per-sample CE loss in float32, with optional label smoothing.

    The output shape is [batch]. Autocast is disabled inside to avoid fp16 underflow.
    """
    with autocast('cuda', enabled=False):
        logits = logits.float()
        # Use PyTorch's native label_smoothing implementation for stability
        return torch.nn.functional.cross_entropy(
            logits,
            target,
            label_smoothing=float(ls_eps),
            reduction="none",
        )


def kl_safe_vec(p_logits: torch.Tensor, q_logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Return per-sample KL divergence (sum over classes), scaled by tau^2.

    Uses float32 math under autocast-disabled region for stability.
    """
    with autocast('cuda', enabled=False):
        p_log = torch.log_softmax(p_logits.float() / tau, dim=1)
        q = torch.softmax(q_logits.float() / tau, dim=1)
        kl = torch.nn.functional.kl_div(p_log, q, reduction="none")  # [B, C]
        return kl.sum(dim=1) * (tau * tau)


def get_kd_target(
    cfg: dict,
    logger,
    ep: int,
    zsyn_ng: torch.Tensor | None,
    t1_logit: torch.Tensor | None,
    t2_logit: torch.Tensor | None,
):
    """Select KD target with synergy gating and warmup mixing.

    - kd_target in {"synergy","auto"}: use synergy only when last_synergy_acc >= thr
    - During warmup (teacher_adapt_kd_warmup), blend synergy with avg using kd_ens_alpha
    - Fallback to avg if synergy is unavailable or gate not passed
    """
    kd_mode = cfg.get("kd_target", "avg")
    try:
        thr = float(cfg.get("enable_kd_after_syn_acc", 0.8))
    except Exception:
        thr = 0.8
    last_syn = 0.0
    try:
        if logger is not None and hasattr(logger, "get_metric"):
            last_syn = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0)) or 0.0)
        else:
            last_syn = float(cfg.get("last_synergy_acc", 0.0))
    except Exception:
        last_syn = float(cfg.get("last_synergy_acc", 0.0))
    warm = int(cfg.get("teacher_adapt_kd_warmup", 0))
    ens = float(cfg.get("kd_ens_alpha", 0.0))

    avg_t = None
    if t1_logit is not None and t2_logit is not None:
        avg_t = (t1_logit + t2_logit) / 2.0

    # normalize to 0-1 scale if needed
    try:
        if last_syn > 1.5:
            last_syn = last_syn / 100.0
        if thr > 1.5:
            thr = thr / 100.0
    except Exception:
        pass

    use_syn = (kd_mode in ("synergy", "auto")) and (zsyn_ng is not None) and (last_syn >= thr)
    if use_syn:
        if ep < warm and avg_t is not None and ens > 0.0:
            return (1.0 - ens) * zsyn_ng + ens * avg_t
        return zsyn_ng
    return avg_t

def student_distillation_update(
    teacher_wrappers,
    ib_mbm, synergy_head,
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

    The teachers and IB_MBM are frozen to generate synergy logits while the
    student is optimized using a combination of cross-entropy and KD losses.
    If the IB_MBM operates in query mode, the optional feature-level KD term
    aligns the student query with the teacher attention output in the IB_MBM
    latent space.
    """
    # 1) freeze teacher + ib_mbm
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

    ib_mbm_reqgrad_states = []
    ib_mbm_train_state = ib_mbm.training
    for p in ib_mbm.parameters():
        ib_mbm_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    ib_mbm.eval()

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

    # B-Step 시작 전 안전 복원 (A-Step에서 freeze된 경우 대비)
    for p in student_model.parameters():
        p.requires_grad_(True)
    student_model.train()
    
    # B-Step 시작 전 trainable 파라미터 수 기록
    s_train = sum(p.requires_grad for p in student_model.parameters())
    s_total = sum(1 for p in student_model.parameters())
    logger.info(f"[PPF][B-Step] student trainable: {s_train}/{s_total} ({100*s_train/s_total:.1f}%)")

    autocast_ctx, scaler = get_amp_components(cfg)
    # ---------------------------------------------------------
    # IB_MBM forward
    # ---------------------------------------------------------
    # no attention weights returned in simplified IB_MBM
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
        # τ 스케줄(선형 보간): [t0, t1]로 주어지면 B-step 내에서 2→4 등 선형 증가
        tau_sched = cfg.get("tau_schedule", None)
        if tau_sched and isinstance(tau_sched, (list, tuple)) and len(tau_sched) == 2:
            try:
                t0, t1 = float(tau_sched[0]), float(tau_sched[1])
                if "student_epochs_schedule" in cfg:
                    total_ep = int(cfg["student_epochs_schedule"][max(0, int(cfg.get("cur_stage",1))-1)])
                else:
                    total_ep = int(cfg.get("student_iters", 1))
                r = 0.0 if total_ep <= 1 else float(ep) / float(max(1, total_ep - 1))
                cur_tau = t0 + r * (t1 - t0)
            except Exception:
                pass
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
            x, y = x.to(cfg["device"], non_blocking=True), y.to(cfg["device"], non_blocking=True)
            
            # 채널-라스트 포맷 적용 (Conv 연산 가속)
            if cfg.get("use_channels_last", True) and x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)

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

                # 교사 모델이 필요한지 확인 (KD, Feature KD, IB 중 하나라도 사용하는 경우)
                need_teachers = (cfg.get("kd_alpha", 0.0) > 0.0) or \
                               (cfg.get("feat_kd_alpha", 0.0) > 0.0) or \
                               bool(cfg.get("use_ib", False))
                
                if need_teachers:
                    with torch.no_grad():
                        t1_out = teacher_wrappers[0](x_mixed)
                        t2_out = teacher_wrappers[1](x_mixed)
                        t1_dict = t1_out[0] if isinstance(t1_out, tuple) else t1_out
                        t2_dict = t2_out[0] if isinstance(t2_out, tuple) else t2_out

                        feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) \
                                   else "feat_2d"
                        f1_2d = t1_dict[feat_key]
                        f2_2d = t2_dict[feat_key]
                else:
                    # 교사 모델 forward를 스킵하는 경우 None으로 설정
                    t1_dict = t2_dict = None
                    f1_2d = f2_2d = None

                feat_kd_key = cfg.get("feat_kd_key", "feat_2d")
                s_feat = feat_dict[feat_kd_key]
                
                # 첫 배치에서 feat_kd_key와 차원 확인
                if ep == 0 and step == 0:
                    logging.info(f"[B-Step] feat_kd_key={feat_kd_key}, s_feat.shape={s_feat.shape}")
                
                # IB/KD 타깃이 필요할 때만 IB_MBM 실행 (need_teachers와 동일한 조건)
                need_ibm = need_teachers
                if need_ibm:
                    # 스택 전 shape 검증 (차원 불일치 조기 발견)
                    if f1_2d.shape[1] != f2_2d.shape[1]:
                        logging.error(f"[IB-MBM] KV dim mismatch: f1={f1_2d.shape}, f2={f2_2d.shape}. Check distill_out_dim and adapters.")
                        raise RuntimeError("KV dim mismatch")
                    
                    with torch.no_grad():
                        syn_feat_ng, mu_ng, logvar_ng = ib_mbm(
                            s_feat, torch.stack([f1_2d, f2_2d], dim=1)
                        )
                        zsyn_ng = synergy_head(syn_feat_ng)
                else:
                    mu_ng = logvar_ng = zsyn_ng = None

                # stable CE/KL calculations
                # CE: mixup/cutmix는 라벨 두 개(y_a,y_b)와 lambda로 직접 혼합
                ls = float(cfg.get("ce_label_smoothing", cfg.get("label_smoothing", 0.0)))
                if mix_mode in ("mixup", "cutmix"):
                    ce_loss_val = (
                        F.cross_entropy(s_logit.float(), y_a, label_smoothing=ls)
                        * lam
                        + F.cross_entropy(s_logit.float(), y_b, label_smoothing=ls)
                        * (1.0 - lam)
                    )
                else:
                    ce_vec = ce_safe_vec(s_logit, y, ls_eps=ls)  # [B]
                # KD는 no-grad 타깃 사용 (필요할 때만)
                kd_loss_val = 0.0
                if cfg.get("kd_alpha", 0.0) > 0 and need_teachers:
                    # KD 타겟 선택: synergy 게이팅 + 워밍업 혼합 + fallback avg
                    t1_logit = t1_dict["logit"] if t1_dict is not None else None
                    t2_logit = t2_dict["logit"] if t2_dict is not None else None
                    kd_tgt = get_kd_target(cfg, logger, ep, zsyn_ng, t1_logit, t2_logit)

                    # 혼합은 get_kd_target에서만 수행 (중복 혼합 방지)
                    
                    if kd_tgt is not None:
                        # KL with current KD temperature schedule
                        kd_vec = kl_safe_vec(s_logit, kd_tgt.detach(), tau=cur_tau)  # [B]
                        kd_loss_val = None  # set below after cw
                    else:
                        loss_kd = torch.tensor(0.0, device=s_logit.device)
                        kd_loss_val = loss_kd
                else:
                    loss_kd = torch.tensor(0.0, device=s_logit.device)
                    kd_loss_val = loss_kd

                # ---- DEBUG: 첫 batch 모양 확인 ----
                if ep == 0 and step == 0 and cfg.get("debug_verbose", False):
                    logging.debug(
                        "[DBG/student] x %s s_logit %s zsyn %s",
                        tuple(x.shape),
                        tuple(s_logit.shape),
                        tuple(zsyn_ng.shape) if 'zsyn_ng' in locals() and zsyn_ng is not None else None,
                    )

            # ── (B1) sample-weights (always same dtype as losses) ───────────
            if cfg.get("use_disagree_weight", False) and need_teachers and t1_dict is not None and t2_dict is not None:
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

            # cw 기본값 설정 (안전 가드)
            cw = torch.ones(y.size(0), device=s_logit.device, dtype=s_logit.dtype)

            if cfg.get("use_ib", False) and logvar_ng is not None:
                cw = torch.exp(-logvar_ng).mean(dim=1).to(s_logit.dtype)
                # 정규화: 평균을 1.0 부근으로 맞추고 가드 범위로 제한
                cw_mean = (cw.mean() + 1e-8)
                min_cw = float(cfg.get("min_cw", 0.1))
                max_cw = float(cfg.get("max_cw", 1.5))
                cw = (cw / cw_mean).clamp(min_cw, max_cw)
                weights = weights * cw

            # apply cw to CE and KD
            if 'ce_vec' in locals():
                ce_loss_val = (weights * ce_vec).mean()
            if isinstance(kd_loss_val, torch.Tensor) and kd_loss_val.dim() == 0:
                pass
            elif 'kd_vec' in locals() and isinstance(kd_vec, torch.Tensor):
                kd_loss_val = (weights * kd_vec).mean()

            # --- μ‑MSE with certainty weight (normalized) ---------------------
            feat_kd_val = None
            if cfg.get("feat_kd_alpha", 0.0) > 0 and need_teachers and mu_ng is not None:
                # 차원 불일치 안전장치
                if s_feat.shape[1] == mu_ng.shape[1]:
                    diff = (s_feat - mu_ng).pow(2).sum(dim=1)  # s_feat에서 grad 흐름
                    cw_feat = torch.exp(-logvar_ng).mean(dim=1).detach()
                    # 정규화: 평균 1.0 부근으로 맞추고 과도한 다운/업웨이팅 방지
                    cw_feat = (cw_feat / (cw_feat.mean() + 1e-8)).clamp(0.5, 1.5).to(s_feat.dtype)
                    feat_kd_val = (cw_feat * diff).mean()
                else:
                    # 첫 배치에서만 경고 출력 (로그 과다 방지)
                    if ep == 0 and step == 0:
                        logging.warning(
                            "[B-Step] Feat-KD skipped (dim mismatch): s=%s, mu=%s",
                            tuple(s_feat.shape),
                            tuple(mu_ng.shape),
                        )
                    feat_kd_val = None

            # IB KL은 B-Step에서 제외 (A-Step에서만 최적화)
            ib_loss_val = None
            # if cfg.get("use_ib_on_student", False):
            #     ib_loss_val = ib_loss(mu_ng, logvar_ng, beta=get_beta(cfg, global_ep + ep))
            
            # KD α 워밍업 (0→base)
            kd_alpha_base = float(cfg.get("kd_alpha", 0.0))
            kw = int(cfg.get("kd_warmup_epochs", 0))
            kd_alpha_eff = kd_alpha_base
            try:
                if kw > 0:
                    # τ 스케줄 r 재사용 가능, 아니면 ep 기준
                    if tau_sched and isinstance(tau_sched, (list, tuple)) and len(tau_sched) == 2:
                        if "student_epochs_schedule" in cfg:
                            total_ep = int(cfg["student_epochs_schedule"][max(0, int(cfg.get("cur_stage",1))-1)])
                        else:
                            total_ep = int(cfg.get("student_iters", 1))
                        r = 0.0 if total_ep <= 1 else float(ep) / float(max(1, total_ep - 1))
                        warm = min(1.0, r)
                    else:
                        warm = min(1.0, float(ep + 1) / float(kw))
                    kd_alpha_eff = kd_alpha_base * warm
            except Exception:
                kd_alpha_eff = kd_alpha_base

            # KD 손실 클램프: KD가 CE를 압도하지 않도록 스케일 (weights 적용 후 값 사용)
            kdmr = cfg.get("kd_max_ratio", None)
            if kdmr is not None and isinstance(kdmr, (int, float)) and kd_loss_val is not None:
                try:
                    with torch.no_grad():
                        ce_s = ce_loss_val.detach().clamp_min(1e-6)
                        kd_s = kd_loss_val.detach().clamp_min(1e-6)
                        scale = torch.clamp((float(kdmr) * ce_s) / kd_s, max=1.0)
                    kd_loss_val = kd_loss_val * scale
                except Exception:
                    pass

            # 최종 loss는 조건부 합 (상수 0 텐서 더하지 않기)
            loss = _get_cfg_val(cfg, "ce_alpha", 1.0) * ce_loss_val
            if kd_alpha_eff > 0 and (('kd_tgt' in locals() and kd_tgt is not None) or zsyn_ng is not None):
                loss = loss + kd_alpha_eff * kd_loss_val
            if feat_kd_val is not None:
                loss = loss + cfg.get("feat_kd_alpha", 0.0) * feat_kd_val
            
            # 추가 안전장치: loss가 너무 크면 clipping (soft/hard 선택) + 워밍업 게이트
            if cfg.get("use_loss_clamp", False):
                max_v = float(cfg.get("loss_clamp_max", 1000.0))
                mode = str(cfg.get("loss_clamp_mode", "soft"))
                lc_warm = int(cfg.get("loss_clamp_warmup_epochs", 0))
                if (global_ep + ep) < lc_warm:
                    pass  # 워밍업 동안 clamp 미적용
                else:
                    if mode == "soft":
                        loss = soft_clip_loss(loss, max_v)
                    else:
                        loss = torch.clamp(loss, 0.0, max_v)
            else:
                # 기본적으로는 클리핑 해제 (안정화를 위해)
                pass
            
            # loss.requires_grad 보증 (디버깅용)
            if not loss.requires_grad:
                logging.error(
                    "loss grad off: ce=%s kd=%s feat=%s",
                    ce_loss_val.requires_grad,
                    (kd_loss_val.requires_grad if isinstance(kd_loss_val, torch.Tensor) else None),
                    (feat_kd_val.requires_grad if isinstance(feat_kd_val, torch.Tensor) else None),
                )

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
    for p, rg in zip(ib_mbm.parameters(), ib_mbm_reqgrad_states):
        p.requires_grad = rg
    ib_mbm.train(ib_mbm_train_state)
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
