# modules/trainer_student.py

import torch
import copy
import logging
from utils.common import smart_tqdm, mixup_data, cutmix_data, mixup_criterion, get_amp_components
from utils.training import get_tau, get_beta, StageMeter
from models import IB_MBM

from modules.loss_safe import ce_safe, kl_safe
from modules.losses import (
    ib_loss, certainty_weights, feat_mse_loss
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

                # stable CE/KL calculations (per-sample vector) then apply cw weights
                ls = cfg.get("ce_label_smoothing", cfg.get("label_smoothing", 0.0))
                ce_vec = ce_safe_vec(s_logit, y, ls_eps=ls)  # [B]
                # KD는 no-grad 타깃 사용 (필요할 때만)
                kd_loss_val = 0.0
                if cfg.get("kd_alpha", 0.0) > 0 and need_teachers:
                    # KD 타겟 선택
                    # 시너지 품질이 낮으면 avg로 대체
                    synergy_good = False
                    try:
                        synergy_good = float(cfg.get("last_synergy_acc", 0.0)) >= float(cfg.get("enable_kd_after_syn_acc", 0.0))
                    except Exception:
                        pass
                    kd_target = cfg.get("kd_target", "synergy")
                    if cfg.get("use_avg_until_synergy", False) and kd_target == "synergy" and not synergy_good:
                        kd_target = "avg"
                    if kd_target == "synergy" and zsyn_ng is not None:
                        kd_tgt = zsyn_ng
                    elif kd_target == "avg" and t1_dict is not None and t2_dict is not None:
                        # Teacher 평균
                        t1_logit = t1_dict["logit"]
                        t2_logit = t2_dict["logit"]
                        kd_tgt = (t1_logit + t2_logit) / 2.0
                    elif kd_target == "weighted_conf" and t1_dict is not None and t2_dict is not None:
                        # Teacher confidence 기반 가중 평균 (use .max(...).values; not .amax)
                        t1_logit = t1_dict["logit"]
                        t2_logit = t2_dict["logit"]
                        p1 = torch.softmax(t1_logit / cur_tau, dim=1).max(dim=1).values  # [B]
                        p2 = torch.softmax(t2_logit / cur_tau, dim=1).max(dim=1).values  # [B]
                        w1 = (p1 / (p1 + p2 + 1e-8)).unsqueeze(1)  # [B,1]
                        w2 = 1.0 - w1
                        kd_tgt = w1 * t1_logit + w2 * t2_logit
                    else:
                        # 기본값: synergy
                        kd_tgt = zsyn_ng if zsyn_ng is not None else torch.zeros_like(s_logit)

                    # Warmup 동안 시너지/앙상블 혼합 (ens_alpha 비율)
                    warmup_epochs = int(cfg.get("teacher_adapt_kd_warmup", 0))
                    ens_alpha = float(cfg.get("kd_ens_alpha", 0.0))
                    if (
                        warmup_epochs > 0 and ep < warmup_epochs and ens_alpha > 0.0
                        and zsyn_ng is not None and t1_dict is not None and t2_dict is not None
                    ):
                        t1_logit = t1_dict["logit"]
                        t2_logit = t2_dict["logit"]
                        avg_t = (t1_logit + t2_logit) / 2.0
                        kd_tgt = (1.0 - ens_alpha) * zsyn_ng + ens_alpha * avg_t
                    
                    if kd_tgt is not None:
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
                cw = cw.clamp(float(cfg.get("min_cw", 0.1)), 1.0)
                weights = weights * cw

            # apply cw to CE and KD
            ce_loss_val = (cw * ce_vec).mean()
            if isinstance(kd_loss_val, torch.Tensor) and kd_loss_val.dim() == 0:
                pass
            elif 'kd_vec' in locals() and isinstance(kd_vec, torch.Tensor):
                kd_loss_val = (cw * kd_vec).mean()

            # --- μ‑MSE with certainty weight ---------------------------------
            feat_kd_val = None
            if cfg.get("feat_kd_alpha", 0.0) > 0 and need_teachers and mu_ng is not None:
                # 차원 불일치 안전장치
                if s_feat.shape[1] == mu_ng.shape[1]:
                    diff = (s_feat - mu_ng).pow(2).sum(dim=1)  # s_feat에서 grad 흐름
                    feat_kd_val = (cw * diff).mean()
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
            
            # 추가 안전장치: loss가 너무 크면 clipping (클리핑 완화)
            if cfg.get("use_loss_clamp", False):
                loss = torch.clamp(loss, 0.0, cfg.get("loss_clamp_max", 1000.0))
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
