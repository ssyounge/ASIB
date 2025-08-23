# modules/trainer_student.py

import torch
import torch.nn.functional as F
import copy
import logging
from utils.common import smart_tqdm
from utils.common.misc import mixup_data, cutmix_data, get_amp_components
from utils.training import get_tau, get_beta, StageMeter

from modules.losses import (
    soft_clip_loss,
    dkd_loss,
)
from modules.disagreement import sample_weights_from_disagreement
from torch.amp import autocast
from utils.ema import EMATeacher


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

        if len(teacher_logits) > 0:
            kd_vals = [kl_safe_vec(student_logits, t_logit, tau=temperature).mean() for t_logit in teacher_logits]
            kd_loss = sum(kd_vals) / float(len(kd_vals))
        
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


def _interp_tau(cfg: dict, ep: int, total_ep: int) -> float:
    """Return tau using simple linear schedule if tau_schedule=[t0, t1] is provided.

    Fallback to constant tau when schedule is absent or invalid.
    """
    try:
        ts = cfg.get("tau_schedule", None)
        if ts and isinstance(ts, (list, tuple)) and len(ts) == 2:
            t0, t1 = float(ts[0]), float(ts[1])
            r = 0.0 if total_ep <= 1 else float(ep) / float(max(1, total_ep - 1))
            return t0 + r * (t1 - t0)
    except Exception:
        pass
    try:
        return float(cfg.get("tau", 4.0))
    except Exception:
        return 4.0


# Note: legacy get_kd_target helper removed; KD target selection is handled inline within
#       student_distillation_update for clarity and fewer indirections.

# --- synergy gating helper ---------------------------------------------------
def _synergy_gate_ok(cfg, logger) -> bool:
    """
    Return True if synergy quality passes the gate:
    last_synergy_acc >= enable_kd_after_syn_acc (handles 0~1 or % scale).
    """
    try:
        thr = float(cfg.get("enable_kd_after_syn_acc", 0.8))
    except Exception:
        thr = 0.8
    try:
        if logger is not None and hasattr(logger, "get_metric"):
            last = float(logger.get_metric("last_synergy_acc", cfg.get("last_synergy_acc", 0.0)) or 0.0)
        else:
            last = float(cfg.get("last_synergy_acc", 0.0))
    except Exception:
        last = float(cfg.get("last_synergy_acc", 0.0))
    # normalize mixed scales
    if thr > 1.5:
        thr /= 100.0
    if last > 1.5:
        last /= 100.0
    return last >= thr
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
    # 0) optional EMA teacher for KD-only ensemble
    ema_teacher = EMATeacher(student_model, decay=float(cfg.get("ema_decay", 0.999))) if cfg.get("use_ema_teacher", False) else None
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
    ib_mbm_train_state = None
    if ib_mbm is not None:
        ib_mbm_train_state = ib_mbm.training
        for p in ib_mbm.parameters():
            ib_mbm_reqgrad_states.append(p.requires_grad)
            p.requires_grad = False
        ib_mbm.eval()

    syn_reqgrad_states = []
    syn_train_state = None
    if synergy_head is not None:
        syn_train_state = synergy_head.training
        for p in synergy_head.parameters():
            syn_reqgrad_states.append(p.requires_grad)
            p.requires_grad = False
        synergy_head.eval()

    # ------------------------------------------------------------
    # 학생 Epoch 결정 로직 – per_stage > schedule > 단일 키/iters 순으로 우선 적용
    # ------------------------------------------------------------
    cur_stage_idx = int(cfg.get("cur_stage", 1)) - 1  # 0-base
    if "student_epochs_per_stage" in cfg and isinstance(cfg["student_epochs_per_stage"], (list, tuple)):
        try:
            student_epochs = int(cfg["student_epochs_per_stage"][cur_stage_idx])
        except (IndexError, ValueError, TypeError):
            raise ValueError(
                "[trainer_student] student_epochs_per_stage가 "
                f"stage {cur_stage_idx+1} 에 대해 정의돼 있지 않습니다."
            )
    elif "student_epochs_schedule" in cfg and isinstance(cfg["student_epochs_schedule"], (list, tuple)):
        try:
            student_epochs = int(cfg["student_epochs_schedule"][cur_stage_idx])
        except (IndexError, ValueError, TypeError):
            raise ValueError(
                "[trainer_student] student_epochs_schedule가 "
                f"stage {cur_stage_idx+1} 에 대해 정의돼 있지 않습니다."
            )
    else:
        student_epochs = int(cfg.get("student_epochs", cfg.get("student_iters", 1)))

    # ──────────────────────────────────────────────────────────
    stage_meter = StageMeter(cfg.get("cur_stage", 1), logger, cfg, student_model)
    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    logger.info(f"[StudentDistill] Using student_epochs={student_epochs}")

    # B-Step 시작: 기존 partial-freeze 상태를 그대로 유지하고 학습 모드만 설정
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
    ema_step = 0
    for ep in range(student_epochs):
        # per-epoch LR logging (optimizer param groups)
        try:
            if optimizer is not None and hasattr(optimizer, "param_groups"):
                for i, pg in enumerate(optimizer.param_groups):
                    lr_val = pg.get("lr", 0.0)
                    logging.info(f"[LR] epoch={global_ep+ep+1} group{i} lr={lr_val:.6f}")
        except Exception:
            pass
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
        else:
            total_epochs = cfg.get("total_epochs", 1)
        # unified tau interpolation (single source)
        if "student_epochs_per_stage" in cfg and isinstance(cfg["student_epochs_per_stage"], (list, tuple)):
            total_ep = int(cfg["student_epochs_per_stage"][max(0, int(cfg.get("cur_stage", 1)) - 1)])
        elif "student_epochs_schedule" in cfg and isinstance(cfg["student_epochs_schedule"], (list, tuple)):
            total_ep = int(cfg["student_epochs_schedule"][max(0, int(cfg.get("cur_stage", 1)) - 1)])
        else:
            total_ep = int(cfg.get("student_epochs", cfg.get("student_iters", 1)))
        cur_tau = _interp_tau(cfg, ep, total_ep)
        try:
            logger.update_metric(f"student_ep{ep+1}_tau", float(cur_tau))
        except Exception:
            pass
        distill_loss_sum = 0.0
        cnt = 0
        # train accuracy meter (mixup/cutmix 시에는 스킵)
        train_correct = 0
        train_total = 0
        # gating/KD/clamp meters (per-epoch)
        gate_on_cnt = 0
        kd_syn_cnt = 0
        batch_cnt = 0
        kd_clamp_cnt = 0
        kd_scale_sum = 0.0
        kd_scale_used = 0
        syn_chosen_samples = 0
        total_kd_samples = 0
        # monitoring counters
        disagree_sum = 0.0
        disagree_cnt = 0
        need_ibm_skip_cnt = 0
        probe_hits = 0
        # kd target usage counters (per-epoch)
        kd_tgt_teacher_cnt = 0
        kd_tgt_avg_cnt = 0
        kd_tgt_syn_cnt = 0
        kd_tgt_wconf_cnt = 0
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
            # reset per-batch locals to avoid stale values across iterations
            kd_vec = None
            ce_vec = None
            kd_tgt = None
            zsyn_ng = None
            mu_ng = None
            logvar_ng = None
            choose_syn = None
            x, y = x.to(cfg["device"], non_blocking=True), y.to(cfg["device"], non_blocking=True)
            # Inputs will be mixed for both student and teacher paths during B-Step
            
            # 채널-라스트 포맷 적용 (Conv 연산 가속)
            if cfg.get("use_channels_last", True) and x.dim() == 4:
                x = x.to(memory_format=torch.channels_last)

            if mix_mode == "cutmix":
                x_mixed, y_a, y_b, lam, idx = cutmix_data(
                    x, y, alpha=cfg["cutmix_alpha_distill"], return_index=True
                )
            elif mix_mode == "mixup":
                x_mixed, y_a, y_b, lam, idx = mixup_data(x, y, alpha=cfg["mixup_alpha"], return_index=True)
            else:
                x_mixed, y_a, y_b, lam = x, y, y, 1.0

            with autocast_ctx:
                # (A) Student forward (query)
                # Student sees mixed inputs for CE
                feat_dict, s_logit, _ = student_model(x_mixed)

                # 교사 모델이 필요한지 확인 (KD, Feature KD, IB 중 하나라도 사용하는 경우)
                need_teachers = (cfg.get("kd_alpha", 0.0) > 0.0) or \
                               (cfg.get("feat_kd_alpha", 0.0) > 0.0) or \
                               bool(cfg.get("use_ib", False))

                # ------------------------------------------------------------------
                # View selection for KD targets and Synergy (default: clean)
                # - kd_target_view: "clean" | "mixed" → teachers for KD targets
                # - synergy_view  : "clean" | "mixed" → student query + teacher KV for IB_MBM
                # ------------------------------------------------------------------
                kd_view = str(cfg.get("kd_target_view", "clean")).lower()
                syn_view = str(cfg.get("synergy_view", "clean")).lower()
                kd_x = x if kd_view == "clean" else x_mixed
                syn_x = x if syn_view == "clean" else x_mixed

                # Student query features for synergy (clean by default)
                feat_kd_key = cfg.get("feat_kd_key", "feat_2d")
                use_grad_for_q = bool(cfg.get("feat_kd_alpha", 0.0) > 0.0)
                if need_teachers:
                    with torch.set_grad_enabled(use_grad_for_q):
                        q_feat_dict, _, _ = student_model(syn_x)
                    s_feat_q = q_feat_dict[feat_kd_key]
                else:
                    s_feat_q = None

                # Teachers forward per view
                if need_teachers:
                    feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) else "feat_2d"
                    t_feats_kd, t_logits_kd = [], []
                    t_feats_syn, t_logits_syn = [], []
                    teachers_for_kd = list(teacher_wrappers)
                    if ema_teacher is not None:
                        teachers_for_kd.append(ema_teacher)
                    teachers_for_syn = list(teacher_wrappers)
                    with torch.no_grad():
                        for tw in teachers_for_kd:
                            out = tw(kd_x)
                            t_dict = out[0] if isinstance(out, tuple) else out
                            t_feats_kd.append(t_dict[feat_key])
                            t_logits_kd.append(t_dict["logit"])
                        for tw in teachers_for_syn:
                            out = tw(syn_x)
                            t_dict = out[0] if isinstance(out, tuple) else out
                            t_feats_syn.append(t_dict[feat_key])
                            t_logits_syn.append(t_dict["logit"])
                else:
                    # 교사 모델 forward를 스킵하는 경우 빈 리스트로 설정
                    t_feats_kd, t_logits_kd = [], []
                    t_feats_syn, t_logits_syn = [], []

                s_feat = feat_dict[feat_kd_key]
                
                # 첫 배치에서 feat_kd_key와 차원 확인
                if ep == 0 and step == 0:
                    logging.info(f"[B-Step] feat_kd_key={feat_kd_key}, s_feat.shape={s_feat.shape}")
                
                # KD target mode selection (supports: teacher, avg, weighted_conf, synergy, auto, auto_min)
                mode = str(cfg.get("kd_target", "teacher")).lower()
                kd_tgt = None
                kd_tgt_mode_this = "none"
                allow_syn = False
                # Pre-compute weighted teacher average for auto mode if needed
                avg_t_simple = None
                t_logits = t_logits_kd
                teacher_w_current = None
                if len(t_logits) >= 1:
                    try:
                        # Base weights (equal or from config)
                        if cfg.get("teacher_weights"):
                            w = torch.tensor(cfg["teacher_weights"], device=t_logits[0].device, dtype=t_logits[0].dtype)
                            w = w[: len(t_logits)]
                        else:
                            w = torch.ones(len(t_logits), device=t_logits[0].device, dtype=t_logits[0].dtype)
                        # Optional teacher2 decay schedule
                        if (len(t_logits) >= 2) and ("teacher2_weight_decay_start_epoch" in cfg):
                            try:
                                t_start = int(cfg.get("teacher2_weight_decay_start_epoch", 0))
                                t_len = int(cfg.get("teacher2_weight_decay_len", 0))
                                if t_len > 0:
                                    prog = min(1.0, max(0.0, float(ep + 1 - t_start) / float(t_len)))
                                    w[1] = w[1] * (1.0 - prog)
                            except Exception:
                                pass
                        # Normalize
                        w = w / (w.sum() + 1e-8)
                        teacher_w_current = w
                        stack = torch.stack(t_logits, dim=0)
                        avg_t_simple = (stack * w.view(-1, 1, 1)).sum(dim=0)
                    except Exception:
                        avg_t_simple = torch.stack(t_logits, dim=0).mean(dim=0)
                if mode == "teacher" and len(t_logits) >= 1:
                    idx = int(cfg.get("kd_teacher_index", 0))
                    idx = max(0, min(idx, len(t_logits) - 1))
                    kd_tgt = t_logits[idx]
                    kd_tgt_mode_this = "teacher"
                elif mode == "avg" and len(t_logits) >= 1:
                    if cfg.get("teacher_weights"):
                        try:
                            w = torch.tensor(cfg["teacher_weights"], device=t_logits[0].device, dtype=t_logits[0].dtype)
                            w = w[: len(t_logits)]
                            # Normalize once, then sum
                            w = w / (w.sum() + 1e-8)
                            stack = torch.stack(t_logits, dim=0)
                            kd_tgt = (stack * w.view(-1, 1, 1)).sum(dim=0)
                        except Exception:
                            kd_tgt = torch.stack(t_logits, dim=0).mean(dim=0)
                    else:
                        kd_tgt = torch.stack(t_logits, dim=0).mean(dim=0)
                    kd_tgt_mode_this = "avg"
                elif mode == "weighted_conf" and len(t_logits) >= 1:
                    # Per-sample confidence-weighted teacher ensemble
                    tau_w = float(cfg.get("tau_gate", 1.0))
                    if len(t_logits) == 1:
                        kd_tgt = t_logits[0]
                        kd_tgt_mode_this = "teacher"
                    elif len(t_logits) == 2:
                        t1_logit, t2_logit = t_logits[0], t_logits[1]
                        q1 = torch.softmax(t1_logit / tau_w, dim=1)
                        q2 = torch.softmax(t2_logit / tau_w, dim=1)
                        p1 = q1.amax(dim=1)
                        p2 = q2.amax(dim=1)
                        denom = (p1 + p2).clamp_min(1e-8)
                        w1 = (p1 / denom).unsqueeze(1)
                        w2 = 1.0 - w1
                        kd_tgt = w1 * t1_logit + w2 * t2_logit
                    else:
                        # General K-teacher case using max-prob as weight
                        probs = []
                        for tlog in t_logits:
                            q = torch.softmax(tlog / tau_w, dim=1)
                            probs.append(q.amax(dim=1))
                        W = torch.stack(probs, dim=0)  # [K, B]
                        W = (W / (W.sum(dim=0, keepdim=True) + 1e-8)).unsqueeze(-1)  # [K,B,1]
                        stack = torch.stack(t_logits, dim=0)  # [K,B,C]
                        kd_tgt = (W * stack).sum(dim=0)
                    kd_tgt_mode_this = "weighted_conf"
                else:
                    # synergy/auto path
                    if mode == "auto_min":
                        # always allow synergy path; per-sample gating via KL-min later
                        allow_syn = True
                    else:
                        allow_syn = _synergy_gate_ok(cfg, logger) if mode in ("synergy", "auto") else False
                    if not allow_syn and len(t_logits) >= 1:
                        # fallback: per-sample confidence-weighted ensemble (better than plain avg in imbalanced teachers)
                        tau_w = float(cfg.get("tau_gate", 1.0))
                        if teacher_w_current is not None and teacher_w_current.numel() >= len(t_logits):
                            # Use dynamic weights if available (broadcast per-batch, per-logit)
                            stack = torch.stack(t_logits, dim=0)
                            kd_tgt = (teacher_w_current.view(-1, 1, 1) * stack).sum(dim=0)
                        else:
                            if len(t_logits) == 2:
                                t1_logit, t2_logit = t_logits[0], t_logits[1]
                                q1 = torch.softmax(t1_logit / tau_w, dim=1)
                                q2 = torch.softmax(t2_logit / tau_w, dim=1)
                                p1 = q1.amax(dim=1)
                                p2 = q2.amax(dim=1)
                                denom = (p1 + p2).clamp_min(1e-8)
                                w1 = (p1 / denom).unsqueeze(1)
                                w2 = 1.0 - w1
                                kd_tgt = w1 * t1_logit + w2 * t2_logit
                            else:
                                probs = []
                                for tlog in t_logits:
                                    q = torch.softmax(tlog / tau_w, dim=1)
                                    probs.append(q.amax(dim=1))
                                W = torch.stack(probs, dim=0)
                                W = (W / (W.sum(dim=0, keepdim=True) + 1e-8)).unsqueeze(-1)
                                stack = torch.stack(t_logits, dim=0)
                                kd_tgt = (W * stack).sum(dim=0)
                        kd_tgt_mode_this = "weighted_conf"

                # Synergy precheck to optionally skip IB_MBM
                if need_teachers:
                    if len(t_logits_kd) >= 2:
                        t1_top = t_logits_kd[0].argmax(1)
                        t2_top = t_logits_kd[1].argmax(1)
                        disagree = (t1_top != t2_top).float().mean().item()
                    else:
                        disagree = 1.0
                    disagree_sum += float(disagree)
                    disagree_cnt += 1
                    syn_win_prev = float(logger.get_metric("syn_win_ratio", 0.0) or 0.0) if logger else 0.0
                    skip_syn = (disagree < float(cfg.get("skip_syn_disagree_thr", 0.15))) and (syn_win_prev < float(cfg.get("skip_syn_win_thr", 0.05)))
                else:
                    skip_syn = False
                # IB/KD 타깃이 필요할 때만 IB_MBM 실행 (synergy 게이트 통과 + multi-teacher + not skipped)
                need_ibm = bool(cfg.get("use_ib", False)) and allow_syn and (len(t_feats_syn) > 1) and (not skip_syn)
                # Light probing: occasionally evaluate synergy even when skip_syn is true to keep syn_win_ratio fresh
                probe_every = int(cfg.get("synergy_probe_every", 0))
                if allow_syn and (not need_ibm) and skip_syn and probe_every > 0:
                    if (step % probe_every) == 0:
                        need_ibm = True
                        probe_hits += 1
                if allow_syn and (not need_ibm):
                    need_ibm_skip_cnt += 1
                if need_ibm:
                    # 스택 전 shape 검증 (차원 불일치 조기 발견)
                    dims = [tf.shape[1] for tf in t_feats_syn]
                    if any(d != dims[0] for d in dims):
                        logging.error(f"[IB-MBM] KV dim mismatch among teachers: dims={dims}. Check distill_out_dim and adapters.")
                        raise RuntimeError("KV dim mismatch")
                    with torch.no_grad():
                        kv = torch.stack(t_feats_syn, dim=1)  # [B, K, D]
                        # Use μ for synergy evaluation/targets by default
                        # Query uses clean view features by default
                        syn_feat_ng, mu_ng, logvar_ng = ib_mbm(s_feat_q, kv, sample=False)
                        use_mu_for_kd = True  # force μ-targets by default
                        zsyn_ng = synergy_head(mu_ng)
                        # Optional synergy logit scaling to counter over-flat logits
                        try:
                            syn_scale = float(cfg.get("synergy_logit_scale", 1.0))
                        except Exception:
                            syn_scale = 1.0
                        if syn_scale != 1.0:
                            zsyn_ng = zsyn_ng * syn_scale
                else:
                    mu_ng = logvar_ng = zsyn_ng = None

                # stable CE/KL calculations
                # CE: mixup/cutmix는 라벨 두 개(y_a,y_b)와 lambda로 직접 혼합
                ls = float(cfg.get("ce_label_smoothing", 0.0))
                if mix_mode in ("mixup", "cutmix"):
                    # compute per-sample CE for mixup/cutmix to allow cw weighting
                    ce_a = F.cross_entropy(s_logit.float(), y_a, reduction="none", label_smoothing=ls)
                    ce_b = F.cross_entropy(s_logit.float(), y_b, reduction="none", label_smoothing=ls)
                    ce_vec = lam * ce_a + (1.0 - lam) * ce_b
                else:
                    ce_vec = ce_safe_vec(s_logit, y, ls_eps=ls)  # [B]
                # KD는 no-grad 타깃 사용 (필요할 때만)
                kd_loss_val = 0.0
                if cfg.get("kd_alpha", 0.0) > 0 and need_teachers:
                    # (optional) teacher centering for stability (center first, then averages)
                    if cfg.get("kd_center_teacher", False) and (t_logits is not None) and len(t_logits) > 0:
                        t_logits = [t - t.mean(dim=1, keepdim=True) for t in t_logits]
                        # also recompute avg_t_simple consistently after centering (with dynamic weights)
                        try:
                            if len(t_logits) >= 1:
                                if cfg.get("teacher_weights"):
                                    w = torch.tensor(cfg["teacher_weights"], device=t_logits[0].device, dtype=t_logits[0].dtype)
                                    w = w[: len(t_logits)]
                                else:
                                    w = torch.ones(len(t_logits), device=t_logits[0].device, dtype=t_logits[0].dtype)
                                if (len(t_logits) >= 2) and ("teacher2_weight_decay_start_epoch" in cfg):
                                    try:
                                        t_start = int(cfg.get("teacher2_weight_decay_start_epoch", 0))
                                        t_len = int(cfg.get("teacher2_weight_decay_len", 0))
                                        if t_len > 0:
                                            prog = min(1.0, max(0.0, float(ep + 1 - t_start) / float(t_len)))
                                            w[1] = w[1] * (1.0 - prog)
                                    except Exception:
                                        pass
                                w = w / (w.sum() + 1e-8)
                                teacher_w_current = w
                                stack = torch.stack(t_logits, dim=0)
                                avg_t_simple = (stack * w.view(-1, 1, 1)).sum(dim=0)
                        except Exception:
                            pass

                    # two_view KD: mixup-aware probability mixing from two clean views
                    if mix_mode in ("mixup", "cutmix") and str(cfg.get("kd_target_mode", "clean")).lower() == "two_view":
                        with torch.no_grad():
                            # ensure KD view is clean
                            base_x = x
                            if len(teacher_wrappers) >= 1:
                                # Reuse already computed clean-view teacher logits for view-A
                                logits_a = []
                                for la in t_logits_kd:
                                    la = la.float()
                                    if cfg.get("kd_center_teacher", False):
                                        la = la - la.mean(dim=1, keepdim=True)
                                    logits_a.append(la)
                                # Compute permuted clean-view teacher logits for view-B only once
                                logits_b = []
                                for tw in teachers_for_kd:
                                    out_b = tw(base_x[idx])
                                    lb = (out_b[0] if isinstance(out_b, tuple) else out_b)["logit"].float()
                                    if cfg.get("kd_center_teacher", False):
                                        lb = lb - lb.mean(dim=1, keepdim=True)
                                    logits_b.append(lb)
                                # Confidence-based weights (softer via tau_gate)
                                tau_w = float(cfg.get("tau_gate", 2.0))
                                Wa = []
                                Wb = []
                                for la in logits_a:
                                    Wa.append(torch.softmax(la / tau_w, dim=1).amax(dim=1))  # [B]
                                for lb in logits_b:
                                    Wb.append(torch.softmax(lb / tau_w, dim=1).amax(dim=1))  # [B]
                                Wa = torch.stack(Wa, dim=0)
                                Wb = torch.stack(Wb, dim=0)
                                Wa = (Wa / (Wa.sum(dim=0, keepdim=True) + 1e-8)).unsqueeze(-1)  # [K,B,1]
                                Wb = (Wb / (Wb.sum(dim=0, keepdim=True) + 1e-8)).unsqueeze(-1)  # [K,B,1]
                                la = (torch.stack(logits_a, dim=0) * Wa).sum(dim=0)  # [B,C]
                                lb = (torch.stack(logits_b, dim=0) * Wb).sum(dim=0)  # [B,C]
                                q1 = torch.softmax(la / cur_tau, dim=1)
                                q2 = torch.softmax(lb / cur_tau, dim=1)
                                q_mix = lam * q1 + (1.0 - lam) * q2
                                from modules.losses import kl_safe_vec_probs as _kl_probs
                                kd_vec = _kl_probs(s_logit, q_mix, tau=cur_tau)
                                kd_tgt_mode_this = "two_view"

                    # KD 타겟 선택: teacher/avg 우선, synergy/auto는 게이트 통과 시 zsyn, 워밍업 혼합 지원
                    if kd_tgt is None and (zsyn_ng is not None) and allow_syn and mode in ("synergy", "auto", "auto_min"):
                        ens = float(cfg.get("kd_ens_alpha", 0.0))
                        warm = int(cfg.get("teacher_adapt_kd_warmup", 0))
                        # 단순 평균을 기본으로 사용
                        avg_t = avg_t_simple
                        if ep < warm and avg_t is not None and ens > 0.0 and mode != "auto":
                            kd_tgt = (1.0 - ens) * zsyn_ng + ens * avg_t
                            kd_tgt_mode_this = "synergy"
                        else:
                            # auto/auto_min 모드: per-sample로 더 나은 타겟 선택 (policy: student_kl | label_ce)
                            if mode in ("auto", "auto_min") and (avg_t is not None):
                                auto_policy = str(cfg.get("kd_auto_policy", "student_kl")).lower()
                                tau_use = float(cur_tau)
                                if auto_policy == "label_ce":
                                    # label-based CE comparison (mixup-aware)
                                    if mix_mode in ("mixup", "cutmix"):
                                        ce_syn_a = F.cross_entropy(zsyn_ng.float(), y_a, reduction="none")
                                        ce_syn_b = F.cross_entropy(zsyn_ng.float(), y_b, reduction="none")
                                        ce_avg_a = F.cross_entropy(avg_t.float(),  y_a, reduction="none")
                                        ce_avg_b = F.cross_entropy(avg_t.float(),  y_b, reduction="none")
                                        ce_syn = lam * ce_syn_a + (1.0 - lam) * ce_syn_b
                                        ce_avg = lam * ce_avg_a + (1.0 - lam) * ce_avg_b
                                    else:
                                        ce_syn = F.cross_entropy(zsyn_ng.float(), y, reduction="none")
                                        ce_avg = F.cross_entropy(avg_t.float(),   y, reduction="none")
                                    choose_syn = (ce_syn < ce_avg)
                                    # KL to chosen target for optimization
                                    kd_vec_syn = kl_safe_vec(s_logit, zsyn_ng.detach(), tau=tau_use)
                                    kd_vec_avg = kl_safe_vec(s_logit,   avg_t.detach(), tau=tau_use)
                                    kd_vec = torch.where(choose_syn, kd_vec_syn, kd_vec_avg)
                                else:
                                    # default: student-KL comparison
                                    kd_vec_syn = kl_safe_vec(s_logit, zsyn_ng.detach(), tau=tau_use)
                                    kd_vec_avg = kl_safe_vec(s_logit,   avg_t.detach(), tau=tau_use)
                                    choose_syn = (kd_vec_syn < kd_vec_avg)
                                    kd_vec = torch.where(choose_syn, kd_vec_syn, kd_vec_avg)

                                kd_tgt_mode_this = mode
                                kd_loss_val = None
                                # Bind DKD target to per-sample selected logits for hybrid DKD
                                if cfg.get("use_dkd_with_synergy", False):
                                    kd_tgt = torch.where(choose_syn.unsqueeze(1), zsyn_ng.detach(), avg_t.detach())
                                with torch.no_grad():
                                    win_ratio = choose_syn.float().mean().item()
                                if logger is not None:
                                    logger.update_metric(f"student_ep{ep+1}_auto_syn_win_ratio", float(win_ratio))
                                    prev = float(logger.get_metric("syn_win_ratio", 0.0) or 0.0)
                                    logger.update_metric("syn_win_ratio", float(0.8 * prev + 0.2 * win_ratio))
                                syn_chosen_samples += int(choose_syn.sum().item())
                                total_kd_samples += int(choose_syn.numel())
                            else:
                                kd_tgt = zsyn_ng
                                kd_tgt_mode_this = "synergy"
                    # Fallback when synergy allowed but IB skipped: ensure KD target present
                    if (kd_tgt is None) and allow_syn and (not need_ibm) and (len(t_logits_kd) >= 1):
                        tau_w = float(cfg.get("tau_gate", 1.0))
                        if teacher_w_current is not None and teacher_w_current.numel() >= len(t_logits_kd):
                            kd_tgt = (teacher_w_current.view(-1, 1, 1) * torch.stack(t_logits_kd, dim=0)).sum(dim=0)
                        else:
                            if len(t_logits_kd) == 1:
                                kd_tgt = t_logits_kd[0]
                            elif len(t_logits_kd) == 2:
                                t1, t2 = t_logits_kd[0], t_logits_kd[1]
                                q1 = torch.softmax(t1 / tau_w, dim=1).amax(dim=1)
                                q2 = torch.softmax(t2 / tau_w, dim=1).amax(dim=1)
                                w1 = (q1 / (q1 + q2 + 1e-8)).unsqueeze(1)
                                w2 = 1.0 - w1
                                kd_tgt = w1 * t1 + w2 * t2
                            else:
                                probs = [torch.softmax(t / tau_w, dim=1).amax(dim=1) for t in t_logits_kd]
                                W = torch.stack(probs, dim=0)
                                W = (W / (W.sum(dim=0, keepdim=True) + 1e-8)).unsqueeze(-1)
                                kd_tgt = (torch.stack(t_logits_kd, dim=0) * W).sum(dim=0)
                        kd_tgt_mode_this = "weighted_conf"

                    # meters update (per-batch)
                    gate_on_cnt += int(allow_syn)
                    kd_syn_cnt  += int(allow_syn)
                    batch_cnt   += 1
                    # kd target usage counters
                    if kd_tgt_mode_this == "teacher":
                        kd_tgt_teacher_cnt += 1
                    elif kd_tgt_mode_this == "avg":
                        kd_tgt_avg_cnt += 1
                    elif kd_tgt_mode_this == "synergy":
                        kd_tgt_syn_cnt += 1
                    elif kd_tgt_mode_this == "weighted_conf":
                        kd_tgt_wconf_cnt += 1

                    if kd_tgt is not None and kd_vec is None:
                        # KL with temperature schedule (synergy 전용 tau_syn 지원)
                        if kd_tgt_mode_this == "synergy":
                            tau_syn = float(cfg.get("tau", cur_tau))
                            kd_vec = kl_safe_vec(s_logit, kd_tgt.detach(), tau=tau_syn)
                        else:
                            kd_vec = kl_safe_vec(s_logit, kd_tgt.detach(), tau=cur_tau)
                        kd_loss_val = None  # set below after cw
                    elif kd_vec is None:
                        kd_loss_val = torch.tensor(0.0, device=s_logit.device)
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
            if cfg.get("use_disagree_weight", False) and need_teachers and len(t_logits) >= 2:
                weights = sample_weights_from_disagreement(
                    t_logits[0],
                    t_logits[1],
                    y,
                    mode=cfg.get("disagree_mode", "pred"),
                    lambda_high=cfg.get("disagree_lambda_high", 1.0),
                    lambda_low=cfg.get("disagree_lambda_low", 1.0),
                )
            else:
                weights = torch.ones_like(y, dtype=s_logit.dtype, device=y.device)

            # AMP / float16 환경에서도 안전
            weights = weights.to(s_logit.dtype)
            # Monitor dynamic teacher weights (if available)
            try:
                if teacher_w_current is not None:
                    logger.update_metric(f"student_ep{ep+1}_teacher_w0", float(teacher_w_current[0].item()))
                    if teacher_w_current.numel() > 1:
                        logger.update_metric(f"student_ep{ep+1}_teacher_w1", float(teacher_w_current[1].item()))
            except Exception:
                pass

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
                # (선택) cw 통계 로깅
                try:
                    logger.update_metric(f"student_ep{ep+1}_cw_mean", float(cw.mean().item()))
                except Exception:
                    pass

            # KD uncertainty weighting (entropy-based)
            if cfg.get("kd_uncertainty_weight", 0.0) > 0 and (kd_vec is not None):
                with torch.no_grad():
                    q_used = None
                    try:
                        if (choose_syn is not None) and (zsyn_ng is not None) and ('avg_t' in locals()) and (avg_t is not None):
                            chosen_logits = torch.where(choose_syn.unsqueeze(1), zsyn_ng, avg_t)
                            q_used = F.softmax(chosen_logits / max(1e-6, float(cur_tau)), dim=1)
                        elif kd_tgt is not None:
                            q_used = F.softmax(kd_tgt.detach() / max(1e-6, float(cur_tau)), dim=1)
                    except Exception:
                        q_used = None
                    if q_used is not None:
                        H = -(q_used * (q_used + 1e-8).log()).sum(dim=1)
                        beta_u = float(cfg.get("kd_uncertainty_weight", 0.5))
                        w_unc = (1.0 / (1.0 + beta_u * H)).clamp_(float(cfg.get("kd_uncertainty_min", 0.2)), 1.0).to(s_logit.dtype)
                        # Label-correctness guard: upweight KD where teacher target matches label with high prob
                        try:
                            kd_correct_min = float(cfg.get("kd_correct_min", 0.0))
                        except Exception:
                            kd_correct_min = 0.0
                        if kd_correct_min > 0.0:
                            try:
                                if mix_mode in ("mixup", "cutmix"):
                                    p_y = lam * q_used.gather(1, y_a.view(-1, 1)).squeeze(1) \
                                        + (1.0 - lam) * q_used.gather(1, y_b.view(-1, 1)).squeeze(1)
                                else:
                                    p_y = q_used.gather(1, y.view(-1, 1)).squeeze(1)
                                w_lab = p_y.clamp_min(kd_correct_min).to(s_logit.dtype)
                                w_unc = w_unc * w_lab
                            except Exception:
                                pass
                        weights = weights * w_unc
                        try:
                            logger.update_metric(f"student_ep{ep+1}_kd_unc_w_mean", float(w_unc.mean().item()))
                        except Exception:
                            pass

            # apply cw to CE and KD (with optional DKD over synergy/avg target)
            if ce_vec is not None:
                ce_loss_val = (weights * ce_vec).mean()
            if kd_vec is not None:
                # Optional DKD hybrid over selected kd_tgt
                if cfg.get("use_dkd_with_synergy", False) and (kd_tgt is not None):
                    kd_loss_val = dkd_loss(
                        s_logit,
                        kd_tgt.detach(),
                        y,
                        alpha=float(cfg.get("dkd_alpha", 1.0)),
                        beta=float(cfg.get("dkd_beta", 2.0)),
                        temperature=float(cur_tau),
                    )
                else:
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
            # Simple epoch-based warmup (0→base over kw epochs) and log effective kd_alpha
            if kw > 0:
                warm = min(1.0, float(ep + 1) / float(kw))
                kd_alpha_eff = kd_alpha_base * warm
            # KD cooldown in the last K epochs: linearly scale kd_alpha_eff → 0
            try:
                kd_cool = int(cfg.get("kd_cooldown_epochs", 0))
                remain = int(total_ep) - (ep + 1)
                if kd_cool > 0 and remain < kd_cool:
                    kd_alpha_eff = kd_alpha_base * float(remain / max(1, kd_cool))
            except Exception:
                pass
            try:
                logger.update_metric(f"student_ep{ep+1}_kd_alpha_eff", float(kd_alpha_eff))
            except Exception:
                pass

            # KD 손실 클램프: 워밍업 이후에만 적용 (과도한 초기 제약 방지)
            kdmr = cfg.get("kd_max_ratio", None)
            kd_clamp_enable = True
            try:
                kw = int(cfg.get("kd_warmup_epochs", 0))
                # Enable clamp only after warmup epochs have finished
                kd_clamp_enable = (ep + 1) > max(0, kw)
            except Exception:
                kd_clamp_enable = True
            if kd_clamp_enable and kdmr is not None and isinstance(kdmr, (int, float)) and kd_loss_val is not None:
                try:
                    with torch.no_grad():
                        ce_s = ce_loss_val.detach().clamp_min(1e-6)
                        kd_s = kd_loss_val.detach().clamp_min(1e-6)
                        scale = torch.clamp((float(kdmr) * ce_s) / kd_s, max=1.0)
                    kd_loss_val = kd_loss_val * scale
                    try:
                        sv = float(scale.item()) if hasattr(scale, "item") else float(scale)
                        if sv < 0.999:
                            kd_clamp_cnt += 1
                            kd_scale_sum += sv
                            kd_scale_used += 1
                    except Exception:
                        pass
                except Exception:
                    pass
            # Optional one-line logging of gate/kd/clamp per epoch handled below

            # 최종 loss는 조건부 합 (상수 0 텐서 더하지 않기)
            loss = _get_cfg_val(cfg, "ce_alpha", 1.0) * ce_loss_val
            if kd_alpha_eff > 0 and (kd_vec is not None):
                loss = loss + kd_alpha_eff * kd_loss_val
            if feat_kd_val is not None:
                loss = loss + cfg.get("feat_kd_alpha", 0.0) * feat_kd_val
            # Optional temperature regularization on synergy head's log_temp
            try:
                temp_reg = float(cfg.get("synergy_temp_reg", 0.0))
            except Exception:
                temp_reg = 0.0
            if temp_reg > 0.0 and (synergy_head is not None) and hasattr(synergy_head, "learnable_temp") and getattr(synergy_head, "learnable_temp", False):
                loss = loss + temp_reg * synergy_head.log_temp.pow(2).mean()
            
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
                ema_step += 1
                if ema_teacher is not None and (ema_step % int(cfg.get("ema_update_every", 1)) == 0):
                    ema_teacher.update(student_model)
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
                ema_step += 1
                if ema_teacher is not None and (ema_step % int(cfg.get("ema_update_every", 1)) == 0):
                    ema_teacher.update(student_model)

            bs = x.size(0)
            distill_loss_sum += loss.item() * bs
            cnt += bs
            # update train accuracy only when labels are not mixed
            if mix_mode == "none":
                with torch.no_grad():
                    pred = s_logit.argmax(dim=1)
                    train_correct += (pred == y).sum().item()
                    train_total += bs
            stage_meter.step(bs)

        ep_loss = distill_loss_sum / cnt
        # CE/KD split logging for easier diagnosis
        try:
            logger.update_metric(f"student_ep{ep+1}_ce", float(ce_loss_val.detach()))
            logger.update_metric(
                f"student_ep{ep+1}_kd",
                float(kd_loss_val.detach()) if isinstance(kd_loss_val, torch.Tensor) else 0.0,
            )
        except Exception:
            pass
        # epoch train acc (optional)
        if train_total > 0:
            try:
                logger.update_metric(f"student_ep{ep+1}_train_acc", 100.0 * train_correct / train_total)
            except Exception:
                pass
        # epoch-level gating/clamp summaries for analysis
        try:
            gate_ratio = gate_on_cnt / max(1, batch_cnt)
            # Use per-sample ratio only; fallback to 0.0 if no KD samples
            kd_syn_ratio = (syn_chosen_samples / max(1, total_kd_samples)) if total_kd_samples > 0 else 0.0
            logger.update_metric(f"student_ep{ep+1}_gate_ratio", float(gate_ratio))
            logger.update_metric(f"student_ep{ep+1}_kd_syn_ratio", float(kd_syn_ratio))
            logger.update_metric(f"student_ep{ep+1}_kd_clamp_ratio", float(kd_clamp_cnt) / max(1, batch_cnt))
            logger.update_metric(
                f"student_ep{ep+1}_kd_scale_mean",
                (kd_scale_sum / max(1, kd_scale_used)) if kd_scale_used > 0 else 1.0,
            )
            # extra monitoring
            if disagree_cnt > 0:
                logger.update_metric(f"student_ep{ep+1}_disagree_mean", float(disagree_sum / max(1, disagree_cnt)))
            if gate_ratio > 0:
                logger.update_metric(
                    f"student_ep{ep+1}_need_ibm_skip_ratio",
                    float(need_ibm_skip_cnt) / max(1, gate_on_cnt)
                )
            logger.update_metric(f"student_ep{ep+1}_probe_hits", int(probe_hits))
            # epoch-level KD target mode used (majority among teacher/avg/synergy)
            if (
                kd_tgt_teacher_cnt >= kd_tgt_wconf_cnt
                and kd_tgt_teacher_cnt >= kd_tgt_syn_cnt
                and kd_tgt_teacher_cnt >= kd_tgt_avg_cnt
                and kd_tgt_teacher_cnt > 0
            ):
                kd_tgt_mode_used = "teacher"
            elif (
                kd_tgt_wconf_cnt >= kd_tgt_syn_cnt
                and kd_tgt_wconf_cnt >= kd_tgt_avg_cnt
                and kd_tgt_wconf_cnt > 0
            ):
                kd_tgt_mode_used = "weighted_conf"
            elif kd_tgt_syn_cnt >= kd_tgt_avg_cnt and kd_tgt_syn_cnt > 0:
                kd_tgt_mode_used = "synergy"
            elif kd_tgt_avg_cnt > 0:
                kd_tgt_mode_used = "avg"
            else:
                kd_tgt_mode_used = str(cfg.get("kd_target", "teacher")).lower()
            logger.update_metric(f"student_ep{ep+1}_kd_tgt_mode", kd_tgt_mode_used)
        except Exception:
            pass

        # (C) validate
        test_acc = eval_student(student_model, testloader, cfg["device"], cfg)

        extra = ""
        if cfg.get("use_loss_clamp", False):
            extra = f" | clamp={(kd_scale_sum / max(1, kd_scale_used)) if kd_scale_used > 0 else 1.0:.2f}"
        kd_clamp_ratio = float(kd_clamp_cnt) / max(1, batch_cnt)
        logging.info(
            "[StudentDistill ep=%d] loss=%.4f testAcc=%.2f best=%.2f | gate=%.2f kdSyn=%.2f kdClamp=%.2f tau=%.2f%s",
            ep + 1,
            ep_loss,
            test_acc,
            best_acc,
            gate_ratio,
            kd_syn_ratio,
            kd_clamp_ratio,
            cur_tau,
            extra,
        )
        # One-line visibility for auto_min per-sample KL gating
        try:
            syn_win = float(logger.get_metric(f"student_ep{ep+1}_auto_syn_win_ratio", 0.0) or 0.0)
            logging.info("[auto_min] ep=%d syn_win=%.2f", ep + 1, syn_win)
        except Exception:
            pass

        # Periodic synergy re-evaluation to refresh gating (every 10 epochs)
        if (ep + 1) % 10 == 0 and ib_mbm is not None and synergy_head is not None:
            try:
                from modules.trainer_teacher import eval_synergy as _eval_syn
                syn_acc = _eval_syn(
                    teacher_wrappers,
                    ib_mbm,
                    synergy_head,
                    testloader,
                    device=cfg["device"], cfg=cfg, student_model=student_model,
                    update_logger=False, logger=None,
                )
                # 모니터링 전용: logger 갱신 금지
            except Exception:
                pass
        if wandb and wandb.run:
            try:
                wandb.log({
                    "student/loss": ep_loss,
                    "student/acc": test_acc,
                    "student/epoch": global_ep + ep + 1,
                    "student/gate_ratio": float(gate_ratio),
                    "student/kd_syn_ratio": float(kd_syn_ratio),
                    "student/kd_clamp_ratio": float(kd_clamp_cnt) / max(1, batch_cnt),
                    "student/kd_scale_mean": (kd_scale_sum / max(1, kd_scale_used)) if kd_scale_used > 0 else 1.0,
                    "student/kd_tgt_mode": kd_tgt_mode_used,
                    "student/kd_alpha_eff": float(logger.get_metric(f"student_ep{ep+1}_kd_alpha_eff", 0.0)) if logger else 0.0,
                })
            except Exception:
                pass

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
    if ib_mbm is not None:
        for p, rg in zip(ib_mbm.parameters(), ib_mbm_reqgrad_states):
            p.requires_grad = rg
        ib_mbm.train(ib_mbm_train_state)
    if synergy_head is not None:
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
