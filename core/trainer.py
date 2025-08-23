# core/trainer.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
import logging
from typing import List, Optional, Dict, Any, Tuple

from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update
from modules.disagreement import compute_disagreement_rate
from utils.logging import ExperimentLogger
from utils.training.metrics import ExperimentMeter


def _resolve_a_step_alias(cfg):
    """안전하게 a_step_lr을 teacher_lr에 alias."""
    # 안전 캐스팅
    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default
    
    tlr = cfg.get("teacher_lr", 0.0)
    twd = cfg.get("teacher_weight_decay", 0.0)
    
    if _to_float(tlr, 0.0) == 0.0 and "a_step_lr" in cfg:
        cfg["teacher_lr"] = cfg["a_step_lr"]
        logging.info("[Optim] teacher_lr aliased from a_step_lr=%.2e", _to_float(cfg["teacher_lr"]))
    
    if _to_float(twd, 0.0) == 0.0 and "a_step_weight_decay" in cfg:
        cfg["teacher_weight_decay"] = cfg["a_step_weight_decay"]
        logging.info("[Optim] teacher_weight_decay aliased from a_step_weight_decay=%.2e", _to_float(cfg["teacher_weight_decay"]))
    
    return cfg


def create_optimizers_and_schedulers(
    teacher_wrappers: List[torch.nn.Module],
    ib_mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    cfg: Dict[str, Any],
    num_stages: int,
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler, 
           optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizers and schedulers for training."""
    
    # 1. a_step_lr alias 적용 (옵티마이저 만들기 직전)
    cfg = _resolve_a_step_alias(cfg)
    
    # 2. 안전 경고: IB/교사적응인데 lr=0일 때
    eff_tlr = float(cfg.get("teacher_lr", 0.0))
    if (cfg.get("use_ib", False) or cfg.get("teacher_adapt_epochs", 0) > 0) and eff_tlr == 0.0:
        logging.warning("[Optim] use_ib/teacher_adapt on but effective teacher_lr=0. IB/MBM/Head will NOT train.")
    
    # Teacher parameters - 기본적으로 교사 백본은 고정, 어댑터/IB/Head만 학습
    use_tf = bool(cfg.get("use_teacher_finetuning", False))  # 전역 스위치: 교사 미세조정 여부
    only_da = bool(cfg.get("train_distill_adapter_only", not use_tf))  # adapter만 학습할지 여부
    
    teacher_params: List[torch.nn.Parameter] = []
    for tw in teacher_wrappers:
        if use_tf:
            # 교사 전체 미세조정 허용 시에만 전체 파라미터 추가
            teacher_params += [p for p in tw.parameters() if p.requires_grad]
        elif only_da and hasattr(tw, "distillation_adapter"):
            # 교사 distillation adapter만 학습
            teacher_params += [p for p in tw.distillation_adapter.parameters() if p.requires_grad]
        else:
            # 교사 백본은 추가하지 않음 (기본값)
            pass
    
    ib_mbm_params = [
        p for p in (ib_mbm.parameters() if ib_mbm is not None else []) if p.requires_grad
    ]
    syn_params = [
        p for p in (synergy_head.parameters() if synergy_head is not None else []) if p.requires_grad
    ]

    # 3. 파라미터 그룹 확인 로그 (개수 + 총 파라미터 수)
    def _count_trainable(m):
        return sum(p.requires_grad for p in m.parameters())
    
    def _count_trainable_numel(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    
    logging.info("[Trainable] teacher_params=%d | ib_mbm: %d tensors | %d params, head: %d | %d | student: %d tensors | %d params",
                 len(teacher_params),
                 _count_trainable(ib_mbm) if ib_mbm is not None else 0,
                 _count_trainable_numel(ib_mbm) if ib_mbm is not None else 0,
                 _count_trainable(synergy_head) if synergy_head is not None else 0,
                 _count_trainable_numel(synergy_head) if synergy_head is not None else 0,
                 _count_trainable(student_model),
                 _count_trainable_numel(student_model))
    
    # Teacher/IB/Head optimizer (A-step) – use teacher_lr (alias 적용됨)
    # 빈 파라미터 그룹 체크: IB off, teacher_adapt_epochs=0, distill adapter 없으면 None
    if len(teacher_params) + len(ib_mbm_params) + len(syn_params) == 0:
        teacher_optimizer = None
        logging.info("[Optim] No trainable teacher/IB/Head parameters → teacher_optimizer=None")
    else:
        teacher_optimizer = optim.Adam(
            [
                {"params": teacher_params, "lr": cfg["teacher_lr"]},
                {"params": ib_mbm_params, "lr": cfg["teacher_lr"] * cfg.get("ib_mbm_lr_factor", 1.0)},
                {"params": syn_params, "lr": cfg["teacher_lr"] * cfg.get("ib_mbm_lr_factor", 1.0)},
            ],
            weight_decay=cfg["teacher_weight_decay"],
            betas=(
                cfg.get("adam_beta1", 0.9),
                cfg.get("adam_beta2", 0.999),
            ),
        )
        # Sanity check: grad-clip 대상(teacher_params+ib_mbm_params+syn_params)과 옵티마 그룹 일치 여부 로그
        total_group_params = sum(p.numel() for g in teacher_optimizer.param_groups for p in g["params"])
        total_expected = sum(p.numel() for p in teacher_params + ib_mbm_params + syn_params)
        if total_group_params != total_expected:
            logging.warning("[Optim] Param-group vs expected mismatch: groups=%d expected=%d",
                            total_group_params, total_expected)

    # Student optimizer (select by cfg.optimizer)
    if str(cfg.get("optimizer", "adamw")).lower() == "sgd":
        student_optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, student_model.parameters()),
            lr=float(cfg["student_lr"]),
            momentum=cfg.get("b_step_momentum", 0.9),
            nesterov=cfg.get("b_step_nesterov", True),
            weight_decay=float(cfg["student_weight_decay"]),
        )
    else:
        # warn if legacy b_step_lr is present but not used
        if "b_step_lr" in cfg:
            logging.warning("[Optim] AdamW in use; b_step_lr is deprecated. Use student_lr instead.")
        student_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, student_model.parameters()),
            lr=float(cfg["student_lr"]),
            weight_decay=float(cfg["student_weight_decay"]),
            betas=(
                cfg.get("adam_beta1", 0.9),
                cfg.get("adam_beta2", 0.999),
            ),
            eps=1e-8,
        )

    # -----------------------------
    # Schedulers (unified via schedule.* with legacy fallback)
    # -----------------------------
    sch = cfg.get("schedule", {}) or {}
    sched_type = str(sch.get("type", "step")).lower()
    # teacher epochs (per stage)
    teacher_epochs_per_stage = int(cfg.get("teacher_adapt_epochs", 0))
    teacher_total_epochs = int(num_stages * teacher_epochs_per_stage)
    # student epochs (sum over stages)
    if "student_epochs_per_stage" in cfg and isinstance(cfg["student_epochs_per_stage"], (list, tuple)):
        student_total_epochs = int(sum(int(e) for e in cfg["student_epochs_per_stage"]))
    else:
        # legacy fallback
        student_total_epochs = int(num_stages * cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15)))

    # teacher step/gamma (schedule.* first, fallback to legacy per-role keys)
    t_step = sch.get("step_size", cfg.get("teacher_step_size", 10))
    t_gamma = sch.get("gamma", cfg.get("teacher_gamma", 0.1))
    s_step = sch.get("step_size", cfg.get("student_step_size", 10))
    s_gamma = sch.get("gamma", cfg.get("student_gamma", 0.1))

    if sched_type == "none":
        teacher_scheduler = None if teacher_optimizer else None
        student_scheduler = None
    elif sched_type == "cosine":
        teacher_scheduler = CosineAnnealingLR(teacher_optimizer, T_max=max(1, teacher_total_epochs)) if teacher_optimizer else None
        student_scheduler = CosineAnnealingLR(student_optimizer, T_max=max(1, student_total_epochs))
    else:
        teacher_scheduler = StepLR(teacher_optimizer, step_size=int(t_step), gamma=float(t_gamma)) if teacher_optimizer else None
        student_scheduler = StepLR(student_optimizer, step_size=int(s_step), gamma=float(s_gamma))
    
    # teacher_scheduler T_max 안전 처리 확인 로그
    if teacher_optimizer is None:
        logging.info("[Sched] teacher_optimizer=None → teacher_scheduler=None")
    elif teacher_total_epochs == 0:
        logging.info("[Sched] teacher_adapt_epochs=0 → teacher_scheduler T_max=1 (safety)")
    else:
        logging.info("[Sched] teacher_scheduler T_max=%d", max(1, teacher_total_epochs))

    logging.info(
        "[Sched] type=%s | teacher(total=%d, step=%s, gamma=%s) | student(total=%d, step=%s, gamma=%s)",
        sched_type, teacher_total_epochs, t_step, t_gamma, student_total_epochs, s_step, s_gamma
    )

    return (
        teacher_optimizer,
        teacher_scheduler,
        student_optimizer,
        student_scheduler,
    )


def run_training_stages(
    teacher_wrappers: List[torch.nn.Module],
    ib_mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Dict[str, Any],
    exp_logger: ExperimentLogger,
    num_stages: int,
) -> float:
    """Run the main training stages."""
    
    # Create experiment meter for total metrics (전체 실험 시작 시간 기록)
    exp_meter = ExperimentMeter(exp_logger, cfg, student_model)
    
    # 효과적 학습률을 메타에 기록
    eff_tlr = float(cfg.get("teacher_lr", 0.0))
    eff_twd = float(cfg.get("teacher_weight_decay", 0.0))
    exp_logger.update_metric("effective_teacher_lr", eff_tlr)
    exp_logger.update_metric("effective_teacher_wd", eff_twd)
    logging.info("[Meta] effective_teacher_lr=%.2e, effective_teacher_wd=%.2e", eff_tlr, eff_twd)
    
    # Create optimizers and schedulers
    (
        teacher_optimizer,
        teacher_scheduler,
        student_optimizer,
        student_scheduler,
    ) = create_optimizers_and_schedulers(
        teacher_wrappers, ib_mbm, synergy_head, student_model, cfg, num_stages
    )

    # Training loop
    global_ep = 0
    def get_level_for_stage(cfg_local, key: str, stage_idx1: int) -> int:
        sched_key = f"{key}_schedule"
        if sched_key in cfg_local and isinstance(cfg_local[sched_key], (list, tuple)):
            try:
                return int(cfg_local[sched_key][stage_idx1 - 1])
            except Exception:
                return int(cfg_local.get(key, -1))
        return int(cfg_local.get(key, -1))

    for stage in range(1, num_stages + 1):
        cfg["cur_stage"] = stage
        logging.info(f"=== Stage {stage}/{num_stages} ===")
        
        # Stage별 PPF 상태 로깅
        s_freeze = get_level_for_stage(cfg, "student_freeze_level", stage)
        t1_freeze = get_level_for_stage(cfg, "teacher1_freeze_level", stage)
        t2_freeze = get_level_for_stage(cfg, "teacher2_freeze_level", stage)
        s_bn_freeze = cfg.get("student_freeze_bn", False)
        t1_bn_freeze = cfg.get("teacher1_freeze_bn", True)
        t2_bn_freeze = cfg.get("teacher2_freeze_bn", True)
        logging.info(f"[PPF][Stage {stage}] s_freeze={s_freeze} t1_freeze={t1_freeze} t2_freeze={t2_freeze} | BN(s/t1/t2)={s_bn_freeze}/{t1_bn_freeze}/{t2_bn_freeze}")

        if cfg.get("use_partial_freeze", False):
            from modules.partial_freeze import apply_partial_freeze
            # Always allow student PPF when requested
            apply_partial_freeze(student_model, s_freeze, cfg.get("student_freeze_bn", False))
            # Apply teacher PPF only when both teachers exist
            try:
                if len(teacher_wrappers) >= 2:
                    apply_partial_freeze(teacher_wrappers[0], t1_freeze, cfg.get("teacher1_freeze_bn", True))
                    apply_partial_freeze(teacher_wrappers[1], t2_freeze, cfg.get("teacher2_freeze_bn", True))
                else:
                    logging.info("[PPF] Skipping teacher partial_freeze (need 2 teachers, got %d)", len(teacher_wrappers) if teacher_wrappers is not None else 0)
            except Exception:
                logging.debug("[PPF] teacher_wrappers not available for partial_freeze; skipping")
        
        # Teacher adaptive update
        if (
            cfg.get("use_partial_freeze", False)
            or cfg.get("teacher_adapt_epochs", 0) > 0
            or cfg.get("use_cccp", False)
            or cfg.get("use_teacher_finetuning", False)
        ):
            best_syn = teacher_adaptive_update(
                teacher_wrappers,
                ib_mbm,
                synergy_head,
                student_model,
                train_loader,
                test_loader,
                cfg,
                exp_logger,
                teacher_optimizer,
                teacher_scheduler,
                global_ep=global_ep,
            )
            # Record synergy gate metric for B-Step immediately after A-Step
            try:
                if isinstance(best_syn, (int, float)) and float(best_syn) >= 0:
                    exp_logger.update_metric("last_synergy_acc", float(best_syn) / 100.0)
                    exp_logger.update_metric("last_synergy_acc_pct", float(best_syn))
                    cfg["last_synergy_acc"] = float(best_syn) / 100.0
            except Exception:
                pass
            
            # A-Step이 student를 freeze했을 수 있으므로 반드시 복원
            for p in student_model.parameters():
                p.requires_grad = True
            student_model.train()
            
            # A-Step 후 trainable 파라미터 수 기록
            s_train = sum(p.requires_grad for p in student_model.parameters())
            s_total = sum(1 for p in student_model.parameters())
            t1_train = sum(p.requires_grad for p in teacher_wrappers[0].parameters())
            t1_total = sum(1 for p in teacher_wrappers[0].parameters())
            t2_train = sum(p.requires_grad for p in teacher_wrappers[1].parameters())
            t2_total = sum(1 for p in teacher_wrappers[1].parameters())
            
            logging.info(f"[PPF][A-Step] student: {s_train}/{s_total} ({100*s_train/s_total:.1f}%) | t1: {t1_train}/{t1_total} ({100*t1_train/t1_total:.1f}%) | t2: {t2_train}/{t2_total} ({100*t2_train/t2_total:.1f}%)")
            
            # CSV에 기록
            exp_logger.update_metric(f"stage{stage}_student_trainable_params", s_train)
            exp_logger.update_metric(f"stage{stage}_teacher1_trainable_params", t1_train)
            exp_logger.update_metric(f"stage{stage}_teacher2_trainable_params", t2_train)
            
            # IB_MBM/시너지 헤드 trainable 카운트도 기록
            if ib_mbm is not None:
                ib_train = sum(p.requires_grad for p in ib_mbm.parameters())
                ib_total = sum(1 for p in ib_mbm.parameters())
                exp_logger.update_metric(f"stage{stage}_ib_mbm_trainable", ib_train)
                logging.info(f"[PPF][A-Step] IB_MBM: {ib_train}/{ib_total} ({100*ib_train/ib_total:.1f}%)")
            
            if synergy_head is not None:
                syn_train = sum(p.requires_grad for p in synergy_head.parameters())
                syn_total = sum(1 for p in synergy_head.parameters())
                exp_logger.update_metric(f"stage{stage}_synergy_head_trainable", syn_train)
                logging.info(f"[PPF][A-Step] SynergyHead: {syn_train}/{syn_total} ({100*syn_train/syn_total:.1f}%)")
            
            # Log teacher disagreement (with sampling for speed) — only when 2 teachers exist
            try:
                if len(teacher_wrappers) >= 2:
                    disagree_rate = compute_disagreement_rate(
                        teacher_wrappers[0], teacher_wrappers[1], test_loader, cfg["device"],
                        cfg=cfg,
                        max_samples=cfg.get("disagreement_max_samples", None),
                        max_batches=cfg.get("disagreement_max_batches", 10)
                    )
                    logging.info(f"[Stage {stage}] Teacher disagreement= {disagree_rate:.2f}%")
                    exp_logger.update_metric(f"stage{stage}_teacher_disagree", disagree_rate)
                else:
                    logging.info("[Stage %d] Skip disagreement (need 2 teachers, got %d)", stage, len(teacher_wrappers) if teacher_wrappers is not None else 0)
            except Exception:
                logging.debug("[Stage %d] disagreement check skipped (teachers unavailable)", stage)
        
        # Student distillation
        student_acc = student_distillation_update(
            teacher_wrappers,
            ib_mbm,
            synergy_head,
            student_model,
            train_loader,
            test_loader,
            cfg,
            exp_logger,
            student_optimizer,
            student_scheduler,
            global_ep=global_ep,
        )
        
        # Update global epoch counter (A-Step skipped → teacher_epochs defaults to 0)
        teacher_epochs = int(cfg.get("teacher_adapt_epochs", 0))
        # student epochs: per_stage > schedule > single key/iters
        if "student_epochs_per_stage" in cfg and isinstance(cfg["student_epochs_per_stage"], (list, tuple)):
            student_epochs = int(cfg["student_epochs_per_stage"][stage - 1])
        elif "student_epochs_schedule" in cfg and isinstance(cfg["student_epochs_schedule"], (list, tuple)):
            student_epochs = int(cfg["student_epochs_schedule"][stage - 1])
        else:
            student_epochs = int(cfg.get("student_epochs", cfg.get("student_iters", 1)))
        global_ep += teacher_epochs + student_epochs
        
        # Log stage results
        exp_logger.update_metric(f"stage{stage}_student_acc", student_acc)
        if cfg.get("use_partial_freeze", False):
            if 'best_syn' in locals():
                exp_logger.update_metric(f"stage{stage}_teacher_acc", best_syn)
        
        # Get stage metrics from exp_logger (if available)
        # For now, we'll use placeholder values and collect from logs later
        stage_wall_min = exp_logger.get_metric(f"stage{stage}_wall_min", 0.0)
        stage_gpu_h = exp_logger.get_metric(f"stage{stage}_gpu_h", 0.0)
        stage_gflops = exp_logger.get_metric(f"stage{stage}_gflops", 0.0)
        
        # Add stage metrics to experiment meter
        exp_meter.add_stage_metrics(stage_wall_min, stage_gpu_h, stage_gflops, student_acc)
    
    # Log total experiment summary
    exp_meter.finish_experiment()
    
    return student_acc


def run_continual_learning(
    teacher_wrappers: List[torch.nn.Module],
    ib_mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    cfg: Dict[str, Any],
    exp_logger: ExperimentLogger,
) -> float:
    """Run continual learning training."""
    
    logging.info(
        "Continual-Learning mode ON (β=%.3f)", cfg.get("ib_beta", 0.01)
    )

    from utils.data import get_split_cifar100_loaders
    from utils.cl_utils import ReplayBuffer, EWC
    from modules.trainer_student import eval_student

    num_tasks = cfg.get("num_tasks", 5)
    replay_ratio = cfg.get("replay_ratio", 0.5)
    replay_cap = cfg.get("replay_capacity", 2000)

    task_loaders = get_split_cifar100_loaders(
        num_tasks=num_tasks,
        batch_size=cfg.get("batch_size", 128),
        augment=cfg.get("data_aug", True),
        root=cfg.get("data_root", "./data"),
    )

    buffer = ReplayBuffer(replay_cap)
    ewc = EWC(cfg.get("lambda_ewc", 0.4))
    device = cfg["device"]

    global_ep = 0
    for task_id in range(num_tasks):
        logging.info("=== Task %d/%d ===", task_id + 1, num_tasks)
        student_model.train()
        epochs = cfg.get("epochs", 1)
        for ep in range(epochs):
            for x, y in tl:
                x, y = x.to(device), y.to(device)
                r_bs = int(replay_ratio * x.size(0))
                if len(buffer.buffer) == 0:
                    x_batch, y_batch = x, y
                else:
                    r_x, r_y = buffer.sample(r_bs, device)
                    x_batch = torch.cat([x, r_x], dim=0)
                    y_batch = torch.cat([y, r_y], dim=0)
                
                # Training step (simplified)
                # ... (continual learning training logic)
                pass
        
        # Update buffer
        buffer.update(tl.dataset)
        
        # Evaluate
        acc = eval_student(student_model, vl, device, cfg)
        exp_logger.update_metric(f"task{task_id+1}_acc", acc)
    
    return acc

def create_optimizers_and_schedulers_legacy(
    teacher: torch.nn.Module,
    student_model: torch.nn.Module,
    cfg: Dict[str, Any],
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler, 
           optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizers and schedulers for training (legacy version)."""
    
    # Teacher optimizer
    teacher_optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, teacher.parameters()),
        lr=cfg.get("teacher_lr", 0.001),
        weight_decay=cfg.get("teacher_weight_decay", 0.0001),
    )

    # Teacher scheduler
    teacher_scheduler = CosineAnnealingLR(
        teacher_optimizer, T_max=cfg.get("teacher_epochs", 10)
    )

    # Student optimizer
    student_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=cfg.get("student_lr", 0.001),
        weight_decay=cfg.get("student_weight_decay", 0.0001),
    )

    # Student scheduler
    student_scheduler = CosineAnnealingLR(
        student_optimizer, T_max=cfg.get("student_epochs", 10)
    )

    return teacher_optimizer, teacher_scheduler, student_optimizer, student_scheduler 