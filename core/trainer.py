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


def create_optimizers_and_schedulers(
    teacher_wrappers: List[torch.nn.Module],
    mbm: torch.nn.Module,
    synergy_head: torch.nn.Module,
    student_model: torch.nn.Module,
    cfg: Dict[str, Any],
    num_stages: int,
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler, 
           optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """Create optimizers and schedulers for training."""
    
    # Teacher parameters
    teacher_params: List[torch.nn.Parameter] = []
    only_da = cfg.get("train_distill_adapter_only", False)
    for tw in teacher_wrappers:
        param_src = (
            tw.distillation_adapter.parameters()
            if only_da and hasattr(tw, "distillation_adapter")
            else tw.parameters()
        )
        for p in param_src:
            if p.requires_grad:
                teacher_params.append(p)
    
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    # Teacher optimizer
    teacher_optimizer = optim.Adam(
        [
            {"params": teacher_params, "lr": cfg["teacher_lr"]},
            {
                "params": mbm_params,
                "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
            },
            {
                "params": syn_params,
                "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
            },
        ],
        weight_decay=cfg["teacher_weight_decay"],
        betas=(
            cfg.get("adam_beta1", 0.9),
            cfg.get("adam_beta2", 0.999),
        ),
    )

    # Teacher scheduler
    teacher_total_epochs = num_stages * cfg.get(
        "teacher_iters", cfg.get("teacher_adapt_epochs", 5)
    )
    if cfg.get("lr_schedule", "step") == "cosine":
        teacher_scheduler = CosineAnnealingLR(
            teacher_optimizer, T_max=teacher_total_epochs
        )
    else:
        teacher_scheduler = StepLR(
            teacher_optimizer,
            step_size=cfg.get("teacher_step_size", 10),
            gamma=cfg.get("teacher_gamma", 0.1),
        )

    # Student optimizer
    student_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=cfg["student_lr"],
        weight_decay=cfg["student_weight_decay"],
        betas=(
            cfg.get("adam_beta1", 0.9),
            cfg.get("adam_beta2", 0.999),
        ),
        eps=1e-8,
    )

    # Student scheduler
    if "student_epochs_schedule" in cfg:
        student_total_epochs = sum(int(e) for e in cfg["student_epochs_schedule"])
    else:
        student_total_epochs = num_stages * cfg.get(
            "student_iters", cfg.get("student_epochs_per_stage", 15)
        )
    if cfg.get("lr_schedule", "step") == "cosine":
        student_scheduler = CosineAnnealingLR(
            student_optimizer, T_max=student_total_epochs
        )
    else:
        student_scheduler = StepLR(
            student_optimizer,
            step_size=cfg.get("student_step_size", 10),
            gamma=cfg.get("student_gamma", 0.1),
        )

    return (
        teacher_optimizer,
        teacher_scheduler,
        student_optimizer,
        student_scheduler,
    )


def run_training_stages(
    teacher_wrappers: List[torch.nn.Module],
    mbm: torch.nn.Module,
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
    
    # Create optimizers and schedulers
    (
        teacher_optimizer,
        teacher_scheduler,
        student_optimizer,
        student_scheduler,
    ) = create_optimizers_and_schedulers(
        teacher_wrappers, mbm, synergy_head, student_model, cfg, num_stages
    )

    # Training loop
    global_ep = 0
    for stage in range(1, num_stages + 1):
        logging.info(f"=== Stage {stage}/{num_stages} ===")
        
        # Teacher adaptive update
        if cfg.get("use_partial_freeze", False):
            te1_acc = teacher_adaptive_update(
                teacher_wrappers,
                mbm,
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
            
            # Log teacher disagreement
            disagree_rate = compute_disagreement_rate(
                teacher_wrappers[0], teacher_wrappers[1], test_loader, cfg["device"]
            )
            logging.info(f"[Stage {stage}] Teacher disagreement= {disagree_rate:.2f}%")
            exp_logger.update_metric(f"stage{stage}_teacher_disagree", disagree_rate)
        
        # Student distillation
        student_acc = student_distillation_update(
            teacher_wrappers,
            mbm,
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
        
        # Update global epoch counter
        teacher_epochs = cfg.get("teacher_adapt_epochs", 1)
        student_epochs = cfg.get("student_epochs_schedule", [15])[stage - 1]
        global_ep += teacher_epochs + student_epochs
        
        # Log stage results
        exp_logger.update_metric(f"stage{stage}_student_acc", student_acc)
        if cfg.get("use_partial_freeze", False):
            exp_logger.update_metric(f"stage{stage}_teacher_acc", te1_acc)
        
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
    mbm: torch.nn.Module,
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