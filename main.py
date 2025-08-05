#!/usr/bin/env python3
"""
main.py

Implements a multi-stage distillation flow using:
 (A) Teacher adaptive update (Teacher/MBM partial freeze)
 (B) Student distillation
Repeated for 'num_stages' times, as in ASIB multi-stage self-training.
"""

import logging
import os
import json
import torch
from typing import Optional, List
from utils.logging import init_logger, ExperimentLogger, get_logger

import hydra
from omegaconf import DictConfig, OmegaConf

# Import core functions
from core import (
    create_student_by_name,
    create_teacher_by_name,
    partial_freeze_teacher_auto,
    partial_freeze_student_auto,
    run_training_stages,
    run_continual_learning,
    _renorm_ce_kd,
    setup_partial_freeze_schedule,
    setup_safety_switches,
    auto_set_mbm_query_dim,
    cast_numeric_configs,
)

from utils.common import set_random_seed, check_label_range, get_model_num_classes, count_trainable_parameters
from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders

# partial freeze
from modules.partial_freeze import (
    apply_partial_freeze,
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_student_resnet,
)

# Teacher creation (factory):
from models.common.base_wrapper import MODEL_REGISTRY
from models.common import registry as _reg  # ensure_scanned()

# --- 중복 방지 플래그
_HP_LOGGED = False
# — W&B --------------------------------------------------------
try:
    import wandb
except ModuleNotFoundError:
    wandb = None


@hydra.main(config_path="configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # ── tqdm 전체 OFF ─────────────────────────────
    if cfg.get("disable_tqdm", False):
        os.environ["PROGRESS"] = 0  # utils.progress 에서 사용

    # ── W&B API-key (config 우선) ────────────────
    if cfg.get("use_wandb", False) and cfg.get("wandb_api_key", ""):
        os.environ.setdefault("WANDB_API_KEY", str(cfg["wandb_api_key"]))

    # ────────────────── LOGGING & W&B ──────────────────
    # results_dir이 제대로 설정되어 있는지 확인
    exp_dir = cfg.get("results_dir", ".")
    if exp_dir == "." and "experiment" in cfg:
        # experiment 섹션에서 results_dir 확인
        exp_dir = cfg["experiment"].get("results_dir", ".")
    
    lvl = cfg.get("log_level") or "WARNING"  # INFO → WARNING으로 변경
    logger = get_logger(
        exp_dir,
        level=lvl,
        stream_level="WARNING" if lvl.upper() == "DEBUG" else lvl,  # INFO → WARNING으로 변경
    )

    global _HP_LOGGED
    if not _HP_LOGGED:
        safe_cfg = {k: v for k, v in cfg.items() if not isinstance(v, logging.Logger)}
        logger.warning("HParams:\n%s", json.dumps(safe_cfg, indent=2))  # info → warning으로 변경
        if wandb is not None and wandb.run:
            wandb.config.update(cfg, allow_val_change=False)
        _HP_LOGGED = True

    if cfg.get("use_wandb", False):
        if wandb is None:
            logging.warning("[W&B] wandb not installed ‑‑ skipping")
        else:
            wandb.init(
                project=cfg.get("wandb_project", "kd_monitor"),
                entity=cfg.get("wandb_entity") or None,
                name=cfg.get("wandb_run_name") or cfg.get("exp_id", "run"),
                config=cfg,
            )
            logging.info("[W&B] dashboard → %s", wandb.run.url)

    # ────────────────── CONFIG PROCESSING ──────────────────
    # Convert learning rate and weight decay fields to float when merged
    cast_numeric_configs(cfg)

    # ────────────────── EXPERIMENT LOGGER ──────────────────
    exp_logger = ExperimentLogger(cfg, exp_name="asib")

    # ────────────────── DEVICE & SEED ──────────────────
    device = cfg.get("device", "cuda")
    if device == "cuda":
        if torch.cuda.is_available():
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
            )
        else:
            logging.warning("No CUDA => Using CPU")
            device = "cpu"

    # fix seed
    seed = cfg.get("seed", 42)
    deterministic = cfg.get("deterministic", True)
    set_random_seed(seed, deterministic=deterministic)

    # ────────────────── DATA LOADING ──────────────────
    dataset = cfg.get("dataset_name", "cifar100")
    batch_size = cfg.get("batch_size", 128)
    data_root = cfg.get("data_root", "./data")
    
    if cfg.get("overlap_pct", -1) >= 0:
        from data.cifar100_overlap import get_overlap_loaders
        (A_tr, A_te), (B_tr, B_te), _ = get_overlap_loaders(
            pct_overlap=cfg["overlap_pct"],
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
            seed=seed,
        )
        # 학생은 **두 교사 클래스의 합집합**을 모두 보게 해야 함
        train_loader = torch.utils.data.ConcatDataset([A_tr.dataset, B_tr.dataset])
        test_loader = torch.utils.data.ConcatDataset([A_te.dataset, B_te.dataset])
        train_loader = torch.utils.data.DataLoader(
            train_loader,
            batch_size=batch_size,
            shuffle=True,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_loader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=cfg.get("num_workers", 2),
            pin_memory=True,
        )
    elif dataset == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
        )
    elif dataset == "imagenet32":
        train_loader, test_loader = get_imagenet32_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset}")

    # ────────────────── MODEL SETUP ──────────────────
    if isinstance(train_loader.dataset, torch.utils.data.ConcatDataset):
        num_classes = 100  # CIFAR-100 fixed
    else:
        n_classes = getattr(train_loader.dataset, "classes", None)
        if n_classes is None:
            n_classes = getattr(train_loader.dataset, "num_classes", None)
        if n_classes is None:
            raise AttributeError("Dataset must expose `classes` or `num_classes`")
        num_classes = len(n_classes) if not isinstance(n_classes, int) else n_classes
    cfg["num_classes"] = num_classes
    exp_logger.update_metric("num_classes", num_classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset == "cifar100"

    # Teacher models
    # Check if config is nested under 'experiment'
    if 'experiment' in cfg:
        cfg = cfg['experiment']
    
    # ────────────────── TEACHER MODELS ──────────────────
    teacher1_name = (
        cfg.get("teacher1", {})
        .get("model", {})
        .get("teacher", {})
        .get("name")
    )
    if not teacher1_name:
        raise ValueError(
            "YAML 에 'teacher1.model.teacher.name' 가 없습니다 (teacher1 모델 미지정)."
        )
    teacher1_ckpt_path = cfg.get(
        "teacher1_ckpt", f"./checkpoints/{teacher1_name}_ft.pth"
    )
    
    # 모델 생성 시 INFO 메시지 억제
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    teacher1 = create_teacher_by_name(
        teacher_name=teacher1_name,
        num_classes=num_classes,
        pretrained=cfg.get("teacher1_pretrained", True),
        small_input=small_input,
        cfg=cfg,
    ).to(device)
    
    # 로깅 레벨 복원
    logging.getLogger().setLevel(original_level)
    
    model_classes = get_model_num_classes(teacher1)
    if model_classes != num_classes:
        raise ValueError(
            f"Teacher1 head expects {model_classes} classes but dataset provides {num_classes}"
        )

    if os.path.exists(teacher1_ckpt_path):
        teacher1.load_state_dict(
            torch.load(teacher1_ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        logging.info("Loaded teacher1 from %s", teacher1_ckpt_path)
        
        # Teacher1 test accuracy
        teacher1.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = teacher1(data)
                # tuple인 경우 처리
                if isinstance(output, tuple):
                    output = output[1]  # logits는 보통 두 번째 요소
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        teacher1_acc = 100.0 * correct / total
        logging.info("Teacher1 (%s) testAcc=%.2f%%", teacher1_name, teacher1_acc)

    teacher2_name = (
        cfg.get("teacher2", {})
        .get("model", {})
        .get("teacher", {})
        .get("name")
    )
    if not teacher2_name:
        raise ValueError(
            "YAML 에 'teacher2.model.teacher.name' 가 없습니다 (teacher2 모델 미지정)."
        )
    teacher2_ckpt_path = cfg.get(
        "teacher2_ckpt", f"./checkpoints/{teacher2_name}_ft.pth"
    )
    
    # 모델 생성 시 INFO 메시지 억제
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    teacher2 = create_teacher_by_name(
        teacher_name=teacher2_name,
        num_classes=num_classes,
        pretrained=cfg.get("teacher2_pretrained", True),
        small_input=small_input,
        cfg=cfg,
    ).to(device)
    
    # 로깅 레벨 복원
    logging.getLogger().setLevel(original_level)
    
    model_classes = get_model_num_classes(teacher2)
    if model_classes != num_classes:
        raise ValueError(
            f"Teacher2 head expects {model_classes} classes but dataset provides {num_classes}"
        )

    if os.path.exists(teacher2_ckpt_path):
        teacher2.load_state_dict(
            torch.load(teacher2_ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        logging.info("Loaded teacher2 from %s", teacher2_ckpt_path)
        
        # Teacher2 test accuracy
        teacher2.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = teacher2(data)
                # tuple인 경우 처리
                if isinstance(output, tuple):
                    output = output[1]  # logits는 보통 두 번째 요소
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        teacher2_acc = 100.0 * correct / total
        logging.info("Teacher2 (%s) testAcc=%.2f%%", teacher2_name, teacher2_acc)

    # Student model
    student_name = (
        cfg.get("model", {})
        .get("student", {})
        .get("model", {})
        .get("student", {})
        .get("name")
    )
    if not student_name:
        raise ValueError(
            "YAML 에 'model.student.name' 가 없습니다 (student 모델 미지정)."
        )
    
    # 모델 생성 시 INFO 메시지 억제
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    
    student_model = create_student_by_name(
        student_name,
        pretrained=cfg.get("student_pretrained", True),
        small_input=small_input,
        num_classes=num_classes,
        cfg=cfg,
    ).to(device)
    
    # 로깅 레벨 복원
    logging.getLogger().setLevel(original_level)

    # ────────────────── MBM & SYNERGY HEAD ──────────────────
    from models.mbm import build_from_teachers
    mbm, synergy_head = build_from_teachers(
        [teacher1, teacher2], cfg
    )
    
    # Move MBM and synergy head to device
    mbm = mbm.to(device)
    synergy_head = synergy_head.to(device)

    # ────────────────── CONFIG SETUP ──────────────────
    num_stages = int(cfg.get("num_stages", 2))
    
    # Setup partial freeze schedule
    setup_partial_freeze_schedule(cfg, num_stages)
    setup_safety_switches(cfg, num_stages)
    
    # Auto-set mbm_query_dim
    auto_set_mbm_query_dim(student_model, cfg)
    
    # Renormalize ce_alpha and kd_alpha
    _renorm_ce_kd(cfg)

    # ────────────────── TRAINING ──────────────────
    if cfg.get("cl_mode", False):
        # Continual Learning mode
        final_acc = run_continual_learning(
            [teacher1, teacher2],
            mbm,
            synergy_head,
            student_model,
            cfg,
            exp_logger,
        )
    else:
        # Standard training mode
        final_acc = run_training_stages(
            [teacher1, teacher2],
            mbm,
            synergy_head,
            student_model,
            train_loader,
            test_loader,
            cfg,
            exp_logger,
            num_stages,
        )

    # ────────────────── FINAL LOGGING ──────────────────
    exp_logger.update_metric("final_student_acc", final_acc)
    exp_logger.save_results()
    
    logging.info(f"Training completed. Final student accuracy: {final_acc:.2f}%")
    
    return final_acc


if __name__ == "__main__":
    main() 