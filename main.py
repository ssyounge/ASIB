#!/usr/bin/env python3
import logging
import os
import json
from typing import Optional, List

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from utils.logging import ExperimentLogger, get_logger

from core import (
    create_student_by_name,
    create_teacher_by_name,
    run_training_stages,
    run_continual_learning,
    renorm_ce_kd,
    setup_partial_freeze_schedule_with_cfg,
    setup_safety_switches_with_cfg,
    auto_set_ib_mbm_query_dim_with_model,
    cast_numeric_configs,
)

from utils.common import (
    set_random_seed,
    check_label_range,
    get_model_num_classes,
)

from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


def _to_dict(cfg: DictConfig):
    return OmegaConf.to_container(cfg, resolve=True)

def normalize_exp(exp: dict):
    """Flatten any remaining nesting in experiment config and promote method values to top level"""
    if isinstance(exp, dict):
        # dataset/schedule/method가 이중 중첩이면 평탄화
        for k in ("dataset", "schedule", "method"):
            v = exp.get(k)
            if isinstance(v, dict) and k in v and isinstance(v[k], dict):
                exp[k] = v[k]
        
        # teacher1, teacher2, model.student의 중첩도 평탄화
        for k in ("teacher1", "teacher2"):
            v = exp.get(k)
            if isinstance(v, dict) and k in v and isinstance(v[k], dict):
                exp[k] = v[k]
        
        # model.student 중첩 평탄화
        model = exp.get("model", {})
        if isinstance(model, dict):
            student = model.get("student", {})
            if isinstance(student, dict) and "student" in student and isinstance(student["student"], dict):
                model["student"] = student["student"]
        
        # method 값을 최상위로 반영(이름 제외 선택)
        if isinstance(exp.get("method"), dict):
            for mk, mv in exp["method"].items():
                if mk != "name" and mk not in exp:
                    exp[mk] = mv
    
    return exp

@hydra.main(config_path="configs", version_base="1.3")
def main(cfg: DictConfig):
    # 1) experiment 서브트리만 사용
    exp = cfg.experiment if "experiment" in cfg else cfg
    exp_dict = _to_dict(exp)
    
    # 2) 중첩 평탄화
    exp_dict = normalize_exp(exp_dict)

    # 2) 로거
    exp_dir = exp_dict.get("results_dir", ".")
    logger = get_logger(exp_dir, level=exp_dict.get("log_level", "INFO"))
    logger.info("HParams:\n%s", json.dumps({k: v for k, v in exp_dict.items()}, indent=2))

    # 3) W&B (옵션)
    if exp_dict.get("use_wandb", False):
        if wandb is None:
            logger.warning("[W&B] wandb not installed – skipping")
        else:
            wandb.init(
                project=exp_dict.get("wandb_project", "kd_monitor"),
                entity=exp_dict.get("wandb_entity"),
                name=exp_dict.get("wandb_run_name", exp_dict.get("exp_id", "run")),
                config=exp_dict,
            )
            logger.info("[W&B] %s", wandb.run.url)

    # 4) 숫자 캐스팅/안전 스위치
    cast_numeric_configs(exp_dict)

    # 5) 로그 저장기
    exp_logger = ExperimentLogger(exp_dict, exp_name="asib")

    # 6) 디바이스/시드
    device = exp_dict.get("device", "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available. Falling back to CPU.")
        device = "cpu"

    set_random_seed(exp_dict.get("seed", 42), deterministic=exp_dict.get("deterministic", True))

    # 7) 데이터 로더
    ds_cfg = exp_dict.get("dataset", {})
    dataset_name = ds_cfg.get("name", "cifar100")
    data_root = ds_cfg.get("root", "./data")
    batch_size = int(ds_cfg.get("batch_size", 128))
    num_workers = int(ds_cfg.get("num_workers", 2))
    data_aug = ds_cfg.get("data_aug", True)
    small_input = bool(exp_dict.get("small_input", ds_cfg.get("small_input", dataset_name == "cifar100")))

    if exp_dict.get("overlap_pct", -1) >= 0:
        from data.cifar100_overlap import get_overlap_loaders, CIFAR100OverlapDataset
        (A_tr, A_te), (B_tr, B_te), _ = get_overlap_loaders(
            pct_overlap=exp_dict["overlap_pct"],
            batch_size=batch_size,
            num_workers=num_workers,
            augment=data_aug,
            seed=exp_dict.get("seed", 42),
        )
        base_train_loader, base_test_loader = get_cifar100_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug
        )
        all_classes = sorted(list(set(A_tr.dataset.class_indices + B_tr.dataset.class_indices)))
        combined_train_dataset = CIFAR100OverlapDataset(base_train_loader.dataset, all_classes)
        combined_test_dataset = CIFAR100OverlapDataset(base_test_loader.dataset, all_classes)
        train_loader = torch.utils.data.DataLoader(
            combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            combined_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
    elif dataset_name == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug
        )
    elif dataset_name == "imagenet32":
        train_loader, test_loader = get_imagenet32_loaders(
            root=data_root, batch_size=batch_size, num_workers=num_workers, augment=data_aug
        )
    else:
        raise ValueError(f"Unknown dataset.name={dataset_name}")

    # 8) 클래스 수
    if isinstance(train_loader.dataset, torch.utils.data.ConcatDataset):
        num_classes = 100
    else:
        n_classes = getattr(train_loader.dataset, "classes", None)
        if n_classes is None:
            n_classes = getattr(train_loader.dataset, "num_classes", None)
        if n_classes is None:
            raise AttributeError("Dataset must expose `classes` or `num_classes`")
        num_classes = len(n_classes) if not isinstance(n_classes, int) else n_classes

    exp_logger.update_metric("num_classes", num_classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    # 9) 교사/학생 생성
    # teacher1
    t1 = exp_dict.get("teacher1", {})
    t1_name = t1.get("name")
    if not t1_name:
        raise ValueError("Missing 'experiment.teacher1.name'")
    t1_ckpt = exp_dict.get("teacher1_ckpt", f"./checkpoints/teachers/{t1_name}_ft.pth")

    logging.getLogger().setLevel(logging.WARNING)
    teacher1 = create_teacher_by_name(
        teacher_name=t1_name,
        num_classes=num_classes,
        pretrained=t1.get("pretrained", True),
        small_input=small_input,
        cfg=exp_dict,
    ).to(device)
    logging.getLogger().setLevel(logging.INFO)

    if os.path.exists(t1_ckpt):
        try:
            sd = torch.load(t1_ckpt, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(t1_ckpt, map_location=device)
        teacher1.load_state_dict(sd, strict=False)
        logging.info("Loaded teacher1 from %s", t1_ckpt)

    # teacher2
    t2 = exp_dict.get("teacher2", {})
    t2_name = t2.get("name")
    if not t2_name:
        raise ValueError("Missing 'experiment.teacher2.name'")
    t2_ckpt = exp_dict.get("teacher2_ckpt", f"./checkpoints/teachers/{t2_name}_ft.pth")

    logging.getLogger().setLevel(logging.WARNING)
    teacher2 = create_teacher_by_name(
        teacher_name=t2_name,
        num_classes=num_classes,
        pretrained=t2.get("pretrained", True),
        small_input=small_input,
        cfg=exp_dict,
    ).to(device)
    logging.getLogger().setLevel(logging.INFO)

    if os.path.exists(t2_ckpt):
        try:
            sd = torch.load(t2_ckpt, map_location=device, weights_only=True)
        except TypeError:
            sd = torch.load(t2_ckpt, map_location=device)
        teacher2.load_state_dict(sd, strict=False)
        logging.info("Loaded teacher1 from %s", t2_ckpt)

    # 학생
    s = exp_dict.get("model", {}).get("student", {})
    s_name = s.get("name")
    if not s_name:
        raise ValueError("Missing 'experiment.model.student.name'")

    logging.getLogger().setLevel(logging.WARNING)
    student = create_student_by_name(
        s_name,
        pretrained=s.get("pretrained", False),
        small_input=small_input,
        num_classes=num_classes,
        cfg=exp_dict,
    ).to(device)
    logging.getLogger().setLevel(logging.INFO)

    # 10) IB_MBM & Synergy Head
    from models import build_ib_mbm_from_teachers as build_from_teachers
    ib_mbm, synergy_head = build_from_teachers([teacher1, teacher2], exp_dict)
    ib_mbm = ib_mbm.to(device)
    synergy_head = synergy_head.to(device)

    # 11) 학습 전 설정
    num_stages = int(exp_dict.get("num_stages", 1))
    setup_partial_freeze_schedule_with_cfg(exp_dict, num_stages)
    setup_safety_switches_with_cfg(exp_dict, num_stages)
    auto_set_ib_mbm_query_dim_with_model(student, exp_dict)
    renorm_ce_kd(exp_dict)

    # 12) 로그
    logging.info("🚀 Starting training process...")
    logging.info(f"CL mode: {exp_dict.get('cl_mode', False)}")
    logging.info(f"Number of stages: {num_stages}")
    logging.info(f"Student epochs per stage: {exp_dict.get('student_epochs_per_stage', [])}")
    logging.info(f"Batch size: {batch_size}")
    logging.info(f"Device: {device}")
    logging.info(f"Train loader size: {len(train_loader)} batches")
    logging.info(f"Test loader size: {len(test_loader)} batches")
    logging.info(f"Dataset classes: {num_classes}")

    # 13) 트레이닝
    if exp_dict.get("cl_mode", False):
        logging.info("📚 Running in Continual Learning mode...")
        final_acc = run_continual_learning(
            [teacher1, teacher2], ib_mbm, synergy_head, student, exp_dict, exp_logger
        )
    else:
        logging.info("🎯 Running in Standard training mode...")
        try:
            logging.info("📊 Training will start now - you should see epoch progress logs...")
            final_acc = run_training_stages(
                [teacher1, teacher2],
                ib_mbm,
                synergy_head,
                student,
                train_loader,
                test_loader,
                exp_dict,
                exp_logger,
                num_stages,
            )
            logging.info(f"✅ run_training_stages completed with accuracy: {final_acc:.2f}%")
        except Exception as e:
            logging.error(f"❌ run_training_stages failed: {e}", exc_info=True)
            final_acc = 0.0

    logging.info(f"✅ Training completed. Final student accuracy: {final_acc:.2f}%")
    exp_logger.update_metric("final_student_acc", final_acc)
    exp_logger.save_results()
    return final_acc


if __name__ == "__main__":
    main() 