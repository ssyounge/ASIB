#!/usr/bin/env python3
# scripts/run_single_teacher.py
"""Single-teacher KD runner."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.misc import set_random_seed, check_label_range, get_model_num_classes
from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders
from main import (
    create_teacher_by_name,
    create_student_by_name,
    partial_freeze_teacher_auto,
    apply_partial_freeze,
)

from methods.vanilla_kd import VanillaKDDistiller
from methods.fitnet import FitNetDistiller
from methods.dkd import DKDDistiller
from methods.at import ATDistiller
from methods.crd import CRDDistiller


METHOD_MAP = {
    "vanilla_kd": VanillaKDDistiller,
    "fitnet": FitNetDistiller,
    "dkd": DKDDistiller,
    "at": ATDistiller,
    "crd": CRDDistiller,
}



def build_distiller(method, teacher, student, cfg):
    cls = METHOD_MAP.get(method)
    if cls is None:
        raise ValueError(f"Unknown method {method}")
    if method == "vanilla_kd":
        return cls(teacher, student, alpha=cfg.get("kd_alpha", 0.5), temperature=cfg.get("tau_start", 4.0), config=cfg)
    if method == "fitnet":
        s_channels = student.get_feat_dim()
        t_channels = teacher.get_feat_dim()
        return cls(
            teacher,
            student,
            s_channels=s_channels,
            t_channels=t_channels,
            config=cfg,
        )
    if method == "dkd":
        return cls(teacher, student, config=cfg)
    if method == "at":
        return cls(teacher, student, config=cfg)
    if method == "crd":
        return cls(teacher, student, config=cfg)
    raise ValueError(method)


@hydra.main(config_path="../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    from utils.config_utils import flatten_hydra_config
    cfg = flatten_hydra_config(cfg)

    # ──────────────────────────────────────────────────────────────
    # YAML/CLI override 로 인해 숫자가 문자열로 들어올 수 있다.
    # Optimizer 쪽에서 TypeError 가 나지 않도록 미리 float 캐스팅.
    for key in (
        "teacher_lr", "student_lr",
        "teacher_weight_decay", "student_weight_decay",
        "ce_alpha", "kd_alpha",
        "reg_lambda", "mbm_reg_lambda",
    ):
        if key in cfg and isinstance(cfg[key], str):
            try:
                cfg[key] = float(cfg[key])
            except ValueError:
                # 빈 문자열 등은 그대로 둠
                pass
    # ──────────────────────────────────────────────────────────────

    method = cfg.get("method", "vanilla_kd")
    if method != "asmb":
        cfg["use_partial_freeze"] = False
    teacher_type = cfg.get("teacher_type", cfg.get("default_teacher_type"))
    student_type = cfg.get("student_type", "resnet_adapter")
    print(
        f">>> [run_single_teacher.py] method={method} teacher={teacher_type} student={student_type}"
    )
    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    set_random_seed(cfg.get("seed", 42))

    dataset = cfg.get("dataset_name", "cifar100")
    batch_size = cfg.get("batch_size", 128)
    data_root = cfg.get("data_root", "./data")
    if dataset == "cifar100":
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
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset}")

    num_classes = len(train_loader.dataset.classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset in ("cifar100", "imagenet32")

    teacher_type = cfg.get("teacher_type", cfg.get("default_teacher_type"))
    teacher_ckpt_path = cfg.get("teacher_ckpt", f"./checkpoints/{teacher_type}_ft.pth")
    teacher = create_teacher_by_name(
        teacher_type,
        pretrained=cfg.get("teacher_pretrained", True),
        small_input=small_input,
        num_classes=num_classes,
        cfg=cfg,
    ).to(device)
    model_classes = get_model_num_classes(teacher)
    if model_classes != num_classes:
        raise ValueError(
            f"Teacher head expects {model_classes} classes but dataset provides {num_classes}"
        )
    if os.path.exists(teacher_ckpt_path):
        teacher.load_state_dict(
            torch.load(teacher_ckpt_path, map_location=device, weights_only=True),
            strict=False,
        )
        print(f"[run_single_teacher.py] Loaded teacher from {teacher_ckpt_path}")
    if cfg.get("use_partial_freeze", True):
        partial_freeze_teacher_auto(
            teacher,
            cfg.get("teacher_type", cfg.get("default_teacher_type")),
            freeze_bn=cfg.get("teacher_freeze_bn", True),
            freeze_ln=cfg.get("teacher_freeze_ln", True),
            use_adapter=cfg.get("teacher_use_adapter", False),
            bn_head_only=cfg.get("teacher_bn_head_only", False),
            freeze_level=cfg.get("teacher_freeze_level", 1),
        )

    student = create_student_by_name(
        cfg.get("student_type", "resnet_adapter"),
        pretrained=cfg.get("student_pretrained", True),
        small_input=small_input,
        num_classes=num_classes,
        cfg=cfg,
    ).to(device)
    if cfg.get("student_ckpt"):
        student.load_state_dict(
            torch.load(cfg["student_ckpt"], map_location=device, weights_only=True),
            strict=False,
        )
    apply_partial_freeze(
        student,
        cfg.get("student_freeze_level", -1 if not cfg.get("use_partial_freeze") else 0),
        cfg.get("student_freeze_bn", False),
    )

    distiller = build_distiller(method, teacher, student, cfg)
    acc = distiller.train_distillation(
        train_loader,
        test_loader,
        lr=cfg.get("student_lr", 1e-3),
        weight_decay=cfg.get("weight_decay", cfg.get("student_weight_decay", 1e-4)),
        epochs=cfg.get("epochs", 10),
        device=device,
        cfg=cfg,
    )

    ckpt_dir = cfg.get("ckpt_dir", cfg.get("results_dir", "results"))
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"final_student_{method}.pth")
    torch.save(student.state_dict(), ckpt)
    print(f"[run_single_teacher] final_acc={acc:.2f}% -> {ckpt}")


if __name__ == "__main__":
    main()
