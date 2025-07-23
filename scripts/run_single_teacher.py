#!/usr/bin/env python3
# scripts/run_single_teacher.py
"""Single-teacher KD runner."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import torch
from utils.misc import set_random_seed, check_label_range, get_model_num_classes
from data.cifar100 import get_cifar100_loaders
from data.imagenet100 import get_imagenet100_loaders
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

def parse_args():
    p = argparse.ArgumentParser(description="Single teacher KD")
    p.add_argument(
        "--config-name",
        type=str,
        default="base",
        help="Hydra config name (from configs/)",
    )
    p.add_argument("--method", type=str, default="vanilla_kd")
    p.add_argument("--teacher_type", type=str)
    p.add_argument("--teacher_ckpt", type=str)
    p.add_argument("--student_type", type=str)
    p.add_argument("--student_ckpt", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--student_lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--epochs", type=int)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--ckpt_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str)
    p.add_argument("--dataset", "--dataset_name", dest="dataset_name", type=str,
                   help="Dataset to use (cifar100 or imagenet100). Defaults to the config value")
    p.add_argument("--data_aug", type=int)
    p.add_argument("--mixup_alpha", type=float)
    p.add_argument("--cutmix_alpha_distill", type=float)
    p.add_argument("--label_smoothing", type=float)
    p.add_argument("--small_input", type=int)
    p.add_argument("--student_freeze_level", type=int)
    return p.parse_args()


def load_config(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


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


def main():
    args = parse_args()
    cfg_path = f"configs/{args.config_name}.yaml" if args.config_name else None
    base_cfg = load_config(cfg_path)
    cli_cfg = {
        k: v
        for k, v in vars(args).items()
        if v is not None and k != "config_name"
    }
    cfg = {**base_cfg, **cli_cfg}

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

    method = cfg.get("method", args.method)
    if method != "asmb":
        cfg["use_partial_freeze"] = False
    teacher_type = cfg.get("teacher_type", cfg.get("default_teacher_type"))
    student_type = cfg.get("student_type", args.student_type)
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
    else:
        train_loader, test_loader = get_imagenet100_loaders(
            root=data_root,
            batch_size=batch_size,
            num_workers=cfg.get("num_workers", 2),
            augment=cfg.get("data_aug", True),
        )

    num_classes = len(train_loader.dataset.classes)
    check_label_range(train_loader.dataset, num_classes)
    check_label_range(test_loader.dataset, num_classes)

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset == "cifar100"

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
