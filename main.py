#!/usr/bin/env python3
"""
main.py

Implements a multi-stage distillation flow using:
 (A) Teacher adaptive update (Teacher/MBM partial freeze)
 (B) Student distillation
Repeated for 'num_stages' times, as in ASMB multi-stage self-training.
"""

import argparse
import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import yaml

from utils.logger import ExperimentLogger
from utils.misc import set_random_seed
from modules.disagreement import compute_disagreement_rate
from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update
from data.cifar100 import get_cifar100_loaders
from data.imagenet100 import get_imagenet100_loaders

# partial freeze
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_teacher_swin,
    partial_freeze_student_resnet,
    partial_freeze_student_efficientnet,
    partial_freeze_student_swin
)

# Teacher creation (factory):
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t

def create_student_by_name(student_name: str, pretrained: bool = True,
                           small_input: bool = False):
    """
    Returns a student model that follows the common interface
    (feature_dict, logits, ce_loss).
    """
    if student_name == "resnet_adapter":
        from models.students.student_resnet_adapter import (
            create_resnet101_with_extended_adapter,
        )
        return create_resnet101_with_extended_adapter(pretrained=pretrained)

    elif student_name == "efficientnet_adapter":
        from models.students.student_efficientnet_adapter import (
            create_efficientnet_b2_with_adapter,
        )
        return create_efficientnet_b2_with_adapter(pretrained=pretrained)

    elif student_name == "swin_adapter":
        from models.students.student_swin_adapter import (
            create_swin_adapter_student,
        )
        return create_swin_adapter_student(pretrained=pretrained,
                                           small_input=small_input)

    else:
        raise ValueError(f"[create_student_by_name] unknown student_name={student_name}")# MBM

from models.mbm import ManifoldBridgingModule, SynergyHead, build_from_teachers

def parse_args():
    parser = argparse.ArgumentParser()

    # (1) YAML 파일
    parser.add_argument("--config", type=str,
                        default="configs/default.yaml",
                        help="Path to yaml config for distillation")

    # (2) sweep 할 때 바꿀 필드들 ▶ YAML 값을 CLI-인자가 **덮어쓰도록** 할 목적
    parser.add_argument("--teacher1_type", type=str)
    parser.add_argument("--teacher2_type", type=str)
    parser.add_argument("--teacher1_ckpt", type=str)
    parser.add_argument("--teacher2_ckpt", type=str)
    parser.add_argument("--num_stages",   type=int)
    parser.add_argument("--synergy_ce_alpha", type=float)    # α
    parser.add_argument("--student_type", type=str)
    
    # 편의용 하이퍼파라미터
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--teacher_lr", type=float)
    parser.add_argument("--student_lr", type=float)
    parser.add_argument("--student_epochs_per_stage", type=int)
    parser.add_argument("--epochs",     type=int)            # 예: teacher_iters
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--seed", type=int, default=42)

    # optional fine-tune params
    parser.add_argument("--finetune_epochs", type=int)
    parser.add_argument("--finetune_lr", type=float)
    parser.add_argument("--finetune_weight_decay", type=float)
    parser.add_argument("--finetune_use_cutmix", type=int)
    parser.add_argument("--finetune_alpha", type=float)
    parser.add_argument("--finetune_ckpt1", type=str)
    parser.add_argument("--finetune_ckpt2", type=str)
    parser.add_argument("--data_aug", type=int, help="1: use augmentation, 0: disable")
    parser.add_argument("--mixup_alpha", type=float)
    parser.add_argument("--cutmix_alpha_distill", type=float)
    parser.add_argument("--label_smoothing", type=float)
    parser.add_argument(
        "--small_input",
        type=int,
        help="1 to use CIFAR-friendly stems (teachers + Swin student)",
    )
    parser.add_argument("--use_feat_kd", type=int)
    parser.add_argument("--feat_kd_alpha", type=float)
    parser.add_argument("--feat_kd_key", type=str)
    parser.add_argument("--feat_kd_norm", type=str)
    parser.add_argument("--use_disagree_weight", type=int)
    parser.add_argument("--disagree_mode", type=str)
    parser.add_argument("--disagree_lambda_high", type=float)
    parser.add_argument("--disagree_lambda_low", type=float)
    parser.add_argument("--teacher1_use_adapter", type=int)
    parser.add_argument("--teacher1_bn_head_only", type=int)
    parser.add_argument("--teacher2_use_adapter", type=int)
    parser.add_argument("--teacher2_bn_head_only", type=int)
    parser.add_argument("--use_amp", type=int)
    parser.add_argument("--amp_dtype", type=str)
    parser.add_argument("--grad_scaler_init_scale", type=int)

    # MBM options
    parser.add_argument("--mbm_type", type=str)
    parser.add_argument("--mbm_r", type=int)
    parser.add_argument("--mbm_n_head", type=int)
    parser.add_argument("--mbm_learnable_q", type=int)
    return parser.parse_args()

def load_config(cfg_path):
    """Load YAML config if file exists"""
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

def create_teacher_by_name(teacher_name, num_classes=100, pretrained=True, small_input=False):
    if teacher_name == "resnet101":
        return create_resnet101(num_classes=num_classes, pretrained=pretrained, small_input=small_input)
    elif teacher_name == "efficientnet_b2":
        return create_efficientnet_b2(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
        )
    elif teacher_name == "swin_tiny":
        return create_swin_t(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"[create_teacher_by_name] Unknown teacher_name={teacher_name}")

def partial_freeze_teacher_auto(
    model,
    teacher_name,
    freeze_bn=True,
    freeze_ln=True,
    use_adapter=False,
    bn_head_only=False,
    freeze_scope=None,
):
    if teacher_name == "resnet101":
        partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_scope=freeze_scope,
        )
    elif teacher_name == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            use_adapter=use_adapter,
            bn_head_only=bn_head_only,
            freeze_scope=freeze_scope,
        )
    elif teacher_name == "swin_tiny":
        partial_freeze_teacher_swin(
            model,
            freeze_ln=freeze_ln,
            use_adapter=use_adapter,
            freeze_scope=freeze_scope,
        )
    else:
        raise ValueError(f"[partial_freeze_teacher_auto] Unknown teacher_name={teacher_name}")

def partial_freeze_student_auto(
    model,
    student_name="resnet_adapter", 
    freeze_bn=True, 
    freeze_ln=True,
    use_adapter=False,
    freeze_scope=None
):
    if student_name == "resnet_adapter":
        partial_freeze_student_resnet(model, freeze_bn=freeze_bn, use_adapter=use_adapter, freeze_scope=freeze_scope)
    elif student_name == "efficientnet_adapter":
        partial_freeze_student_efficientnet(model, freeze_bn=freeze_bn, use_adapter=use_adapter, freeze_scope=freeze_scope)
    elif student_name == "swin_adapter":
        partial_freeze_student_swin(model, freeze_ln=freeze_ln, use_adapter=use_adapter, freeze_scope=freeze_scope)
    else:
        partial_freeze_student_resnet(model, freeze_bn=freeze_bn, freeze_scope=freeze_scope)

def main():
    # 1) parse args
    args = parse_args()

    # 2) load config from YAML
    base_cfg = load_config(args.config)
    cli_cfg = {k: v for k, v in vars(args).items() if v is not None}
    cfg = {**base_cfg, **cli_cfg}

    logger = ExperimentLogger(cfg, exp_name="asmb_experiment")
    logger.update_metric("use_amp", cfg.get("use_amp", False))
    logger.update_metric("amp_dtype", cfg.get("amp_dtype", "float16"))
    logger.update_metric("mbm_type", cfg.get("mbm_type", "MLP"))
    logger.update_metric("mbm_r", cfg.get("mbm_r"))
    logger.update_metric("mbm_n_head", cfg.get("mbm_n_head"))
    logger.update_metric("mbm_learnable_q", cfg.get("mbm_learnable_q"))

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] No CUDA => Using CPU")
        device = "cpu"

    # fix seed
    seed = cfg.get("seed", 42)
    deterministic = cfg.get("deterministic", True)
    set_random_seed(seed, deterministic=deterministic)

    # 3) Data
    dataset = cfg.get("dataset_name", "cifar100")
    batch_size = cfg.get("batch_size", 128)
    data_root = cfg.get("data_root", "./data")
    if dataset == "cifar100":
        train_loader, test_loader = get_cifar100_loaders(
            root=data_root,
            batch_size=batch_size,
            augment=cfg.get("data_aug", True),
        )
    elif dataset == "imagenet100":
        train_loader, test_loader = get_imagenet100_loaders(
            root=data_root,
            batch_size=batch_size,
            augment=cfg.get("data_aug", True),
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset}")

    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset == "cifar100"

    # 4) Create teacher1, teacher2
    teacher1_type = cfg.get("teacher1_type", "resnet101")
    teacher2_type = cfg.get("teacher2_type", "efficientnet_b2")

    teacher1 = create_teacher_by_name(
        teacher_name=teacher1_type,
        num_classes=100,
        pretrained=cfg.get("teacher1_pretrained", True),
        small_input=small_input,
    ).to(device)

    if cfg.get("teacher1_ckpt"):
        teacher1.load_state_dict(
            torch.load(cfg["teacher1_ckpt"], map_location=device, weights_only=True)
        )
        print(f"[Main] Loaded teacher1 from {cfg['teacher1_ckpt']}")

    if cfg.get("use_partial_freeze", True):
        partial_freeze_teacher_auto(
            teacher1, teacher1_type,
            freeze_bn=cfg.get("teacher1_freeze_bn", True),
            freeze_ln=cfg.get("teacher1_freeze_ln", True),
            use_adapter=cfg.get("teacher1_use_adapter", False),
            bn_head_only=cfg.get("teacher1_bn_head_only", False),
            freeze_scope=cfg.get("teacher1_freeze_scope", None)
        )

    teacher2 = create_teacher_by_name(
        teacher_name=teacher2_type,
        num_classes=100,
        pretrained=cfg.get("teacher2_pretrained", True),
        small_input=small_input,
    ).to(device)

    if cfg.get("teacher2_ckpt"):
        teacher2.load_state_dict(
            torch.load(cfg["teacher2_ckpt"], map_location=device, weights_only=True)
        )
        print(f"[Main] Loaded teacher2 from {cfg['teacher2_ckpt']}")

    if cfg.get("use_partial_freeze", True):
        partial_freeze_teacher_auto(
            teacher2, teacher2_type,
            freeze_bn=cfg.get("teacher2_freeze_bn", True),
            freeze_ln=cfg.get("teacher2_freeze_ln", True),
            use_adapter=cfg.get("teacher2_use_adapter", False),
            bn_head_only=cfg.get("teacher2_bn_head_only", False),
            freeze_scope=cfg.get("teacher2_freeze_scope", None)
        )

    # optional fine-tuning of teachers before ASMB stages
    finetune_epochs = int(cfg.get("finetune_epochs", 0))
    if finetune_epochs > 0:
        from modules.cutmix_finetune_teacher import (
            finetune_teacher_cutmix,
            standard_ce_finetune,
        )

        print(f"[Main] Fine-tuning teachers for {finetune_epochs} epochs")
        lr = cfg.get("finetune_lr", 1e-3)
        wd = cfg.get("finetune_weight_decay", 1e-4)
        use_cutmix = bool(cfg.get("finetune_use_cutmix", True))
        alpha = cfg.get("finetune_alpha", 1.0)
        ckpt1 = cfg.get("finetune_ckpt1", "teacher1_finetuned.pth")
        ckpt2 = cfg.get("finetune_ckpt2", "teacher2_finetuned.pth")

        if use_cutmix:
            teacher1, _ = finetune_teacher_cutmix(
                teacher1,
                train_loader,
                test_loader,
                alpha=alpha,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt1,
            )
            teacher2, _ = finetune_teacher_cutmix(
                teacher2,
                train_loader,
                test_loader,
                alpha=alpha,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt2,
            )
        else:
            teacher1, _ = standard_ce_finetune(
                teacher1,
                train_loader,
                test_loader,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt1,
            )
            teacher2, _ = standard_ce_finetune(
                teacher2,
                train_loader,
                test_loader,
                lr=lr,
                weight_decay=wd,
                epochs=finetune_epochs,
                device=device,
                ckpt_path=ckpt2,
            )

    # 5) Student
    student_name  = cfg.get("student_type", "resnet_adapter")   # e.g. resnet_adapter / efficientnet_adapter / swin_adapter
    student_model = create_student_by_name(
        student_name,
        pretrained=cfg.get("student_pretrained", True),
        small_input=small_input,
    ).to(device)

    if cfg.get("student_ckpt"):
        student_model.load_state_dict(
            torch.load(cfg["student_ckpt"], map_location=device, weights_only=True)
        )
        print(f"[Main] Loaded student from {cfg['student_ckpt']}")

    if cfg.get("use_partial_freeze", True):
        partial_freeze_student_auto(
            student_model,
            student_name=student_name,
            freeze_bn=cfg.get("student_freeze_bn", True),
            freeze_ln=cfg.get("student_freeze_ln", True),
            freeze_scope=cfg.get("student_freeze_scope", None),
            use_adapter=cfg.get("student_use_adapter", False)
        )

    # Validate MBM query dimension if specified
    if cfg.get("mbm_query_dim", 0) > 0:
        if hasattr(student_model, "get_feat_dim"):
            s_dim = student_model.get_feat_dim()
            if cfg["mbm_query_dim"] != s_dim:
                raise ValueError(
                    f"mbm_query_dim ({cfg['mbm_query_dim']}) does not match the "
                    f"student feature dimension ({s_dim})."
                )
        else:
            print("[Warning] Student model does not expose get_feat_dim(); unable to validate mbm_query_dim")

    # 6) MBM and synergy head
    mbm_query_dim = cfg.get("mbm_query_dim")
    mbm, synergy_head = build_from_teachers(
        [teacher1, teacher2], cfg, query_dim=mbm_query_dim
    )
    mbm = mbm.to(device)
    synergy_head = synergy_head.to(device)

    # 7) optimizers (collect params once)
    teacher_wrappers = [teacher1, teacher2]

    teacher_params = []
    for tw in teacher_wrappers:
        for p in tw.parameters():
            if p.requires_grad:
                teacher_params.append(p)
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

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
    )
    teacher_scheduler = StepLR(
        teacher_optimizer,
        step_size=cfg.get("teacher_step_size", 10),
        gamma=cfg.get("teacher_gamma", 0.1),
    )

    student_params = [p for p in student_model.parameters() if p.requires_grad]
    student_optimizer = optim.Adam(
        student_params,
        lr=cfg["student_lr"],
        weight_decay=cfg["student_weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    student_scheduler = StepLR(
        student_optimizer,
        step_size=cfg.get("student_step_size", 10),
        gamma=cfg.get("student_gamma", 0.1),
    )

    # 8) multi-stage distillation
    num_stages = cfg.get("num_stages", 2)

    global_ep = 0
    for stage_id in range(1, num_stages + 1):
        print(f"\n=== Stage {stage_id}/{num_stages} ===")

        # restart schedulers each stage
        teacher_scheduler.last_epoch = -1
        student_scheduler.last_epoch = -1

        teacher_epochs = cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5))
        student_epochs = cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15))

        # (A) Teacher adaptive update
        teacher_adaptive_update(
            teacher_wrappers=teacher_wrappers,
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=student_model,
            trainloader=train_loader,
            testloader=test_loader,
            cfg=cfg,
            logger=logger,
            optimizer=teacher_optimizer,
            scheduler=teacher_scheduler,
            global_ep=global_ep,
        )

        global_ep += teacher_epochs

        dis_rate = compute_disagreement_rate(
            teacher1,
            teacher2,
            test_loader,
            device=device,
            cfg=cfg,
        )
        print(f"[Stage {stage_id}] Teacher disagreement= {dis_rate:.2f}%")
        logger.update_metric(f"stage{stage_id}_disagreement_rate", dis_rate)

        # (B) Student distillation
        final_acc = student_distillation_update(
            teacher_wrappers=teacher_wrappers,
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=student_model,
            trainloader=train_loader,
            testloader=test_loader,
            cfg=cfg,
            logger=logger,
            optimizer=student_optimizer,
            scheduler=student_scheduler,
            global_ep=global_ep,
        )
        global_ep += student_epochs
        print(f"[Stage {stage_id}] Student final acc= {final_acc:.2f}%")
        logger.update_metric(f"stage{stage_id}_student_acc", final_acc)

    # 8) save final
    student_ckpt_path = f"{cfg['results_dir']}/final_student_asmb.pth"
    os.makedirs(os.path.dirname(student_ckpt_path), exist_ok=True)
    torch.save(student_model.state_dict(), student_ckpt_path)
    print(f"[main] Distillation done => {student_ckpt_path}")
    logger.update_metric("final_student_ckpt", student_ckpt_path)
    logger.finalize()

if __name__ == "__main__":
    main()
