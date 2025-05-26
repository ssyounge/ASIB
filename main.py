#!/usr/bin/env python3
"""
main.py

Implements a multi-stage distillation flow using:
 (A) Teacher adaptive update (Teacher/MBM partial freeze)
 (B) Student distillation
Repeated for 'num_stages' times, as in ASMB multi-stage self-training.

Now we load config (default.yaml), read teacher1_type, teacher2_type, 
create teacher accordingly, and apply partial_freeze dynamically (with freeze_scope).
"""

import argparse
import copy
import torch
import os
import yaml

from utils.logger import ExperimentLogger
from modules.disagreement import compute_disagreement_rate
from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update
from data.cifar100 import get_cifar100_loaders  # or imagenet100

# partial freeze
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
    partial_freeze_teacher_swin,      # if you have a Swin teacher
    partial_freeze_student_resnet,
    partial_freeze_student_efficientnet,
    partial_freeze_student_swin
)

# Teacher creation (factory):
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t  # if you have a Swin teacher

# Student (example)
from models.students.student_resnet_adapter import StudentResNetAdapter

# MBM
from models.mbm import ManifoldBridgingModule, SynergyHead


###############################################################################
# Factory-like helpers
###############################################################################
def create_teacher_by_name(teacher_name, num_classes=100, pretrained=True):
    """
    Creates a teacher model based on `teacher_name` string.
    Extend if you have more models (resnet50, swin_t, etc.)
    """
    if teacher_name == "resnet101":
        return create_resnet101(num_classes=num_classes, pretrained=pretrained)
    elif teacher_name == "efficientnet_b2":
        return create_efficientnet_b2(num_classes=num_classes, pretrained=pretrained)
    elif teacher_name == "swin_tiny":
        return create_swin_t(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"[create_teacher_by_name] Unknown teacher_name={teacher_name}")

def partial_freeze_teacher_auto(
    model, 
    teacher_name, 
    freeze_bn=True, 
    freeze_ln=True, 
    freeze_scope=None
):
    """
    Calls the appropriate partial_freeze_* function based on `teacher_name`.
    Also passes freeze_scope, freeze_bn/ln for more flexible control.
    """
    if teacher_name == "resnet101":
        # partial_freeze_teacher_resnet(model, freeze_bn=?, freeze_scope=?)
        partial_freeze_teacher_resnet(model, freeze_bn=freeze_bn, freeze_scope=freeze_scope)
    elif teacher_name == "efficientnet_b2":
        partial_freeze_teacher_efficientnet(model, freeze_bn=freeze_bn, freeze_scope=freeze_scope)
    elif teacher_name == "swin_tiny":
        # for Swin teacher
        partial_freeze_teacher_swin(model, freeze_ln=freeze_ln, freeze_scope=freeze_scope)
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
    """
    If you have multiple student types (resnet_adapter, effnet_adapter, etc.)
    """
    if student_name == "resnet_adapter":
        partial_freeze_student_resnet(model, freeze_bn=freeze_bn, use_adapter=use_adapter, freeze_scope=freeze_scope)
    elif student_name == "efficientnet_adapter":
        partial_freeze_student_efficientnet(model, freeze_bn=freeze_bn, use_adapter=use_adapter, freeze_scope=freeze_scope)
    elif student_name == "swin_adapter":
        partial_freeze_student_swin(model, freeze_ln=freeze_ln, use_adapter=use_adapter, freeze_scope=freeze_scope)
    else:
        # default or raise error
        partial_freeze_student_resnet(model, freeze_bn=freeze_bn, freeze_scope=freeze_scope)


###############################################################################
# parse_args, load_config
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to yaml config")
    return parser.parse_args()

def load_config(cfg_path):
    """Load YAML config if file exists"""
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


###############################################################################
# main
###############################################################################
def main():
    # 1) parse args
    args = parse_args()

    # 2) load config from YAML
    base_cfg = load_config(args.config)  # e.g. default.yaml
    # merge CLI -> YAML
    cfg = {**base_cfg, **vars(args)}

    # create logger
    logger = ExperimentLogger(cfg, exp_name="asmb_experiment")

    device = cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA not available => Using CPU")
        device = "cpu"

    # fix seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # 3) Data
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg.get("batch_size", 128))

    # 4) Read teacher1_type, teacher2_type from config
    teacher1_type = cfg.get("teacher1_type", "resnet101")
    teacher2_type = cfg.get("teacher2_type", "efficientnet_b2")

    teacher1 = create_teacher_by_name(
        teacher_name=teacher1_type,
        num_classes=100,
        pretrained=cfg.get("teacher1_pretrained", True)
    ).to(device)

    # optional ckpt
    if cfg.get("teacher1_ckpt"):
        teacher1.load_state_dict(torch.load(cfg["teacher1_ckpt"], map_location=device))

    if cfg.get("use_partial_freeze", True):
        # e.g. read from YAML: teacher1_freeze_bn, teacher1_freeze_scope
        t1_freeze_bn = cfg.get("teacher1_freeze_bn", True)
        t1_freeze_scope = cfg.get("teacher1_freeze_scope", None)
        partial_freeze_teacher_auto(
            teacher1, 
            teacher1_type, 
            freeze_bn=t1_freeze_bn,
            freeze_scope=t1_freeze_scope
        )

    teacher2 = create_teacher_by_name(
        teacher_name=teacher2_type,
        num_classes=100,
        pretrained=cfg.get("teacher2_pretrained", True)
    ).to(device)

    if cfg.get("teacher2_ckpt"):
        teacher2.load_state_dict(torch.load(cfg["teacher2_ckpt"], map_location=device))

    if cfg.get("use_partial_freeze", True):
        t2_freeze_bn = cfg.get("teacher2_freeze_bn", True)
        t2_freeze_scope = cfg.get("teacher2_freeze_scope", None)
        partial_freeze_teacher_auto(
            teacher2, 
            teacher2_type, 
            freeze_bn=t2_freeze_bn,
            freeze_scope=t2_freeze_scope
        )

    # 5) Student creation
    student_name = cfg.get("student_type", "resnet_adapter")
    # e.g. if you have create_student_by_name, do it; else just do ResNet
    student_model = StudentResNetAdapter(pretrained=True).to(device)
    if cfg.get("student_ckpt"):
        student_model.load_state_dict(torch.load(cfg["student_ckpt"], map_location=device))

    if cfg.get("use_partial_freeze", True):
        # e.g. student_freeze_bn, student_freeze_scope, etc.
        s_freeze_bn = cfg.get("student_freeze_bn", True)
        s_freeze_scope = cfg.get("student_freeze_scope", None)
        partial_freeze_student_auto(
            student_model, 
            student_name=student_name,
            freeze_bn=s_freeze_bn,
            freeze_scope=s_freeze_scope,
            use_adapter=cfg.get("student_use_adapter", False)
        )

    # 6) MBM => dimension from teacher1.get_feat_dim() + teacher2.get_feat_dim()
    t1_dim = teacher1.get_feat_dim()
    t2_dim = teacher2.get_feat_dim()
    mbm_in_dim = t1_dim + t2_dim

    mbm = ManifoldBridgingModule(
        in_dim=mbm_in_dim,
        hidden_dim=512,   # or from config
        out_dim=512
    ).to(device)

    synergy_head = SynergyHead(in_dim=512, num_classes=100).to(device)

    # 7) multi-stage distillation
    teacher_wrappers = [teacher1, teacher2]
    num_stages = cfg.get("num_stages", 2)

    for stage_id in range(1, num_stages + 1):
        print(f"\n=== Stage {stage_id}/{num_stages} ===")
        
        # backup teacher states
        teacher_init1 = copy.deepcopy(teacher1.state_dict())
        teacher_init2 = copy.deepcopy(teacher2.state_dict())

        # (A) Teacher adaptive
        teacher_adaptive_update(
            teacher_wrappers=teacher_wrappers,
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=student_model,
            trainloader=train_loader,
            testloader=test_loader,
            cfg=cfg,
            logger=logger,
            teacher_init_state=teacher_init1,
            teacher_init_state_2=teacher_init2
        )

        # measure disagreement
        dis_rate = compute_disagreement_rate(teacher1, teacher2, test_loader, device=device)
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
            logger=logger
        )
        print(f"[Stage {stage_id}] Student final acc= {final_acc:.2f}%")
        logger.update_metric(f"stage{stage_id}_student_acc", final_acc)

    # 8) save final
    student_ckpt_path = f"{cfg['results_dir']}/final_student_asmb.pth"
    torch.save(student_model.state_dict(), student_ckpt_path)
    print(f"[main] Distillation done => {student_ckpt_path}")
    logger.update_metric("final_student_ckpt", student_ckpt_path)
    logger.finalize()

if __name__ == "__main__":
    main()
