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

from utils.logger import ExperimentLogger
from modules.partial_freeze import partial_freeze_teacher, partial_freeze_student
from modules.disagreement import compute_disagreement_rate
from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update
from data.cifar100 import get_cifar100_loaders  # or imagenet100

# 실제 Teacher/Student 생성 함수를 import
# (기존에 TeacherResNetWrapper(pretrained=True)로 직접 new하는 대신, 
#  create_resnet101, create_efficientnet_b2 등 factory 함수를 쓰거나,
#  혹은 TeacherResNetWrapper에 get_feat_dim()을 추가해둔 버전을 사용)
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2

# Student
from models.students.student_resnet_adapter import StudentResNetAdapter

# MBM
from models.mbm import ManifoldBridgingModule, SynergyHead

def parse_args():
    parser = argparse.ArgumentParser()
    # Basic config or partial-freeze, kd hyperparams
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    # Multi-stage count
    parser.add_argument("--num_stages", type=int, default=2)

    # Teacher adaptive settings
    parser.add_argument("--teacher_lr", type=float, default=1e-4)
    parser.add_argument("--mbm_lr_factor", type=float, default=5.0)
    parser.add_argument("--teacher_weight_decay", type=float, default=3e-4)
    parser.add_argument("--reg_lambda", type=float, default=1e-5)
    parser.add_argument("--mbm_reg_lambda", type=float, default=1e-4)
    parser.add_argument("--synergy_ce_alpha", type=float, default=0.3)
    parser.add_argument("--teacher_adapt_alpha_kd", type=float, default=0.2)
    parser.add_argument("--teacher_adapt_epochs", type=int, default=5)

    # Student distillation settings
    parser.add_argument("--student_lr", type=float, default=1e-3)
    parser.add_argument("--student_weight_decay", type=float, default=1e-4)
    parser.add_argument("--ce_alpha", type=float, default=0.5)
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--student_epochs_per_stage", type=int, default=10)

    # Teacher or Student model checkpoint
    parser.add_argument("--teacher1_ckpt", type=str, default=None)
    parser.add_argument("--teacher2_ckpt", type=str, default=None)
    parser.add_argument("--student_ckpt", type=str, default=None)

    parser.add_argument("--results_dir", type=str, default="results")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = vars(args)

    logger = ExperimentLogger(cfg, exp_name="asmb_experiment")
    device = cfg["device"]

    # 1) set seed
    torch.manual_seed(cfg["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: no CUDA available, fallback to CPU.")
        cfg["device"] = "cpu"

    # 2) Data
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg["batch_size"])

    # 3) Create Teacher #1, #2
    #    - 예시: Teacher1=ResNet101, Teacher2=EffNet-B2
    teacher1 = create_resnet101(num_classes=100, pretrained=True)
    teacher2 = create_efficientnet_b2(num_classes=100, pretrained=True)

    if cfg["teacher1_ckpt"]:
        teacher1.load_state_dict(torch.load(cfg["teacher1_ckpt"], map_location=device))
    if cfg["teacher2_ckpt"]:
        teacher2.load_state_dict(torch.load(cfg["teacher2_ckpt"], map_location=device))

    teacher1.to(device)
    teacher2.to(device)

    # partial freeze teacher
    partial_freeze_teacher(teacher1)
    partial_freeze_teacher(teacher2)

    # 4) Create Student
    student_model = StudentResNetAdapter(pretrained=True)
    if cfg["student_ckpt"]:
        student_model.load_state_dict(torch.load(cfg["student_ckpt"], map_location=device))
    student_model.to(device)
    partial_freeze_student(student_model)

    # 5) Create MBM + synergy head
    # **수정 포인트**: Teacher1, Teacher2의 feat_dim 합산
    t1_dim = teacher1.get_feat_dim()  # 예: 2048
    t2_dim = teacher2.get_feat_dim()  # 예: 1408
    mbm_in_dim = t1_dim + t2_dim      # 3456

    mbm = ManifoldBridgingModule(
        in_dim=mbm_in_dim,
        hidden_dim=512,
        out_dim=512
    ).to(device)
    synergy_head = SynergyHead(in_dim=512, num_classes=100).to(device)

    # 6) Multi-Stage Distillation
    teacher_wrappers = [teacher1, teacher2]
    for stage_id in range(1, cfg["num_stages"] + 1):
        print(f"\n=== Stage {stage_id}/{cfg['num_stages']} ===")

        # (A) Teacher adaptive update
        teacher_init1 = copy.deepcopy(teacher1.state_dict())
        teacher_init2 = copy.deepcopy(teacher2.state_dict())

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

        # measure teacher disagreement
        dis_rate = compute_disagreement_rate(teacher1, teacher2, test_loader, device=device)
        logger.update_metric(f"stage{stage_id}_disagreement_rate", dis_rate)
        print(f"[Stage {stage_id}] Teacher disagreement rate= {dis_rate:.2f}%")

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

    # 7) Save final Student
    student_ckpt_path = f"{cfg['results_dir']}/final_student_asmb.pth"
    torch.save(student_model.state_dict(), student_ckpt_path)
    print(f"[main] Distillation done. Student ckpt => {student_ckpt_path}")

    logger.update_metric("final_student_ckpt", student_ckpt_path)
    logger.finalize()

if __name__ == "__main__":
    main()
