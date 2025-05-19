#!/usr/bin/env python3
"""
main.py

Example flow:
 1) Parse args & load config
 2) Setup logger, device
 3) Create or load Teacher #1, #2, partial freeze them
 4) Optionally measure disagreement rate
 5) Teacher adaptive update (with reg_lambda)
 6) Student distillation
 7) Evaluate final student => log results
"""

import argparse
import torch
import copy

from utils.logger import ExperimentLogger
from modules.partial_freeze import partial_freeze_teacher, partial_freeze_student
from modules.disagreement import compute_disagreement_rate
from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update

# (예시) teacher / student wrapper
from models.teacher_resnet import TeacherResNetWrapper
from models.student_resnet_adapter import StudentResNetAdapter
from data.cifar100 import get_cifar100_loaders

def parse_args():
    parser = argparse.ArgumentParser()
    # ----------------------
    # Teacher & overall
    # ----------------------
    parser.add_argument("--method", type=str, default="asmb")
    parser.add_argument("--teacher1_ckpt", type=str, default="./ckpt/teacher1.pth")
    parser.add_argument("--teacher2_ckpt", type=str, default="./ckpt/teacher2.pth")
    parser.add_argument("--student_ckpt", type=str, default=None)

    # logger.csv에 기록할 'student' 모델명
    parser.add_argument("--student", type=str, default="resnet_adapter",
                        help="Name of the student model (for logging). E.g. 'resnet_adapter'.")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mbm_lr_factor", type=float, default=1.0)
    parser.add_argument("--teacher_weight_decay", type=float, default=1e-4)
    parser.add_argument("--reg_lambda", type=float, default=1e-5, 
                        help="teacher init-based reg")
    parser.add_argument("--mbm_reg_lambda", type=float, default=0.0,
                        help="MBM extra L2")
    parser.add_argument("--synergy_ce_alpha", type=float, default=0.3)
    parser.add_argument("--teacher_adapt_alpha_kd", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--epochs_teacher", type=int, default=5,
                        help="teacher_adaptive_epochs")
    parser.add_argument("--batch_size", type=int, default=128)

    # ----------------------
    # Student Distillation
    # ----------------------
    # e.g. trainer_student.py expects these
    parser.add_argument("--student_lr", type=float, default=1e-2,
                        help="Student distillation learning rate")
    parser.add_argument("--student_weight_decay", type=float, default=5e-4,
                        help="Student distillation weight decay")
    parser.add_argument("--ce_alpha", type=float, default=0.5,
                        help="Weight for CE in student distillation")
    parser.add_argument("--kd_alpha", type=float, default=0.5,
                        help="Weight for KD in student distillation")
    parser.add_argument("--epochs_student", type=int, default=10,
                        help="student distill epochs")

    # ----------------------
    # Device, results, seed
    # ----------------------
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = vars(args)

    # 1) Logger
    logger = ExperimentLogger(cfg)
    device = cfg["device"]

    # 2) Seed, device
    torch.manual_seed(cfg["seed"])
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, fallback to CPU.")
        cfg["device"] = "cpu"

    # 3) Data loaders
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg["batch_size"])

    # 4) Load Teacher models
    teacher1 = TeacherResNetWrapper(pretrained=True)
    teacher2 = TeacherResNetWrapper(pretrained=True)
    if args.teacher1_ckpt:
        teacher1.load_state_dict(torch.load(args.teacher1_ckpt))
    if args.teacher2_ckpt:
        teacher2.load_state_dict(torch.load(args.teacher2_ckpt))
    teacher1.to(device)
    teacher2.to(device)
    partial_freeze_teacher(teacher1)
    partial_freeze_teacher(teacher2)

    # 5) Create Student
    student_model = StudentResNetAdapter(pretrained=True)
    if args.student_ckpt is not None:
        student_model.load_state_dict(torch.load(args.student_ckpt))
    student_model.to(device)
    partial_freeze_student(student_model)

    # 6) MBM, synergy head
    from models.mbm import ManifoldBridgingModule, SynergyHead
    mbm = ManifoldBridgingModule(in_dim=4096, hidden_dim=1024, out_dim=2048).to(device)
    synergy_head = SynergyHead(in_dim=2048, num_classes=100).to(device)

    # 7) measure teacher disagreement
    dis_rate = compute_disagreement_rate(teacher1, teacher2, test_loader, device=device)
    logger.update_metric("disagreement_rate", dis_rate)
    print(f"[main] Teacher1 & Teacher2 cross-error rate= {dis_rate:.2f}%")

    # 8) Teacher adaptive update
    teacher_wrappers = [teacher1, teacher2]
    teacher1_init = copy.deepcopy(teacher1.state_dict())
    teacher2_init = copy.deepcopy(teacher2.state_dict())

    cfg["teacher_adapt_epochs"] = cfg["epochs_teacher"]
    teacher_adaptive_update(
        teacher_wrappers=teacher_wrappers,
        mbm=mbm,
        synergy_head=synergy_head,
        student_model=student_model,
        trainloader=train_loader,
        cfg=cfg,
        logger=logger,
        teacher_init_state=teacher1_init,
        teacher_init_state_2=teacher2_init
    )
    # measure again
    dis_rate_after = compute_disagreement_rate(teacher1, teacher2, test_loader, device=device)
    logger.update_metric("disagreement_after_adapt", dis_rate_after)
    print(f"[main] After teacher update: cross-error rate= {dis_rate_after:.2f}%")

    # 9) Student distillation
    cfg["student_epochs_per_stage"] = cfg["epochs_student"]
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
    logger.update_metric("final_student_acc", final_acc)

    # 10) Save final student
    student_ckpt_path = f"{logger.exp_id}_student.pth"
    torch.save(student_model.state_dict(), student_ckpt_path)
    logger.update_metric("final_student_ckpt", student_ckpt_path)

    # finalize
    logger.finalize()
    print("[main] Distillation done. Results saved.")

if __name__ == "__main__":
    main()
