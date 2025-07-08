"""Minimal continual-learning loop that re-uses KD modules."""

import os
import torch

from data.cifar100_cl import get_cifar100_cl_loaders
from trainer import teacher_vib_update, student_vib_update
from models.teachers.teacher_resnet import create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from utils.freeze import freeze_all
from utils.eval import evaluate_acc


def run_continual(cfg: dict, kd_method: str, logger=None) -> None:
    """Run continual-learning training using KD modules."""
    device = cfg.get("device", "cuda")
    n_tasks = cfg.get("n_tasks", 10)
    if 100 % n_tasks != 0:
        raise ValueError(
            f"n_tasks={n_tasks} 가 CIFAR-100 을 균등 분할하지 않습니다."
        )
    ckpt_dir = os.path.join(cfg.get("results_dir", "results"), "cl_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    t1 = create_resnet152(pretrained=True, small_input=True).to(device)
    t2 = create_efficientnet_b2(pretrained=True, small_input=True).to(device)
    freeze_all(t1)
    freeze_all(t2)
    t1.eval()
    t2.eval()

    if kd_method == "vib":
        from models.ib.gate_mbm import GateMBM

        num_cls_total = (100 // cfg.get("n_tasks", 10)) * cfg.get("n_tasks", 10)
        vib_mbm = GateMBM(
            t1.get_feat_dim(),                # teacher-1 feat dim
            t2.get_feat_dim(),                # teacher-2 feat dim
            cfg["z_dim"],                     # latent z
            num_cls_total,                    # 100-way (확정)
            beta=cfg.get("beta_bottleneck", 1e-3),
        ).to(device)
    else:
        vib_mbm = None

    for task in range(n_tasks):
        train_loader, test_cur, test_seen = get_cifar100_cl_loaders(
            root=cfg.get("dataset_root", "./data"),
            task_id=task,
            n_tasks=n_tasks,
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 2),
            randaug_N=cfg.get("randaug_N", 0),
            randaug_M=cfg.get("randaug_M", 0),
            persistent_train=cfg.get("persistent_workers", False),
        )

        if task == 0 and vib_mbm is not None:
            opt_t = torch.optim.Adam(
                vib_mbm.parameters(),
                lr=cfg.get("teacher_lr", 1e-3),
                weight_decay=cfg.get("teacher_weight_decay", 0.0),
            )
            teacher_vib_update(
                t1,
                t2,
                vib_mbm,
                train_loader,
                cfg,
                opt_t,
                test_loader=test_cur,
                logger=logger,
            )

        from utils.model_factory import create_student_by_name

        new_num_cls = (task + 1) * (100 // n_tasks)
        student = create_student_by_name(
            cfg.get("student_type", "convnext_tiny"),
            num_classes=new_num_cls,
            pretrained=True,
            small_input=True,
            cfg=cfg,
        ).to(device)

        # ─ 이어 학습용 가중치 로드 ─
        prev_ckpt = f"{ckpt_dir}/task{task-1}_student.pth"
        if task > 0 and os.path.isfile(prev_ckpt):
            miss, _ = student.load_state_dict(
                torch.load(prev_ckpt, map_location="cpu"), strict=False
            )

        if kd_method == "vib":
            from models.ib.proj_head import StudentProj

            proj = StudentProj(
                student.get_feat_dim(),
                cfg["z_dim"],
                hidden_dim=cfg.get("proj_hidden_dim"),
                normalize=True,
            ).to(device)  # ← device 맞춤
            opt_s = torch.optim.AdamW(
                list(student.parameters()) + list(proj.parameters()),
                lr=cfg.get("student_lr", 5e-4),
                weight_decay=cfg.get("student_weight_decay", 5e-4),
            )
            student_vib_update(
                t1,
                t2,
                student,
                vib_mbm,
                proj,
                train_loader,
                cfg,
                opt_s,
                test_loader=test_cur,
                logger=logger,
            )
        else:
            if kd_method == "dkd":
                from methods.dkd import DKDDistiller as Distiller

                distiller = Distiller(
                    teacher_model=t1,
                    student_model=student,
                    alpha=cfg.get("dkd_alpha", 1.0),
                    beta=cfg.get("dkd_beta", 8.0),
                    temperature=cfg.get("dkd_T", 4.0),
                    warmup=cfg.get("dkd_warmup", 5),
                    label_smoothing=cfg.get("label_smoothing", 0.0),
                    config=cfg,
                )
            elif kd_method == "crd":
                from methods.crd import CRDDistiller as Distiller

                distiller = Distiller(
                    teacher_model=t1,
                    student_model=student,
                    alpha=cfg.get("crd_alpha", 0.5),
                    temperature=cfg.get("crd_T", 0.07),
                    label_smoothing=cfg.get("label_smoothing", 0.0),
                    config=cfg,
                )
            else:
                from methods.vanilla_kd import VanillaKDDistiller as Distiller

                distiller = Distiller(
                    teacher_model=t1,
                    student_model=student,
                    alpha=cfg.get("vanilla_alpha", 0.5),
                    temperature=cfg.get("vanilla_T", 4.0),
                    config=cfg,
                )

            distiller.train_distillation(
                train_loader,
                test_cur,
                epochs=cfg.get("student_iters", 60),
                lr=cfg.get("student_lr", 5e-4),
                weight_decay=cfg.get("student_weight_decay", 5e-4),
                device=device,
                cfg=cfg,
            )

        torch.save(student.state_dict(), f"{ckpt_dir}/task{task}_student.pth")

        # ① 현재 task-only
        acc_cur  = evaluate_acc(student, test_cur,  device=device)
        # ② 지금까지 전체 class
        acc_seen = evaluate_acc(student, test_seen, device=device)

        if logger is not None:
            logger.info(
                f"[CIL] task {task} → cur={acc_cur:.2f}%  seen={acc_seen:.2f}%"
            )
            logger.update_metric(f"task{task}_acc_cur",  float(acc_cur))
            logger.update_metric(f"task{task}_acc_seen", float(acc_seen))
        else:
            print(
                f"[CIL] task {task} → cur={acc_cur:.2f}%  seen={acc_seen:.2f}%"
            )

    if logger is not None:
        logger.finalize()

