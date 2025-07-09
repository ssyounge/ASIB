"""Minimal continual-learning loop that re-uses KD modules."""
# trainer_continual.py

import os
import torch

from data.cifar100_cl import get_cifar100_cl_loaders
from trainer import teacher_vib_update, student_vib_update, simple_finetune
from models.teachers.teacher_resnet import create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from utils.freeze import freeze_all
from utils.eval import evaluate_acc


def _remap_for_task(logits: torch.Tensor,
                    target: torch.Tensor,
                    classes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice logits/labels to the current task classes."""
    device = logits.device
    cls_tensor = torch.tensor(classes, dtype=torch.long, device=device)
    logits_t = logits.index_select(dim=1, index=cls_tensor)
    global2local = {c: i for i, c in enumerate(classes)}
    tgt_local = torch.empty_like(target)
    for g, l in global2local.items():
        tgt_local[target == g] = l
    return logits_t, tgt_local


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

    acc_seen_hist = []

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
        # ───────── Task 헤더 ─────────
        head = f"[CIL] ── Task {task+1}/{n_tasks} " + "─"*40
        print("\n" + head)
        if logger is not None:
            logger.info(head)
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

        train_set = train_loader.dataset
        if isinstance(train_set, torch.utils.data.Subset):
            indices = getattr(train_set, 'indices', [])
            base = getattr(train_set, 'dataset', train_set)
            targets = [base.targets[i] for i in indices]
        else:
            targets = getattr(train_set, 'targets', [])
        task_classes = sorted({int(t) for t in targets})
        cfg['task_meta'] = {'classes': task_classes, 'id': task}

        if task == 0 and vib_mbm is not None:
            opt_t = torch.optim.Adam(
                vib_mbm.parameters(),
                lr=float(cfg.get("teacher_lr", 1e-3)),
                weight_decay=float(cfg.get("teacher_weight_decay", 0.0)),
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

        # ─── 이어 학습: 이전 task 가중치 로드 ───
        prev_ckpt = f"{ckpt_dir}/task{task-1}_student.pth"
        prev_student = None
        if task > 0 and os.path.isfile(prev_ckpt):
            prev = torch.load(prev_ckpt, map_location="cpu")
            cur = student.state_dict()

            # ① 새 모델과 **shape** 이 같은 파라미터만 선택
            ok = {k: v for k, v in prev.items() if k in cur and v.shape == cur[k].shape}

            # ② 로드 (classifier 가중치는 자동 스킵)
            student.load_state_dict(ok, strict=False)

            from utils.model_factory import create_student_by_name
            prev_student = create_student_by_name(
                cfg.get("student_type", "convnext_tiny"),
                num_classes=new_num_cls - (100 // n_tasks),
                pretrained=False,
                small_input=True,
                cfg=cfg,
            ).to(device)
            prev_student.load_state_dict(torch.load(prev_ckpt, map_location="cpu"))
            prev_student.eval()

            if logger:
                skipped = [k for k in prev.keys() if k not in ok]
                logger.info(
                    f"[CIL] task{task}: restore {len(ok)}/{len(prev)} params "
                    f"(skipped {len(skipped)} classifier params)"
                )
        else:
            prev_student = None

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
                lr=float(cfg.get("student_lr", 5e-4)),
                weight_decay=float(cfg.get("student_weight_decay", 5e-4)),
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
                prev_student=prev_student,
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
                lr=float(cfg.get("student_lr", 5e-4)),
                weight_decay=float(cfg.get("student_weight_decay", 5e-4)),
                device=device,
                cfg=cfg,
            )

        # ───────── Class-Balanced finetune ─────────
        if cfg.get("cb_finetune_epochs", 0) > 0:
            from data.cifar100_cl import get_balanced_loader
            finetune_loader = get_balanced_loader(
                task_id=task,
                n_tasks=n_tasks,
                buffer_size=cfg.get("buffer_size", 20),
                batch_size=cfg.get("batch_size", 128),
                num_workers=cfg.get("num_workers", 2),
            )
            simple_finetune(
                student,
                finetune_loader,
                lr=cfg.get("cb_finetune_lr", 1e-4),
                epochs=cfg.get("cb_finetune_epochs", 2),
                device=device,
                weight_decay=0.0,
                cfg=cfg,
                ckpt_path=f"{ckpt_dir}/task{task}_student_ft.pth",
            )

        torch.save(student.state_dict(), f"{ckpt_dir}/task{task}_student.pth")

        # ───────── Task 종료 요약 ─────────
        acc_cur  = evaluate_acc(student, test_cur,  device=device)  # 현재 task
        acc_seen = evaluate_acc(student, test_seen, device=device)  # 전체 class

        acc_seen_hist.append(acc_seen)

        msg = (f"[CIL] Task {task} done │ cur={acc_cur:.2f}% "
               f"│ seen={acc_seen:.2f}%")
        print(msg)
        if logger is not None:
            logger.info(msg)
            logger.update_metric(f"task{task}_acc_cur",  float(acc_cur))
            logger.update_metric(f"task{task}_acc_seen", float(acc_seen))

    import numpy as np
    avg_acc = np.mean(acc_seen_hist)
    best_prev = np.maximum.accumulate(acc_seen_hist)
    forget = best_prev[:-1] - np.array(acc_seen_hist[1:])
    avg_forgetting = forget.mean() if len(forget) > 0 else 0.0

    print(f"\n[SUMMARY] AACC={avg_acc:.2f}%  |  Avg-Forgetting={avg_forgetting:.2f}%")
    if logger is not None:
        logger.update_metric("AACC", float(avg_acc))
        logger.update_metric("AvgForget", float(avg_forgetting))
        logger.finalize()

