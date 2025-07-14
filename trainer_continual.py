"""Minimal continual-learning loop that re-uses KD modules."""
# trainer_continual.py
# NOTE: student_vib_update() 내부에 ‘adaptive clipping (top‑1 %)’이 구현돼 있으므로
#       trainer_continual.py에서는 별도 수정이 필요 없습니다.
from __future__ import annotations

import torch
from torch.utils.tensorboard import SummaryWriter
import os, wandb                   # ← os 먼저 import

from data.cifar100_cl import get_cifar100_cl_loaders, _task_classes
from trainer import teacher_vib_update, student_vib_update, simple_finetune
# ============================================================
# 선택적 regularizer 로드 (EWC, LwF, DER …)
# ============================================================
from methods.ewc import EWC          # (규제용)
from methods.lwf import LwF          # (옵션) 추가 실험 대비
from utils.model_factory import create_teacher_by_name
from utils.freeze import freeze_all
from utils.eval import evaluate_acc
import torch.nn as nn
from types import ModuleType
import inspect


# ── student 모델 안에서 Linear classifier 를 찾아 반환 ──
def _find_linear_head(model, *, n_classes: int | None = None) -> tuple[nn.Linear, list[str]]:
    """
    다양한 wrapper 에서 최종 nn.Linear head 를 찾는다.
    1) 미리 정의한 path 후보 검사 → 2) 재귀 탐색 순으로 진행.
    반환값: (모듈객체, ["parent", "child", ...] 경로)
    """
    CANDIDATES = [
        ["classifier"], ["head"], ["fc"],
        ["model", "classifier"], ["model", "head"], ["model", "fc"],
        ["net", "classifier"],   ["net", "head"],   ["net", "fc"],
        ["backbone", "classifier"], ["backbone", "head"], ["backbone", "fc"],
    ]

    def _get_by_path(root, path):
        m = root
        for name in path:
            if not hasattr(m, name):
                return None
            m = getattr(m, name)
        return m

    # ── 1) 후보 path 빠른 검사 ─────────────────────────
    for path in CANDIDATES:
        m = _get_by_path(model, path)
        if isinstance(m, nn.Linear):
            return m, path

    # ── 2) 전체 재귀 탐색 (마지막 Linear 선택) ────────
    last_linear, last_name = None, None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            if n_classes is None or mod.out_features == n_classes:
                last_linear, last_name = mod, name
            else:
                # 기록해 두고, 일단 계속 순회 – out_features 기준으로 나중에 최대값 사용
                last_linear, last_name = mod, name

    if last_linear is not None:
        return last_linear, last_name.split(".")

    raise AttributeError("Linear classifier(head) 를 찾지 못했습니다.")


def _remap_for_task(logits: torch.Tensor,
                    target: torch.Tensor,
                    classes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Slice logits/labels to the current task classes."""
    device = logits.device
    cls_tensor = torch.tensor(classes, dtype=torch.long, device=device)
    logits_t = logits.index_select(dim=1, index=cls_tensor)
    tgt_local = torch.searchsorted(cls_tensor, target, right=False)
    return logits_t, tgt_local


def _expand_head(model, n_new):
    """모델의 최종 nn.Linear head 를 in-place 로 확장."""
    import torch.nn as nn

    old_head, path = _find_linear_head(model, n_classes=None)
    in_f = old_head.in_features
    out_f_old = old_head.out_features

    new_head = nn.Linear(in_f, out_f_old + n_new)
    new_head.weight.data[:out_f_old] = old_head.weight.data.clone()
    new_head.bias.data[:out_f_old] = old_head.bias.data.clone()
    nn.init.normal_(new_head.weight.data[out_f_old:], std=0.02)
    nn.init.constant_(new_head.bias.data[out_f_old:], 0.0)
    # ─ 새 head 를 원래 위치에 다시 꽂아 넣기 ─
    parent = model
    for p in path[:-1]:
        parent = getattr(parent, p)
    setattr(parent, path[-1], new_head.to(old_head.weight.device))
    print(f"[HEAD] expanded: {out_f_old} → {out_f_old + n_new}")


def run_continual(cfg: dict, kd_method: str, logger=None) -> None:
    """Run continual-learning training using KD modules."""
    if logger is None:
        import logging, datetime, pathlib
        log_dir = pathlib.Path(cfg.get("results_dir", "results"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
        logging.basicConfig(
            filename=log_path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s │ %(message)s",
        )
        logger = logging.getLogger("train")
        print(f"[LOGGER] File logging → {log_path}")

    device = cfg.get("device", "cuda")
    n_tasks = cfg.get("n_tasks", 10)
    if 100 % n_tasks != 0:
        raise ValueError(
            f"n_tasks={n_tasks} 가 CIFAR-100 을 균등 분할하지 않습니다."
        )
    ckpt_dir = os.path.join(cfg.get("results_dir", "results"), "cl_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    acc_seen_hist = []
    writer = SummaryWriter(log_dir=cfg.get("tb_log_dir", "runs/kd_monitor"))

    # ─────────────────────────────────────────
    #  WandB: API Key 유무에 따라 자동 Fallback
    #    · ①  환경변수 WANDB_API_KEY 나
    #    · ②  ~/.config/wandb/settings 에 저장된 키
    #        둘 다 없으면 → offline 전환
    # ─────────────────────────────────────────
    if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
        os.environ["WANDB_MODE"] = "offline"
        print("[INFO] W&B API key not found → running in OFFLINE mode")

    wandb_run = wandb.init(
        project=cfg.get("wandb_project", "kd_monitor"),
        name   =cfg.get("wandb_run_name", "run_001"),
    )
    global_step_counter = 0
    ewc_bank = []
    regularizers = ewc_bank  # alias for optional regularizers

    t1 = create_teacher_by_name(
        cfg.get("teacher1_type", "resnet152"),
        pretrained=True,
        small_input=True,
    ).to(device)
    t2 = create_teacher_by_name(
        cfg.get("teacher2_type", "efficientnet_b2"),
        pretrained=True,
        small_input=True,
    ).to(device)
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
            num_cls_total,                    # 100-way (확정)
            cfg["z_dim"],                     # latent z
            beta=cfg.get("beta_bottleneck", 1e-3),
            clamp=(
                cfg.get("latent_clamp_min", -6),
                cfg.get("latent_clamp_max", 2),
            ),
        ).to(device)
    else:
        vib_mbm = None

    for task in range(n_tasks):
        # ───────── Task 헤더 ─────────
        head = f"[CIL] ── Task {task+1}/{n_tasks} " + "─"*40
        print("\n" + head)
        if logger is not None:
            logger.info(head)
        train_loader, test_cur, test_seen, dataset, task_split = get_cifar100_cl_loaders(
            root=cfg.get("dataset_root", "./data"),
            task_id=task,
            n_tasks=n_tasks,
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 2),
            randaug_N=cfg.get("randaug_N", 0),
            randaug_M=cfg.get("randaug_M", 0),
            persistent_train=cfg.get("persistent_workers", False),
            return_task_split=True,
            buffer_size=cfg.get("buffer_size", 20),
        )

        if cfg.get("train_mode") == "continual":
            from utils.dataloader import BalancedReplaySampler

            cur_idx = task_split["cur_indices"]
            rep_idx = task_split.get("replay_indices", [])
            sampler = BalancedReplaySampler(
                cur_idx,
                rep_idx,
                batch_size=cfg.get("batch_size", 128),
                ratio=cfg.get("replay_ratio", 0.5),
                shuffle=True,
            )
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.get("batch_size", 128),
                sampler=sampler,
                num_workers=cfg.get("num_workers", 2),
                persistent_workers=cfg.get("persistent_workers", False),
            )

        cur_cls = _task_classes(task, n_tasks)
        prev_cls = sum((_task_classes(t, n_tasks) for t in range(task)), [])
        cfg['task_meta'] = {
            'classes': cur_cls,
            'cur_class_ids': cur_cls,
            'prev_class_ids': prev_cls,
            'id': task,
        }

        if task == 0 and vib_mbm is not None:
            opt_t = torch.optim.Adam(
                vib_mbm.parameters(),
                lr=float(cfg.get("teacher_lr", 1e-3)),
                weight_decay=float(cfg.get("teacher_weight_decay", 0.0)),
            )
            global_step_counter = teacher_vib_update(
                t1,
                t2,
                vib_mbm,
                train_loader,
                cfg,
                opt_t,
                test_loader=test_cur,
                logger=logger,
                writer=writer,
                wandb_run=wandb_run,
                global_step_offset=global_step_counter,
                ewc_bank=regularizers,
            )

        from utils.model_factory import create_student_by_name

        NUM_ALL = cfg.get("num_classes", 100)
        student = create_student_by_name(
            cfg.get("student_type", "convnext_tiny"),
            num_classes=NUM_ALL,
            pretrained=True,
            small_input=True,
            cfg=cfg,
        ).to(device)

# ---------- NEW : 100-way CE 사전학습 가중치 불러오기 ----------
        ce_ckpt = cfg.get("student_ce_ckpt")
        if task == 0 and ce_ckpt and os.path.isfile(ce_ckpt):
            print(f"[INIT] loading student CE checkpoint → {ce_ckpt}")
            student.load_state_dict(
                torch.load(ce_ckpt, map_location="cpu"), strict=False
            )
        head, path = _find_linear_head(student, n_classes=NUM_ALL)
        if logger:
            logger.info(
                f"[Task {task}] detected head path = {'.'.join(path)}  (out={head.out_features})"
            )

        # ─── 이어 학습: 이전 task 가중치 로드 ───
        prev_ckpt = f"{ckpt_dir}/task{task-1}_student.pth"
        prev_student = None
        if task > 0 and os.path.isfile(prev_ckpt):
            n_new = 100 // n_tasks           # task 당 class 수 (=10)
            head, _ = _find_linear_head(student, n_classes=NUM_ALL)
            if head.out_features < (task + 1) * n_new:
                _expand_head(student, n_new)

            if logger:
                head, _ = _find_linear_head(student, n_classes=NUM_ALL)
                logger.info(f"[Task {task}] Head dim = {head.out_features}")

            student.load_state_dict(
                torch.load(prev_ckpt, map_location="cpu"),
                strict=False
            )

            from utils.model_factory import create_student_by_name
            prev_student = create_student_by_name(
                cfg.get("student_type", "convnext_tiny"),
                num_classes=NUM_ALL,
                pretrained=False,
                small_input=True,
                cfg=cfg,
            ).to(device)
            prev_student.load_state_dict(
                torch.load(prev_ckpt, map_location="cpu"),
                strict=False
            )
            prev_student.eval()

            # (★) partial‑freeze 삭제 → 모든 파라미터 학습
            for p in student.parameters():
                p.requires_grad_(True)

            if logger:
                trainable = [n for n, p in student.named_parameters() if p.requires_grad]
                logger.info(f"[CIL] task{task}: trainable params ⇒ {len(trainable)} tensors "
                            f"{', '.join(trainable[:5])}{' …' if len(trainable)>5 else ''}")
                logger.info(
                    f"[CIL] task{task}: restore from {prev_ckpt}"
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
            def trainable(p):
                return p.requires_grad

            opt_s = torch.optim.AdamW(
                list(filter(trainable, student.parameters())) +
                list(proj.parameters()),
                lr=float(cfg.get("student_lr", 5e-4)),
                weight_decay=float(cfg.get("student_weight_decay", 5e-4)),
            )
            global_step_counter = student_vib_update(
                t1,
                t2,
                student,
                vib_mbm,
                proj,
                train_loader,
                cfg,               # cfg is passed once as a positional argument
                opt_s,
                test_loader=test_cur,
                logger=logger,
                prev_student=prev_student,
                writer=writer,
                wandb_run=wandb_run,
                global_step_offset=global_step_counter,
                ewc_bank=regularizers,
            )

            if cfg.get("use_ewc", False):
                ewc_obj = EWC(
                    student,
                    train_loader,
                    device=device,
                    samples=cfg.get("ewc_samples", 1024),
                    online=cfg.get("ewc_online", False),
                    decay=cfg.get("ewc_decay", 1.0),
                    lambda_=cfg.get("ewc_lambda", 30.0),
                )
                regularizers.append(ewc_obj)
        else:
            if kd_method == "none":
                simple_finetune(
                    student,
                    train_loader,
                    lr=cfg.get("student_lr", 5e-4),
                    epochs=cfg.get("student_iters", 60),
                    device=device,
                    weight_decay=float(cfg.get("student_weight_decay", 5e-4)),
                    cfg=cfg,
                )
            elif kd_method == "dkd":
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

            if kd_method != "none":
                distiller.train_distillation(
                    train_loader,
                    test_cur,
                    epochs=cfg.get("student_iters", 60),
                    lr=float(cfg.get("student_lr", 5e-4)),
                    weight_decay=float(cfg.get("student_weight_decay", 5e-4)),
                    device=device,
                    cfg=cfg,
                )

            if cfg.get("use_ewc", False):
                ewc_obj = EWC(
                    student,
                    train_loader,
                    device=device,
                    samples=cfg.get("ewc_samples", 1024),
                    online=cfg.get("ewc_online", False),
                    decay=cfg.get("ewc_decay", 1.0),
                    lambda_=cfg.get("ewc_lambda", 30.0),
                )
                regularizers.append(ewc_obj)

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
        cur_cls  = _task_classes(task, n_tasks)
        seen_cls = sum((_task_classes(t, n_tasks) for t in range(task + 1)), [])

        acc_cur  = evaluate_acc(
            student, test_cur, device=device, classes=cur_cls
        )  # 현재 task
        acc_seen = evaluate_acc(
            student, test_seen, device=device, classes=seen_cls
        )  # 전체 class

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
    writer.close()
    wandb_run.finish()

