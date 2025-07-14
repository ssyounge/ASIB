# main.py
from __future__ import annotations

import argparse
import os
import sys
import yaml, os                                   # glob 불필요
from pathlib import Path
import torch
from torch.optim import Adam, AdamW
import math
import multiprocessing as mp           # 전역 mp = std multiprocessing
import torch.multiprocessing as _tmp_mp

# CUDA / torch 호출 **전에** spawn 모드 고정
try:
    _tmp_mp.set_start_method("spawn", force=True)
except RuntimeError:
    # 이미 설정돼 있으면 그대로 둔다
    pass

from models.ib.gate_mbm import GateMBM
from models.ib.proj_head import StudentProj
from torch.utils.tensorboard import SummaryWriter
import os, wandb                   # ← os 먼저 import
from utils.misc import set_random_seed
from utils.eval import evaluate_acc
from data.cifar100 import get_cifar100_loaders
from models.ensemble.snapshot_teacher import SnapshotTeacher
from utils.model_factory import create_student_by_name, create_teacher_by_name               # NEW
from trainer import teacher_vib_update, student_vib_update, simple_finetune
from utils.freeze import freeze_all
from utils.logger import ExperimentLogger
from utils.print_cfg import print_hparams
from utils.path_utils import to_writable


def get_method_cfg(method: str, train_mode: str):
    """Load method-specific YAML with fallback for legacy paths."""
    cand1 = Path(f"configs/method/{method}/{train_mode}.yaml")
    cand2 = Path(f"configs/method/{method}.yaml")

    if cand1.exists():
        with cand1.open() as f:
            return yaml.safe_load(f)
    if cand2.exists():
        if train_mode == "continual" and method != "vib":
            raise RuntimeError(
                "train_mode 'continual' is currently only supported for method 'vib'."
            )
        with cand2.open() as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(
        f"No yaml found for method={method}, train_mode={train_mode}"
    )


def main() -> None:
    # ---------- CLI ----------
    parser = argparse.ArgumentParser(description="IB-KD entry point")
    parser.add_argument('--cfg', '--config',
                        dest='cfg',
                        default='configs/base.yaml',
                        help='Comma-separated list of YAML files')
    parser.add_argument('--teacher1_ckpt', type=str, help='Path to teacher-1 checkpoint')
    parser.add_argument('--teacher2_ckpt', type=str, help='Path to teacher-2 checkpoint')
    parser.add_argument('--results_dir', type=str, help='Where to save logs / checkpoints')
    parser.add_argument('--batch_size', type=int, help='Mini-batch size for training')
    parser.add_argument('--method', type=str, help='Override KD algorithm')   # optional
    parser.add_argument('--train_mode', type=str, help='Override scenario')   # optional
    parser.add_argument('--n_tasks', type=int, help='number of tasks for continual learning')
    args = parser.parse_args()
    # ① 여러 YAML 파일 병합 (쉼표 구분) + “어느 파일에서 왔는지” 추적
    cfg: dict = {}
    cfg_src: dict[str, str] = {}

    def _merge_yaml(path: str | os.PathLike, label: str):
        """로드한 YAML을 cfg 에 병합하면서, key → label 매핑도 갱신."""
        with open(path, "r") as f:
            y = yaml.safe_load(f) or {}
        for k, v in y.items():
            cfg[k] = v
            cfg_src[k] = label

    # --- ①‑A. CLI 로 받은 --cfg (base 등) ----------------------------------
    for p in args.cfg.split(','):
        p = p.strip()
        _merge_yaml(p, label=Path(p).stem)
    if not isinstance(cfg, dict):
        raise TypeError(
            f"{args.cfg} 루트는 dict 여야 합니다 (현재: {type(cfg).__name__})"
        )

    # ────────────────────────────────────────────────
    # ② Scenario  →  Method  순으로 추가 YAML 병합
    #
    #    precedence:
    #      base  <  control  <  scenario  <  method  <  CLI‑override
    # --------------------------------------------------------------

    scenario = (args.train_mode or cfg.get('train_mode', 'standard')).lower()
    _merge_yaml(f"configs/scenario/{scenario}.yaml", label="scenario")
    cfg['train_mode'] = scenario

    method = (args.method or cfg.get('method'))
    if method is None:
        raise ValueError(
            "method 가 지정되지 않았습니다. "
            "(control.yaml 에 method: ..., 또는 --method 인수)"
        )
    method = method.lower()
    method_yaml = get_method_cfg(method, scenario) or {}
    for k, v in method_yaml.items():
        cfg[k] = v
        cfg_src[k] = "method"
    cfg['method'] = method

    # --------------------------------------------------------------

    # cuDNN 자동 튜닝 활성 (deterministic 모드가 아니라면)
    if not cfg.get("deterministic", False):
        torch.backends.cudnn.benchmark = True

    for k in (
        'teacher1_ckpt', 'teacher2_ckpt', 'results_dir',
        'batch_size', 'method', 'train_mode', 'n_tasks'
    ):
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
            cfg_src[k] = "CLI"

    # ✔ 자동 경로 보정: repo 안 상대경로 → $HOME/.asmb_kd/
    for key in ("checkpoint_dir",):
        if key in cfg:
            cfg[key] = to_writable(cfg[key])

    # results_dir를 CLI※YAML 최종값으로 덮어쓴 뒤에 생성
    os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)

    # logger는 **최종 cfg**가 완성된 뒤에 생성
    logger = ExperimentLogger(cfg, exp_name=cfg.get("exp_name", "ibkd"))
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

    # ──────────────────────────────────────────────────────────────
    #   (A) 전체 테이블 +  (B) 그룹별 테이블 동시 출력
    # --------------------------------------------------------------
    from utils.print_cfg import print_hparams_grouped
    print_hparams(cfg,          title="All Hyper‑parameters", log_fn=logger.info)
    print_hparams_grouped(cfg,  src_map=cfg_src,              log_fn=logger.info)


    device = cfg.get('device', 'cuda')
    set_random_seed(
        cfg.get('seed', 42),
        deterministic=cfg.get('deterministic', False),
    )
    method = cfg.get('method', 'vib').lower()
    mode   = cfg.get('train_mode', 'standard').lower()
    assert method in {
        'vib', 'dkd', 'crd', 'vanilla', 'fitnet', 'at', 'ce', 'none'
    }, f"unknown method: {method}"
    assert mode   in {'standard', 'continual'}, "unknown train_mode"

    # ──────────────────────────────────────────────────────────────
    # Continual 학습은 현재 VIB‑KD 전용
    # ─ mode 가 continual 인데 method 가 vib 가 아니면 즉시 중단
    # ----------------------------------------------------------------
    if mode == 'continual' and method != 'vib':
        raise NotImplementedError(
            "Continual learning currently supports only the 'vib' method."
        )

    # ───────────────────────────────────────────
    # VIB-KD 전용 하이퍼파라미터는 필요할 때만 읽기
    # -------------------------------------------
    if method == 'vib':
        z_dim = cfg.get('z_dim', 512)
        # ── DEBUG: z_dim 값 및 타입 모니터링 ───────────────────────
        print(f"[DBG] resolved z_dim = {z_dim} (type={type(z_dim).__name__})")
        assert isinstance(z_dim, int), (
            f"z_dim must be an int after config merge, "
            f"but got {type(z_dim).__name__}: {z_dim}"
        )
        mbm_type    = cfg.get('mbm_type', 'GATE')
        beta        = cfg.get('beta_bottleneck', 1e-3)
        proj_hidden = cfg.get('proj_hidden_dim', 1024)
        proj_use_bn = cfg.get('proj_use_bn', True)
    else:
        # 다른 KD 알고리즘은 VIB 관련 값을 전혀 사용하지 않음
        z_dim = mbm_type = beta = proj_hidden = proj_use_bn = None

    if mode == 'continual':
        from trainer_continual import run_continual
        run_continual(cfg, method, logger=logger)
        return

    # ---------- data ----------
    train_loader, test_loader = get_cifar100_loaders(
        root=cfg.get('dataset_root', './data'),
        batch_size=cfg.get('batch_size', 128),
        num_workers=cfg.get('num_workers', 0),
        randaug_N=cfg.get('randaug_N', 0),
        randaug_M=cfg.get('randaug_M', 0),
        persistent_train=cfg.get('persistent_workers', False),
        persistent_test=False,
    )

    # ---------- teachers ----------
    if method != 'ce':
        def _make_teacher(cfg_key, type_key, default_name):
            src = cfg.get(cfg_key, "")
            teacher_type = cfg.get(type_key, default_name)
            # 1) comma-separated snapshot ensemble
            if isinstance(src, str) and "," in src:
                src = src.split('#', 1)[0].strip()
                paths = [p.strip() for p in src.split(',') if p.strip()]
                return SnapshotTeacher(
                    paths,
                    backbone_name=teacher_type,
                    n_cls=cfg.get("num_classes", 100),
                ).to(device)

            # 2) single checkpoint path
            if isinstance(src, str) and src.endswith('.pth'):
                m = create_teacher_by_name(
                    teacher_type,
                    num_classes=cfg.get("num_classes", 100),
                    pretrained=False,
                    small_input=True,
                )
                m.load_state_dict(torch.load(src, map_location='cpu'))
                return m.to(device)

            if src:
                raise ValueError(f"{cfg_key}: un-recognised format \u2192 {src}")
            else:
                raise ValueError(f"invalid {cfg_key}: {src}")

        for key in ("teacher1_ckpt", "teacher2_ckpt"):
            if key in cfg:
                cfg[key] = ",".join(
                    to_writable(p.strip()) for p in str(cfg[key]).split(",")
                )

        t1 = _make_teacher('teacher1_ckpt', 'teacher1_type', 'resnet152')
        t2 = _make_teacher('teacher2_ckpt', 'teacher2_type', 'efficientnet_b2')
        loaded1 = isinstance(t1, SnapshotTeacher)
        if loaded1:
            print(f"[INFO] Loaded teacher1 snapshot ensemble: {len(t1.models)} models")
        loaded2 = isinstance(t2, SnapshotTeacher)
        if loaded2:
            print(f"[INFO] Loaded teacher2 snapshot ensemble: {len(t2.models)} models")

    # optional short fine-tuning before distillation
    ft_epochs = cfg.get('finetune_epochs', 0)
    ft_lr = cfg.get('finetune_lr', 1e-4)
    student_iters = cfg.get('student_iters', 60)

    t1_ckpt = cfg.get('teacher1_ckpt')
    t2_ckpt = cfg.get('teacher2_ckpt')

    if method != 'ce' and t1_ckpt and os.path.exists(t1_ckpt):
        # PyTorch 1.12 이후만 weights_only 지원 → 버전별 fallback
        try:
            t1.load_state_dict(
                torch.load(t1_ckpt, map_location=device, weights_only=True)
            )
        except TypeError:   # 구버전
            t1.load_state_dict(torch.load(t1_ckpt, map_location=device))
        print(f"[INFO] Loaded teacher1 checkpoint: {t1_ckpt}")
        acc1 = evaluate_acc(t1, test_loader, device)
        print(f"[INFO] teacher1 accuracy: {acc1:.2f}%")
        logger.update_metric("teacher1_acc", float(acc1))
        loaded1 = True

    if method != 'ce' and t2_ckpt and os.path.exists(t2_ckpt):
        # PyTorch 1.12 이후만 weights_only 지원 → 버전별 fallback
        try:
            t2.load_state_dict(
                torch.load(t2_ckpt, map_location=device, weights_only=True)
            )
        except TypeError:   # 구버전
            t2.load_state_dict(torch.load(t2_ckpt, map_location=device))
        print(f"[INFO] Loaded teacher2 checkpoint: {t2_ckpt}")
        acc2 = evaluate_acc(t2, test_loader, device)
        print(f"[INFO] teacher2 accuracy: {acc2:.2f}%")
        logger.update_metric("teacher2_acc", float(acc2))
        loaded2 = True

    if method != 'ce' and ft_epochs > 0:
        ft_loader, _ = get_cifar100_loaders(
            root=cfg.get('dataset_root', './data'),
            batch_size=cfg.get('batch_size', 128),
            num_workers=cfg.get('num_workers', 0),
            randaug_N=cfg.get('finetune_randaug_N', 0),
            randaug_M=cfg.get('finetune_randaug_M', 0),
            persistent_train=cfg.get('persistent_workers', False),
        )
        if not loaded1:
            simple_finetune(
                t1,
                ft_loader,
                ft_lr,
                ft_epochs,
                device,
                weight_decay=cfg.get("finetune_weight_decay", 0.0),
                cfg={**cfg, "finetune_eval_loader": test_loader},   # NEW
                ckpt_path=t1_ckpt or "checkpoints/teacher1_ft.pth",
            )
        if not loaded2:
            simple_finetune(
                t2,
                ft_loader,
                ft_lr,
                ft_epochs,
                device,
                weight_decay=cfg.get("finetune_weight_decay", 0.0),
                cfg={**cfg, "finetune_eval_loader": test_loader},   # NEW
                ckpt_path=t2_ckpt or "checkpoints/teacher2_ft.pth",
            )
    if method != 'ce':
        freeze_all(t1)
        freeze_all(t2)
        t1.eval()
        t2.eval()

    # ---------- VIB‑전용 모듈 ----------
    if method == 'vib':
        in1 = t1.get_feat_dim(); in2 = t2.get_feat_dim()
        vib_mbm = GateMBM(
            in1,
            in2,
            cfg.get('num_classes', 100),
            z_dim,
            beta=beta,
            clamp=(
                cfg.get('latent_clamp_min', -6),
                cfg.get('latent_clamp_max', 2),
            ),
            dropout_p=cfg.get('gate_dropout', 0.1),
        ).to(device)
    else:
        vib_mbm = None

    # ---------- student ----------
    student = create_student_by_name(
        cfg.get("student_type", "convnext_tiny"),   # ex) "convnext_small"
        num_classes=cfg.get('num_classes', 100),
        pretrained=True,
        small_input=True,
        cfg=cfg,
    ).to(device)
    if cfg.get('student_ce_ckpt') and os.path.isfile(cfg['student_ce_ckpt']):
        ckpt = torch.load(cfg['student_ce_ckpt'], map_location="cpu")
        missing, unexpected = student.load_state_dict(ckpt, strict=False)
        logger.info(
            f"[Student] CE-ckpt loaded \u2713  missing={len(missing)}  unexpected={len(unexpected)}"
        )
    else:
        logger.warning("[Student] CE-ckpt **NOT** found \u2192 training from scratch")
    if method == 'vib':
        student_proj = StudentProj(
            in_dim     = student.get_feat_dim(),
            out_dim    = z_dim,
            hidden_dim = proj_hidden,
            use_bn     = proj_use_bn,
        ).to(device)
    else:
        student_proj = None  # 다른 KD 알고리즘은 사용 안 함

    if method == 'vib':
        opt_t = Adam(
            vib_mbm.parameters(),
            lr=float(cfg.get("teacher_lr", 1e-3)),            # 이미 YAML에서 1e‑3 지정
            weight_decay=float(cfg.get("teacher_weight_decay", 0.0)),
        )
    else:
        opt_t = None

    if method == 'vib':
        base_lr = float(cfg.get("student_lr", 5e-4))
        opt_s = AdamW(
            list(student.parameters()) + list(student_proj.parameters()),
            lr=base_lr,
            weight_decay=float(cfg.get("student_weight_decay", 0.0)),
        )

        # ─ warm‑up scheduler (cosine 기본) ─
        warm_epochs = cfg.get("lr_warmup_epochs", 5)
        total_epochs = student_iters

        def lr_lambda(cur_epoch):
            if cur_epoch < warm_epochs:
                return (cur_epoch + 1) / warm_epochs
            t = (cur_epoch - warm_epochs) / max(1, total_epochs - warm_epochs)
            return 0.5 * (1 + math.cos(math.pi * t))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt_s, lr_lambda)
    else:
        opt_s = None
        scheduler = None

    # ---------- training ----------
    if method == 'vib':
        global_step_counter = teacher_vib_update(
            t1,
            t2,
            vib_mbm,
            train_loader,
            cfg,
            opt_t,
            test_loader=test_loader,
            logger=logger,
            writer=writer,
            wandb_run=wandb_run,
            global_step_offset=global_step_counter,
        )
        global_step_counter = student_vib_update(
            t1,
            t2,
            student,
            vib_mbm,
            student_proj,
            train_loader,
            cfg,
            opt_s,
            test_loader=test_loader,
            logger=logger,
            scheduler=scheduler,
            writer=writer,
            wandb_run=wandb_run,
            global_step_offset=global_step_counter,
        )

    elif method == 'crd':
        from methods.crd import CRDDistiller
        distiller = CRDDistiller(
            teacher_model=t1,
            student_model=student,
            alpha=cfg.get('crd_alpha', 0.5),
            temperature=cfg.get('crd_T', 0.07),
            label_smoothing=cfg.get('label_smoothing', 0.0),
            config=cfg,
        )
        acc = distiller.train_distillation(
            train_loader,
            test_loader,
            epochs=cfg.get('student_iters', 60),
            lr=cfg.get('student_lr', 5e-4),
            weight_decay=cfg.get('student_weight_decay', 5e-4),
            device=device,
            cfg=cfg,
        )
        logger.update_metric("student_acc", float(acc))

    elif method == 'dkd':
        from methods.dkd import DKDDistiller
        distiller = DKDDistiller(
            teacher_model=t1,
            student_model=student,
            alpha=cfg.get('dkd_alpha', 1.0),
            beta=cfg.get('dkd_beta', 8.0),
            temperature=cfg.get('dkd_T', 4.0),
            warmup=cfg.get('dkd_warmup', 5),
            label_smoothing=cfg.get('label_smoothing', 0.0),
            config=cfg,
        )
        acc = distiller.train_distillation(
            train_loader,
            test_loader,
            epochs=cfg.get('student_iters', 60),
            lr=cfg.get('student_lr', 5e-4),
            weight_decay=cfg.get('student_weight_decay', 5e-4),
            device=device,
            cfg=cfg,
        )
        logger.update_metric("student_acc", float(acc))

    elif method == 'vanilla':
        from methods.vanilla_kd import VanillaKDDistiller
        distiller = VanillaKDDistiller(
            teacher_model=t1,
            student_model=student,
            alpha=cfg.get('vanilla_alpha', 0.5),
            temperature=cfg.get('vanilla_T', 4.0),
            config=cfg,
        )
        acc = distiller.train_distillation(
            train_loader,
            test_loader,
            epochs=cfg.get('student_iters', 60),
            lr=cfg.get('student_lr', 5e-4),
            weight_decay=cfg.get('student_weight_decay', 5e-4),
            device=device,
            cfg=cfg,
        )
        logger.update_metric("student_acc", float(acc))

    elif method == 'fitnet':
        from methods.fitnet import FitNetDistiller
        distiller = FitNetDistiller(
            teacher_model=t1,
            student_model=student,
            alpha_hint=cfg.get('alpha_hint', 1.0),
            alpha_ce=cfg.get('alpha_ce', 1.0),
            label_smoothing=cfg.get('label_smoothing', 0.0),
            config=cfg,
        )
        acc = distiller.train_distillation(
            train_loader,
            test_loader,
            epochs=cfg.get('student_iters', 60),
            lr=cfg.get('student_lr', 5e-4),
            weight_decay=cfg.get('student_weight_decay', 5e-4),
            device=device,
            cfg=cfg,
        )
        logger.update_metric("student_acc", float(acc))

    elif method == 'at':
        from methods.at import ATDistiller
        distiller = ATDistiller(
            teacher_model=t1,
            student_model=student,
            alpha=cfg.get('alpha', 1000),
            layer_key=cfg.get('layer_key', 'feat_4d_layer3'),
            label_smoothing=cfg.get('label_smoothing', 0.0),
            config=cfg,
        )
        acc = distiller.train_distillation(
            train_loader,
            test_loader,
            epochs=cfg.get('student_iters', 60),
            lr=cfg.get('student_lr', 5e-4),
            weight_decay=cfg.get('student_weight_decay', 5e-4),
            device=device,
            cfg=cfg,
        )
        logger.update_metric("student_acc", float(acc))

    elif method == 'ce':
        ce_ckpt = os.path.join(
            cfg.get('results_dir', 'results'),
            cfg.get('student_ce_ckpt', 'student_ce_best.pth'),
        )
        simple_finetune(
            student,
            train_loader,
            lr=cfg.get('student_lr', 5e-4),
            epochs=cfg.get('student_iters', 60),
            device=device,
            weight_decay=cfg.get('student_weight_decay', 0.0),
            cfg=cfg,
            ckpt_path=ce_ckpt,
        )
        acc = evaluate_acc(student, test_loader, device)
        logger.update_metric("student_acc", float(acc))
        logger.finalize()
        writer.close()
        wandb_run.finish()
        sys.exit(0)

    if cfg.get("eval_after_train", True):
        acc = evaluate_acc(
            student,
            test_loader,
            device,
            mixup_active=(
                cfg.get("mixup_alpha", 0) > 0
                or cfg.get("cutmix_alpha_distill", 0) > 0
            ),
        )
        print(f"Final student accuracy: {acc:.2f}%")
        logger.update_metric("test_acc", float(acc))

    logger.finalize()
    writer.close()
    wandb_run.finish()


if __name__ == "__main__":
    mp.freeze_support()
    main()

