# main.py

import argparse
import os
import sys
import yaml
import torch
from torch.optim import Adam, AdamW
import math

from models.ib.vib_mbm import VIB_MBM
from models.ib.proj_head import StudentProj
from utils.misc import set_random_seed
from utils.eval import evaluate_acc
from data.cifar100 import get_cifar100_loaders
from models.teachers.teacher_resnet import create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from utils.model_factory import create_student_by_name               # NEW
from trainer import teacher_vib_update, student_vib_update, simple_finetune
from utils.freeze import freeze_all
from utils.logger import ExperimentLogger
from utils.print_cfg import print_hparams

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="IB-KD entry point")
parser.add_argument('--cfg', default='configs/minimal.yaml', help='YAML config')
parser.add_argument('--teacher1_ckpt', type=str, help='Path to teacher-1 checkpoint')
parser.add_argument('--teacher2_ckpt', type=str, help='Path to teacher-2 checkpoint')
parser.add_argument('--results_dir', type=str, help='Where to save logs / checkpoints')
parser.add_argument('--batch_size', type=int, help='Mini-batch size for training')
parser.add_argument('--method', type=str, help='vib | dkd | crd | vanilla | ce')
args = parser.parse_args()
with open(args.cfg, "r") as f:
    cfg_raw = list(yaml.safe_load_all(f))  # 여러 문서 대비
cfg = cfg_raw[0] if isinstance(cfg_raw, list) else cfg_raw
if not isinstance(cfg, dict):
    raise TypeError(
        f"{args.cfg} 루트는 dict 여야 합니다 (현재: {type(cfg).__name__})"
    )

for k in (
    'teacher1_ckpt', 'teacher2_ckpt', 'results_dir',
    'batch_size', 'method'
):
    v = getattr(args, k, None)
    if v is not None:
        cfg[k] = v

# results_dir를 CLI※YAML 최종값으로 덮어쓴 뒤에 생성
os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)

# logger는 **최종 cfg**가 완성된 뒤에 생성
logger = ExperimentLogger(cfg, exp_name="ibkd")

# 전체 하이퍼파라미터 테이블 출력 (logger 사용)
print_hparams(cfg, log_fn=logger.info)

device = cfg.get('device', 'cuda')
set_random_seed(cfg.get('seed', 42))
method = cfg.get('method', 'vib').lower()
assert method in {'vib', 'dkd', 'crd', 'vanilla', 'ce', 'kd'}, "unknown method"

# ---------- data ----------
train_loader, test_loader = get_cifar100_loaders(
    root=cfg.get('dataset_root', './data'),
    batch_size=cfg.get('batch_size', 128),
    num_workers=cfg.get('num_workers', 0),
    randaug_N=cfg.get('randaug_N', 0),
    randaug_M=cfg.get('randaug_M', 0),
)

# ---------- teachers ----------
if method != 'ce':
    t1 = create_resnet152(pretrained=True, small_input=True).to(device)
    t2 = create_efficientnet_b2(pretrained=True, small_input=True).to(device)

# optional short fine-tuning before distillation
ft_epochs = cfg.get('finetune_epochs', 0)
ft_lr = cfg.get('finetune_lr', 1e-4)
student_iters = cfg.get('student_iters', 60)

t1_ckpt = cfg.get('teacher1_ckpt')
t2_ckpt = cfg.get('teacher2_ckpt')
loaded1 = False
loaded2 = False

if method != 'ce' and t1_ckpt and os.path.exists(t1_ckpt):
    t1.load_state_dict(
        torch.load(t1_ckpt, map_location=device, weights_only=True)
    )
    print(f"[INFO] Loaded teacher1 checkpoint: {t1_ckpt}")
    acc1 = evaluate_acc(t1, test_loader, device)
    print(f"[INFO] teacher1 accuracy: {acc1:.2f}%")
    logger.update_metric("teacher1_acc", float(acc1))
    loaded1 = True

if method != 'ce' and t2_ckpt and os.path.exists(t2_ckpt):
    t2.load_state_dict(
        torch.load(t2_ckpt, map_location=device, weights_only=True)
    )
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

# ---------- VIB-MBM ----------
if method != 'ce':
    in1 = t1.get_feat_dim(); in2 = t2.get_feat_dim()
    mbm = VIB_MBM(in1, in2, cfg['z_dim'], n_cls=100).to(device)
else:
    mbm = None

# ---------- student ----------
student = create_student_by_name(
    cfg.get("student_type", "convnext_tiny"),   # ex) "convnext_small"
    num_classes=100,
    pretrained=True,
    small_input=True,
    cfg=cfg,
).to(device)
if method != 'ce':
    proj = StudentProj(
        in_dim        = student.get_feat_dim(),
        out_dim       = cfg['z_dim'],
        hidden_dim    = cfg.get('proj_hidden_dim'),
        normalize     = True,
        use_bn        = cfg.get('proj_use_bn', False),
    ).to(device)
else:
    proj = None  # CE baseline은 필요 없음

if method != 'ce':
    opt_t = Adam(
        mbm.parameters(),
        lr=float(cfg.get("teacher_lr", 1e-3)),            # 이미 YAML에서 1e‑3 지정
        weight_decay=float(cfg.get("teacher_weight_decay", 0.0)),
    )
else:
    opt_t = None

if method != 'ce':
    base_lr = float(cfg.get("student_lr", 5e-4))
    opt_s = AdamW(
        list(student.parameters()) + list(proj.parameters()),
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
    teacher_vib_update(
        t1,
        t2,
        mbm,
        train_loader,
        cfg,
        opt_t,
        test_loader=test_loader,
        logger=logger,
    )
    student_vib_update(
        t1,
        t2,
        student,
        mbm,
        proj,
        train_loader,
        cfg,
        opt_s,
        test_loader=test_loader,
        logger=logger,
        scheduler=scheduler,
    )

elif method == 'kd':
    from methods.feature_kd import FeatureKD
    distiller = FeatureKD(
        teacher_model=t1,
        student_model=student,
        alpha=cfg.get('kd_alpha', 1.0),
        temperature=cfg.get('kd_T', 4.0),
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

elif method == 'ce':
    ce_ckpt = os.path.join(cfg.get('results_dir', 'results'), 'student_ce_best.pth')
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
    sys.exit(0)

if cfg.get("eval_after_train", True):
    acc = evaluate_acc(
        student,
        test_loader,
        device,
        mixup_active=(cfg.get("mixup_alpha", 0) > 0 or cfg.get("cutmix_alpha_distill", 0) > 0),
    )
    print(f"Final student accuracy: {acc:.2f}%")
    logger.update_metric("test_acc", float(acc))

logger.finalize()

