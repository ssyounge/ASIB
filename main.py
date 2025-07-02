# main.py

import argparse
import os
import yaml
import torch
from torch.optim import Adam, AdamW

from models.ib.vib_mbm import VIB_MBM
from models.ib.proj_head import StudentProj
from utils.misc import set_random_seed
from utils.eval import evaluate_acc
from data.cifar100 import get_cifar100_loaders
from models.teachers.teacher_resnet import create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.students.student_convnext import create_convnext_tiny
from trainer import teacher_vib_update, student_vib_update, simple_finetune
from utils.freeze import freeze_all
from utils.logger import ExperimentLogger

# ---------- CLI ----------
parser = argparse.ArgumentParser(description="IB-KD entry point")
parser.add_argument('--cfg', default='configs/minimal.yaml', help='YAML config')
parser.add_argument('--teacher1_ckpt', type=str, help='Path to teacher-1 checkpoint')
parser.add_argument('--teacher2_ckpt', type=str, help='Path to teacher-2 checkpoint')
parser.add_argument('--results_dir', type=str, help='Where to save logs / checkpoints')
parser.add_argument('--batch_size', type=int, help='Mini-batch size for training')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))

os.makedirs(cfg.get("results_dir", "results"), exist_ok=True)
logger = ExperimentLogger(cfg, exp_name="ibkd")

for k in ('teacher1_ckpt', 'teacher2_ckpt', 'results_dir', 'batch_size'):
    v = getattr(args, k, None)
    if v is not None:
        cfg[k] = v

device = cfg.get('device', 'cuda')
set_random_seed(cfg.get('seed', 42))

# ---------- data ----------
train_loader, test_loader = get_cifar100_loaders(
    root=cfg.get('dataset_root', './data'),
    batch_size=cfg.get('batch_size', 128),
    num_workers=cfg.get('num_workers', 0),
)

# ---------- teachers ----------
t1 = create_resnet152(pretrained=True, small_input=True).to(device)
t2 = create_efficientnet_b2(pretrained=True, small_input=True).to(device)

# optional short fine-tuning before distillation
ft_epochs = cfg.get('finetune_epochs', 0)
ft_lr = cfg.get('finetune_lr', 1e-4)

t1_ckpt = cfg.get('teacher1_ckpt')
t2_ckpt = cfg.get('teacher2_ckpt')
loaded1 = False
loaded2 = False

if t1_ckpt and os.path.exists(t1_ckpt):
    t1.load_state_dict(
        torch.load(t1_ckpt, map_location=device, weights_only=True)
    )
    print(f"[INFO] Loaded teacher1 checkpoint: {t1_ckpt}")
    acc1 = evaluate_acc(t1, test_loader, device)
    print(f"[INFO] teacher1 accuracy: {acc1:.2f}%")
    logger.update_metric("teacher1_acc", float(acc1))
    loaded1 = True

if t2_ckpt and os.path.exists(t2_ckpt):
    t2.load_state_dict(
        torch.load(t2_ckpt, map_location=device, weights_only=True)
    )
    print(f"[INFO] Loaded teacher2 checkpoint: {t2_ckpt}")
    acc2 = evaluate_acc(t2, test_loader, device)
    print(f"[INFO] teacher2 accuracy: {acc2:.2f}%")
    logger.update_metric("teacher2_acc", float(acc2))
    loaded2 = True

if ft_epochs > 0:
    if not loaded1:
        simple_finetune(
            t1,
            train_loader,
            ft_lr,
            ft_epochs,
            device,
            weight_decay=cfg.get("finetune_weight_decay", 0.0),
            cfg=cfg,
            ckpt_path=t1_ckpt or "checkpoints/teacher1_ft.pth",
        )
    if not loaded2:
        simple_finetune(
            t2,
            train_loader,
            ft_lr,
            ft_epochs,
            device,
            weight_decay=cfg.get("finetune_weight_decay", 0.0),
            cfg=cfg,
            ckpt_path=t2_ckpt or "checkpoints/teacher2_ft.pth",
        )
freeze_all(t1)
freeze_all(t2)
t1.eval()
t2.eval()

# ---------- VIB-MBM ----------
in1 = t1.get_feat_dim(); in2 = t2.get_feat_dim()
mbm = VIB_MBM(in1, in2, cfg['z_dim'], n_cls=100).to(device)

# ---------- student ----------
student = create_convnext_tiny(num_classes=100, small_input=True).to(device)
proj = StudentProj(
    student.get_feat_dim(),
    cfg['z_dim'],
    normalize=cfg.get('proj_normalize', True),
    use_bn=cfg.get('proj_use_bn', False),
).to(device)

opt_t = Adam(
    mbm.parameters(),
    lr=float(cfg.get("teacher_lr", 0.0)),
    weight_decay=float(cfg.get("teacher_weight_decay", 0.0)),
)
opt_s = AdamW(
    list(student.parameters()) + list(proj.parameters()),
    lr=float(cfg.get("student_lr", 0.0)),
    weight_decay=float(cfg.get("student_weight_decay", 0.0)),
)

# ---------- training ----------
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
)

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

