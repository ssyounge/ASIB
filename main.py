# main.py

import argparse
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

# ---------- CLI ----------
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default='configs/minimal.yaml')
args = parser.parse_args()
cfg = yaml.safe_load(open(args.cfg))

device = cfg.get('device', 'cuda')
set_random_seed(cfg.get('seed', 42))

# ---------- data ----------
train_loader, test_loader = get_cifar100_loaders(
    batch_size=cfg['batch_size'], num_workers=cfg.get('num_workers', 0)
)

# ---------- teachers ----------
t1 = create_resnet152(pretrained=True, small_input=True).to(device)
t2 = create_efficientnet_b2(pretrained=True, small_input=True).to(device)

# optional short fine-tuning before distillation
ft_epochs = cfg.get('finetune_epochs', 0)
ft_lr = cfg.get('finetune_lr', 1e-4)
if ft_epochs > 0:
    simple_finetune(t1, train_loader, ft_lr, ft_epochs, device, cfg)
    simple_finetune(t2, train_loader, ft_lr, ft_epochs, device, cfg)
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

opt_t = Adam(mbm.parameters(), lr=cfg['teacher_lr'], weight_decay=cfg['teacher_weight_decay'])
opt_s = AdamW(list(student.parameters()) + list(proj.parameters()),
              lr=cfg['student_lr'], weight_decay=cfg['student_weight_decay'])

# ---------- training ----------
teacher_vib_update(t1, t2, mbm, train_loader, cfg, opt_t)
student_vib_update(t1, t2, student, mbm, proj, train_loader, cfg, opt_s)

acc = evaluate_acc(student, test_loader, device)
print(f'Final student accuracy: {acc:.2f}%')
