#!/usr/bin/env python3
"""Simple teacher CE training script."""
import argparse
import os
import sys
import torch
import yaml

sys.path.append(os.path.dirname(__file__) + "/..")

from data.cifar100 import get_cifar100_loaders
from utils.misc import set_random_seed
from models.teachers.teacher_resnet import create_resnet152
from models.teachers.teacher_efficientnet import create_efficientnet_b2

parser = argparse.ArgumentParser()
parser.add_argument("--teacher", required=True, choices=["resnet152", "efficientnet_b2"])
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--wd", type=float, default=5e-4)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

device = args.device
set_random_seed(42)

train_loader, test_loader = get_cifar100_loaders(batch_size=128, num_workers=2)
num_cls = len(train_loader.dataset.classes)

if args.teacher == "resnet152":
    model = create_resnet152(num_classes=num_cls, small_input=True).to(device)
else:
    model = create_efficientnet_b2(num_classes=num_cls, small_input=True).to(device)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
crit = torch.nn.CrossEntropyLoss()

best = 0.0
for ep in range(1, args.epochs + 1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)[1]
        loss = crit(logits, y)
        loss.backward()
        opt.step()
    model.eval(); corr = tot = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)[1].argmax(1)
            corr += (pred == y).sum().item(); tot += y.size(0)
    acc = 100.0 * corr / tot
    print(f"[TeacherCE] ep={ep}/{args.epochs} acc={acc:.2f}")
    if acc > best:
        best = acc
        torch.save(model.state_dict(), args.ckpt)
print(f"[TeacherCE] best={best:.2f} saved={args.ckpt}")
