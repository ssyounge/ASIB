import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import copy
import torch

from data.cifar100 import get_cifar100_loaders
from utils.model_factory import create_teacher_by_name
from utils.eval import evaluate_acc
from utils.misc import progress


def parse_args():
    p = argparse.ArgumentParser(description="Teacher CE fine-tuning")
    p.add_argument("--teacher_type", type=str, required=True,
                   help="Teacher model type (resnet152, efficientnet_b2, ...)")
    p.add_argument("--epochs", type=int, default=10,
                   help="Number of fine-tuning epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay")
    p.add_argument("--ckpt_path", type=str, default="teacher_ft_ce.pth",
                   help="Path to save best checkpoint")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device to train on")
    return p.parse_args()


def train_teacher_ce(model, train_loader, test_loader, lr, weight_decay,
                      epochs, device, ckpt_path):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in progress(train_loader, desc=f"[Train ep={ep}]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            logit = out[1] if isinstance(out, tuple) else out
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
        acc = evaluate_acc(model, test_loader, device=device)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"[ep={ep}/{epochs}] testAcc={acc:.2f}, best={best_acc:.2f}")

    os.makedirs(os.path.dirname(ckpt_path) or '.', exist_ok=True)
    torch.save(best_state, ckpt_path)
    print(f"[Done] bestAcc={best_acc:.2f} -> {ckpt_path}")
    return best_acc


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    train_loader, test_loader = get_cifar100_loaders()
    num_classes = len(train_loader.dataset.classes)

    teacher = create_teacher_by_name(
        args.teacher_type,
        num_classes=num_classes,
        pretrained=True,
        small_input=True,
    )

    train_teacher_ce(
        teacher,
        train_loader,
        test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        device=device,
        ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    main()
