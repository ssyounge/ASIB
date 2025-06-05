# modules/cutmix_finetune_teacher.py

import os
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def cutmix_data(inputs, targets, alpha=1.0):
    """
    inputs: [N, C, H, W]
    targets: [N]
    alpha>0 => cutmix 적용
    alpha<=0 => cutmix 비활성 (return 그대로)
    """
    if alpha <= 0.0:
        # no cutmix
        return inputs, targets, targets, 1.0

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=inputs.device)
    lam = random.betavariate(alpha, alpha)

    # crop box
    W, H = inputs.size(2), inputs.size(3)
    cut_w = int(W * (1 - lam))
    cut_h = int(H * (1 - lam))
    cx = random.randint(0, W)
    cy = random.randint(0, H)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    # clone
    inputs_clone = inputs.clone()
    inputs_clone[:, :, x1:x2, y1:y2] = inputs[indices, :, x1:x2, y1:y2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))

    target_a = targets
    target_b = targets[indices]
    return inputs_clone, target_a, target_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    """
    cutmix 보정된 CE:
    lam * CE(pred, y_a) + (1-lam)*CE(pred, y_b)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch_cutmix(teacher_model, loader, optimizer, alpha=1.0, device="cuda"):
    """
    teacher_model: forward(x, y=None)-> (dict, logit, ce_loss)
      - dict은 사용하지 않고, logit만 이용하여 crossentropy 계산
    """
    teacher_model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct, total = 0, 0

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc="[CutMix Train]")):
        x, y = x.to(device), y.to(device)

        # 1) cutmix
        x_cm, y_a, y_b, lam = cutmix_data(x, y, alpha=alpha)

        # 2) forward
        _, logits, _ = teacher_model(x_cm)  # we only need `logits` for classification
                                            # dict, ce_loss are ignored here

        # 3) loss
        loss = cutmix_criterion(criterion, logits, y_a, y_b, lam)

        # 4) backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 5) log
        bs = x.size(0)
        total_loss += loss.item() * bs

        # cutmix 정확도는 y_a 기준으로 (단순히)
        # 실제론 mix된 픽셀 때문에 정확도가 약간 가짜일 수 있으나, 통상 y_a로 모니터링
        preds = logits.argmax(dim=1)
        correct += (preds == y_a).sum().item()
        total   += bs

    epoch_loss = total_loss / total
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc

@torch.no_grad()
def eval_teacher(teacher_model, loader, device="cuda"):
    teacher_model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        _, logits, _ = teacher_model(x)  # dict, logit, ce_loss
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

def finetune_teacher_cutmix(
    teacher_model,
    train_loader,
    test_loader,
    alpha=1.0,           # CutMix alpha
    lr=1e-3,
    weight_decay=1e-4,
    epochs=10,
    device="cuda",
    ckpt_path="teacher_finetuned_cutmix.pth"
):
    """
    teacher_model: must produce (dict, logit, ce_loss) but we only need 'logit' to do classification
    train_loader, test_loader: standard classification dataset
    alpha: cutmix alpha
    lr, weight_decay, epochs, etc. for standard SGD
    """
    teacher_model = teacher_model.to(device)

    if os.path.exists(ckpt_path):
        print(f"[CutMix] Found checkpoint => load {ckpt_path}")
        teacher_model.load_state_dict(torch.load(ckpt_path))
        test_acc = eval_teacher(teacher_model, test_loader, device=device)
        print(f"[CutMix] loaded => testAcc={test_acc:.2f}")
        return teacher_model, test_acc

    optimizer = optim.SGD(teacher_model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = copy.deepcopy(teacher_model.state_dict())

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch_cutmix(
            teacher_model, train_loader,
            optimizer, alpha=alpha, device=device
        )
        te_acc = eval_teacher(teacher_model, test_loader, device=device)

        if te_acc > best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(teacher_model.state_dict())

        scheduler.step()

        print(
            f"[CutMix|ep={ep}/{epochs}] lr={scheduler.get_last_lr()[0]:.6f}, "
            f"trainAcc={tr_acc:.2f}, testAcc={te_acc:.2f}, best={best_acc:.2f}"
        )

    teacher_model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(teacher_model.state_dict(), ckpt_path)
    print(f"[CutMix] Fine-tune done => bestAcc={best_acc:.2f}, saved={ckpt_path}")
    return teacher_model, best_acc
