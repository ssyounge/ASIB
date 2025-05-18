"""
trainer_student.py

- Student Distillation Update
- Teacher(+MBM)는 고정, Student만 학습 (CE + KD)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from modules.kd_loss import kd_loss_fn, ce_loss_fn

def student_distillation_update(
    teacher_wrappers,
    mbm, synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger
):
    """
    - Teacher/MBM 고정 -> synergy logit
    - Student => CE + KD
    """
    # 1) freeze teacher + mbm
    for tw in teacher_wrappers:
        for p in tw.parameters():
            p.requires_grad = False
    for p in mbm.parameters():
        p.requires_grad = False
    for p in synergy_head.parameters():
        p.requires_grad = False

    # 2) Student Optim
    params_s = [p for p in student_model.parameters() if p.requires_grad]
    optimizer_s = optim.SGD(params_s, lr=cfg["student_lr"], momentum=0.9, weight_decay=cfg["student_weight_decay"])
    
    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    for ep in range(cfg["student_epochs_per_stage"]):
        distill_loss_sum = 0.0
        cnt = 0
        student_model.train()

        for x, y in tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]"):
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            with torch.no_grad():
                feats = []
                for tw in teacher_wrappers:
                    f, _, _ = tw(x)
                    feats.append(f)
                if len(feats) == 1:
                    fsyn = feats[0]
                else:
                    fsyn = mbm(*feats)
                zsyn = synergy_head(fsyn)  # synergy logit

            s_out = student_model(x)
            # CE
            ce_loss_val = ce_loss_fn(s_out, y)
            # KD
            kd_loss_val = kd_loss_fn(s_out, zsyn, T=cfg.get("temperature",4.0))
            # total
            loss = cfg["ce_alpha"]*ce_loss_val + cfg["kd_alpha"]*kd_loss_val

            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()

            bs = x.size(0)
            distill_loss_sum += loss.item()*bs
            cnt += bs

        ep_loss = distill_loss_sum / cnt

        # validate student on testloader
        test_acc = eval_student(student_model, testloader, cfg["device"])

        logger.info(f"[StudentDistill ep={ep+1}] loss={ep_loss:.4f}, testAcc={test_acc:.2f}, best={best_acc:.2f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(student_model.state_dict())

    student_model.load_state_dict(best_state)
    logger.info(f"[StudentDistill] bestAcc={best_acc:.2f}")
    return best_acc

@torch.no_grad()
def eval_student(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return 100.*correct/total
