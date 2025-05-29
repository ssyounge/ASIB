# modules/trainer_student.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from modules.losses import kd_loss_fn, ce_loss_fn
from torch.optim.lr_scheduler import StepLR

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
    optimizer_s = optim.Adam(
        params_s,
        lr=cfg["student_lr"],
        weight_decay=cfg["student_weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )

    scheduler_s = StepLR(
        optimizer_s,
        step_size=cfg.get("student_step_size", 10),
        gamma=cfg.get("student_gamma", 0.1)
    )
    
    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    # 여기도 feat_key 가져옴
    feat_key = cfg.get("feat_key", "feat_2d")

    # 1) student_iters 우선 => 없으면 student_epochs_per_stage
    student_epochs = cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15))
    logger.info(f"[StudentDistill] Using student_epochs={student_epochs}")

    for ep in range(student_epochs):
        distill_loss_sum = 0.0
        cnt = 0
        student_model.train()

        for x, y in tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]"):
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            # (A) Teacher synergy logit
            with torch.no_grad():
                # Teacher #1
                t1_dict, _, _ = teacher_wrappers[0](x)
                f1 = t1_dict[feat_key]  # 2D or 4D
                # Teacher #2
                t2_dict, _, _ = teacher_wrappers[1](x)
                f2 = t2_dict[feat_key]

                # MBM => 텐서 반환
                fsyn = mbm(f1, f2)  # returns a tensor (2D or 4D)

                # synergy_head가 2D를 기대한다고 가정
                if fsyn.dim() == 4:
                    # global pooling => 2D
                    fsyn_2d = torch.nn.functional.adaptive_avg_pool2d(fsyn, (1,1)).flatten(1)
                    zsyn = synergy_head(fsyn_2d)  # (N, num_classes)
                else:
                    # 2D 바로
                    zsyn = synergy_head(fsyn)

            # (B) Student forward
            feat_dict, s_logit, _ = student_model(x)   # (만약 student도 dict 반환하면 logit만 꺼내야 함)

            # CE + KD
            ce_loss_val = ce_loss_fn(s_logit, y)
            kd_loss_val = kd_loss_fn(s_logit, zsyn, T=cfg.get("temperature", 4.0))
            loss = cfg["ce_alpha"] * ce_loss_val + cfg["kd_alpha"] * kd_loss_val

            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()

            bs = x.size(0)
            distill_loss_sum += loss.item()*bs
            cnt += bs

        ep_loss = distill_loss_sum / cnt

        # (C) validate
        test_acc = eval_student(student_model, testloader, cfg["device"])

        logger.info(f"[StudentDistill ep={ep+1}] loss={ep_loss:.4f}, testAcc={test_acc:.2f}, best={best_acc:.2f}")

        scheduler_s.step()

        # (E) best snapshot
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
        feat_dict, s_logit, _ = model(x)
        pred = s_logit.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return 100.*correct/total
