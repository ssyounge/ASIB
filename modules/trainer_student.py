# modules/trainer_student.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from utils.progress import smart_tqdm

from modules.losses import kd_loss_fn, ce_loss_fn
from utils.misc import mixup_data, mixup_criterion
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
    teacher_reqgrad_states = []
    for tw in teacher_wrappers:
        states = []
        for p in tw.parameters():
            states.append(p.requires_grad)
            p.requires_grad = False
        teacher_reqgrad_states.append(states)

    mbm_reqgrad_states = []
    for p in mbm.parameters():
        mbm_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False

    syn_reqgrad_states = []
    for p in synergy_head.parameters():
        syn_reqgrad_states.append(p.requires_grad)
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


    # 1) student_iters 우선 => 없으면 student_epochs_per_stage
    student_epochs = cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15))
    logger.info(f"[StudentDistill] Using student_epochs={student_epochs}")

    for ep in range(student_epochs):
        distill_loss_sum = 0.0
        cnt = 0
        student_model.train()

        for x, y in smart_tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]"):
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            if cfg.get("mixup_alpha", 0.0) > 0.0:
                x_mixed, y_a, y_b, lam = mixup_data(x, y, alpha=cfg["mixup_alpha"])
            else:
                x_mixed, y_a, y_b, lam = x, y, y, 1.0

            # (A) Teacher synergy logit
            with torch.no_grad():
                # Teacher #1
                t1_dict = teacher_wrappers[0](x_mixed)
                # Teacher #2
                t2_dict = teacher_wrappers[1](x_mixed)

                f1_2d = t1_dict["feat_2d"]
                f2_2d = t2_dict["feat_2d"]
                f1_4d = t1_dict.get("feat_4d")
                f2_4d = t2_dict.get("feat_4d")

                fsyn = mbm([f1_2d, f2_2d], [f1_4d, f2_4d])
                zsyn = synergy_head(fsyn)

            # (B) Student forward
            feat_dict, s_logit, _ = student_model(x_mixed)   # (만약 student도 dict 반환하면 logit만 꺼내야 함)

            # CE + KD
            if cfg.get("mixup_alpha", 0.0) > 0.0:
                ce_obj = lambda pred, target: ce_loss_fn(
                    pred,
                    target,
                    label_smoothing=cfg.get("label_smoothing", 0.0),
                )
                ce_loss_val = mixup_criterion(ce_obj, s_logit, y_a, y_b, lam)
            else:
                ce_loss_val = ce_loss_fn(
                    s_logit,
                    y,
                    label_smoothing=cfg.get("label_smoothing", 0.0),
                )
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

    # restore original requires_grad states
    for tw, states in zip(teacher_wrappers, teacher_reqgrad_states):
        for p, rg in zip(tw.parameters(), states):
            p.requires_grad = rg
    for p, rg in zip(mbm.parameters(), mbm_reqgrad_states):
        p.requires_grad = rg
    for p, rg in zip(synergy_head.parameters(), syn_reqgrad_states):
        p.requires_grad = rg

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
