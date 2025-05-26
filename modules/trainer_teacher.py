# modules/trainer_teacher.py

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from modules.kd_loss import kd_loss_fn, ce_loss_fn
from torch.optim.lr_scheduler import StepLR

@torch.no_grad()
def eval_synergy(teacher_wrappers, mbm, synergy_head, loader, device="cuda"):
    ...

def teacher_adaptive_update(
    teacher_wrappers,
    mbm, synergy_head,
    student_model,  
    trainloader,
    testloader,
    cfg,
    logger,
    teacher_init_state=None,
    teacher_init_state_2=None
):
    """
    - teacher_wrappers: [teacher1, teacher2] (requires_grad=True 부분만 학습)
    - mbm, synergy_head도 학습
    - student_model 고정 (KD용)
    - cfg:
       {
         "teacher_lr": 1e-4,
         "mbm_lr_factor": 5.0,
         "teacher_weight_decay": 3e-4,
         "teacher_step_size": 10,
         "teacher_gamma": 0.1,
         ...
       }
    """
    teacher_params = []
    for tw in teacher_wrappers:
        for p in tw.parameters():
            if p.requires_grad:
                teacher_params.append(p)
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    # Optim
    optimizer = optim.Adam([
        {"params": teacher_params, "lr": cfg["teacher_lr"]},
        {"params": mbm_params,     "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0)},
        {"params": syn_params,     "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0)},
    ], weight_decay=cfg["teacher_weight_decay"])

    # StepLR
    scheduler_t = StepLR(
        optimizer,
        step_size=cfg.get("teacher_step_size", 10),
        gamma=cfg.get("teacher_gamma", 0.1)
    )

    best_synergy = -1
    best_state = {
        "teacher_wraps": [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers],
        "mbm": copy.deepcopy(mbm.state_dict()),
        "syn_head": copy.deepcopy(synergy_head.state_dict())
    }

    for ep in range(cfg["teacher_adapt_epochs"]):
        teacher_loss_sum = 0.0
        count = 0

        for batch in tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]"):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            # (A) Student 로짓 (고정)
            with torch.no_grad():
                s_out = student_model(x)

            # (B) Teacher => feats
            feats = []
            for tw in teacher_wrappers:
                f, _, _ = tw(x)
                feats.append(f)

            # (C) MBM => synergy head
            if len(feats) == 1:
                fsyn = feats[0]
            else:
                fsyn = mbm(*feats)
            zsyn = synergy_head(fsyn)

            # (D) Loss 계산
            loss_kd = kd_loss_fn(zsyn, s_out, T=cfg.get("temperature", 4.0))
            loss_ce = ce_loss_fn(zsyn, y)
            synergy_ce_loss = cfg["synergy_ce_alpha"] * loss_ce
            total_loss = cfg["teacher_adapt_alpha_kd"] * loss_kd + synergy_ce_loss

            # L2 reg (Teacher init state)
            reg_loss = 0.0
            if teacher_init_state is not None:
                ...
            if teacher_init_state_2 is not None and len(teacher_wrappers) > 1:
                ...

            total_loss += cfg.get("reg_lambda", 0.0) * reg_loss

            # MBM & synergy_head 정규화
            mbm_reg_loss = 0.0
            for p in mbm_params:
                mbm_reg_loss += p.pow(2).sum()
            for p in syn_params:
                mbm_reg_loss += p.pow(2).sum()
            total_loss += cfg.get("mbm_reg_lambda", 0.0) * mbm_reg_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            teacher_loss_sum += total_loss.item() * x.size(0)
            count += x.size(0)

        ep_loss = teacher_loss_sum / count

        # synergy eval
        if "testloader" in cfg and cfg["testloader"] is not None:
            synergy_test_acc = eval_synergy(
                teacher_wrappers, mbm, synergy_head,
                loader=cfg["testloader"],
                device=cfg["device"]
            )
        else:
            synergy_test_acc = -1

        logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")

        # StepLR step
        scheduler_t.step()

        # best snapshot
        if synergy_test_acc > best_synergy:
            best_synergy = synergy_test_acc
            best_state["teacher_wraps"] = [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers]
            best_state["mbm"] = copy.deepcopy(mbm.state_dict())
            best_state["syn_head"] = copy.deepcopy(synergy_head.state_dict())

    # restore
    for i, tw in enumerate(teacher_wrappers):
        tw.load_state_dict(best_state["teacher_wraps"][i])
    mbm.load_state_dict(best_state["mbm"])
    synergy_head.load_state_dict(best_state["syn_head"])

    return best_synergy
