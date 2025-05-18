"""
trainer_teacher.py

- Teacher Adaptive Update ( + MBM, synergy head 등) 
- Student는 고정, Teacher + MBM만 학습
"""

import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from modules.kd_loss import kd_loss_fn, ce_loss_fn

def teacher_adaptive_update(
    teacher_wrappers,       # list or tuple, e.g. [teacher1_wrapper, teacher2_wrapper]
    mbm, synergy_head,
    student_model,          # fixed
    trainloader,            # or (trainloader1, trainloader2), etc. 
    cfg,
    logger,
    teacher_init_state=None,
    teacher_init_state_2=None
):
    """
    - teacher_wrappers: [teacher1_wrapper, teacher2_wrapper] 
      (model.parameters()가 requires_grad=True 일부분만 존재)
    - mbm, synergy_head도 학습
    - student_model (고정) 로짓 & 레이블 참조
    - cf) cfg: e.g. teacher_lr, synergy_ce_alpha, reg_lambda etc
    """
    # 1) 파라미터 설정
    teacher_params = []
    for tw in teacher_wrappers:
        for p in tw.parameters():
            if p.requires_grad:
                teacher_params.append(p)

    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    synergy_params = [p for p in synergy_head.parameters() if p.requires_grad]

    optimizer = optim.Adam([
        {"params": teacher_params, "lr": cfg["teacher_lr"]},
        {"params": mbm_params,     "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0)},
        {"params": synergy_params, "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0)},
    ], weight_decay=cfg["teacher_weight_decay"])

    best_synergy = -1
    best_state = {
        "teacher_wraps": [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers],
        "mbm": copy.deepcopy(mbm.state_dict()),
        "syn_head": copy.deepcopy(synergy_head.state_dict())
    }

    # 2) Training loop
    for ep in range(cfg["teacher_adapt_epochs"]):
        teacher_loss_sum = 0.0
        count = 0
        for batch in tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]"):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            # forward teacher => synergy
            feats = []
            with torch.no_grad():
                s_out = student_model(x)  # fixed student
            # e.g. single teacher -> teacher_wrappers[0]
            # or multiple teachers => combine
            for tw in teacher_wrappers:
                f, _, _ = tw(x)  # (feat, logit, ce_loss)
                feats.append(f)

            # MBM
            if len(feats) == 1:
                fsyn = feats[0]
            else:
                # concat or 2-Teacher style
                import torch
                fsyn = mbm(*feats)  # e.g. mbm(f1, f2)

            zsyn = synergy_head(fsyn)  # synergy logit

            # KD(teacher syn vs student)
            loss_kd = kd_loss_fn(zsyn, s_out, T=cfg.get("temperature", 4.0))

            # synergy CE
            loss_ce = ce_loss_fn(zsyn, y)
            synergy_ce_loss = cfg["synergy_ce_alpha"] * loss_ce

            total_loss = cfg["teacher_adapt_alpha_kd"] * loss_kd + synergy_ce_loss

            # reg (teacher_init_state)
            if teacher_init_state is not None:
                # ex) L2 distance between teacher param & init
                reg_loss = 0.0
                # teacher_wrappers[0]...
                # ...
                # total_loss += cfg["reg_lambda"] * reg_loss
                pass

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bs = x.size(0)
            teacher_loss_sum += total_loss.item() * bs
            count += bs

        ep_loss = teacher_loss_sum / count

        # synergy test
        synergy_test_acc = -1
        # if testloader is not None: synergy_test_acc = ...
        logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")
        
        if synergy_test_acc > best_synergy:
            best_synergy = synergy_test_acc
            best_state["teacher_wraps"] = [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers]
            best_state["mbm"] = copy.deepcopy(mbm.state_dict())
            best_state["syn_head"] = copy.deepcopy(synergy_head.state_dict())

    # restore best
    for i, tw in enumerate(teacher_wrappers):
        tw.load_state_dict(best_state["teacher_wraps"][i])
    mbm.load_state_dict(best_state["mbm"])
    synergy_head.load_state_dict(best_state["syn_head"])

    return best_synergy
