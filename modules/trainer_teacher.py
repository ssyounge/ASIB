"""
trainer_teacher.py

- Teacher Adaptive Update ( + MBM, synergy head 등)
- Student는 고정, Teacher + MBM만 학습
- L2 정규화(teacher_init_state와의 차이) + MBM 파라미터 regularization 추가
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
      (model.parameters()가 requires_grad=True 인 부분만 업데이트)
    - mbm, synergy_head도 학습
    - student_model 은 고정 (로짓 참고)
    - cfg: e.g. {
        "teacher_lr": 1e-4,
        "mbm_lr_factor": 5.0,
        "teacher_weight_decay": 3e-4,
        "reg_lambda": 1e-5,  (teacher init 대비 L2)
        "mbm_reg_lambda": 1e-4, (MBM 추가 정규화)
        "synergy_ce_alpha": 0.3, ...
      }
    - teacher_init_state, teacher_init_state_2: 
      => 사전학습된 Teacher 파라미터 dict, teacher_adaptive_update 전 단계 state 등.
    """

    # -------------------------
    # (1) 파라미터 설정
    # -------------------------
    teacher_params = []
    for tw in teacher_wrappers:
        for p in tw.parameters():
            if p.requires_grad:
                teacher_params.append(p)

    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    synergy_params = [p for p in synergy_head.parameters() if p.requires_grad]

    # Optimizer (여기서 weight_decay는 전체에 걸림. 추가로 reg_loss를 더할 수도 있음)
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

    # -------------------------
    # (2) Training loop
    # -------------------------
    for ep in range(cfg["teacher_adapt_epochs"]):
        teacher_loss_sum = 0.0
        count = 0
        for batch in tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]"):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            # Teacher forward => synergy
            feats = []
            with torch.no_grad():
                s_out = student_model(x)  # fixed student logit

            for tw in teacher_wrappers:
                f, _, _ = tw(x)  # (feat, logit, ce_loss)
                feats.append(f)

            # MBM
            if len(feats) == 1:
                fsyn = feats[0]
            else:
                fsyn = mbm(*feats)  # e.g. mbm(f1, f2)

            zsyn = synergy_head(fsyn)  # synergy logit

            # KD(teacher syn vs student)
            loss_kd = kd_loss_fn(zsyn, s_out, T=cfg.get("temperature", 4.0))

            # synergy CE
            loss_ce = ce_loss_fn(zsyn, y)
            synergy_ce_loss = cfg["synergy_ce_alpha"] * loss_ce

            # base loss
            total_loss = cfg["teacher_adapt_alpha_kd"] * loss_kd + synergy_ce_loss

            # -------------------------------------------------
            # (A) Teacher init 대비 L2 규제
            # -------------------------------------------------
            reg_loss = 0.0

            # teacher_init_state, teacher_init_state_2가 있다면, 
            # 각 Teacher param과 init 간 거리를 계산
            # (이 예시는 teacher 2명 가정)
            if teacher_init_state is not None:
                for name, param in teacher_wrappers[0].named_parameters():
                    if param.requires_grad and name in teacher_init_state:
                        p0 = teacher_init_state[name]
                        reg_loss += (param - p0).pow(2).sum()

            if teacher_init_state_2 is not None and len(teacher_wrappers) > 1:
                for name, param in teacher_wrappers[1].named_parameters():
                    if param.requires_grad and name in teacher_init_state_2:
                        p0 = teacher_init_state_2[name]
                        reg_loss += (param - p0).pow(2).sum()

            # reg_lambda
            total_loss += cfg.get("reg_lambda", 0.0) * reg_loss

            # -------------------------------------------------
            # (B) MBM + synergy_head 추가 정규화
            # -------------------------------------------------
            # 만약 weight_decay만으로 부족하고, MBM 파라미터에 추가 penalty 주고 싶다면:
            mbm_reg_loss = 0.0
            for p in mbm_params:
                mbm_reg_loss += p.pow(2).sum()
            for p in synergy_params:
                mbm_reg_loss += p.pow(2).sum()

            mbm_reg_coeff = cfg.get("mbm_reg_lambda", 0.0)
            total_loss += mbm_reg_coeff * mbm_reg_loss

            # -------------------------------------------------
            # Backprop
            # -------------------------------------------------
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bs = x.size(0)
            teacher_loss_sum += total_loss.item() * bs
            count += bs

        ep_loss = teacher_loss_sum / count

        # synergy test
        synergy_test_acc = -1
        # TODO: if you want, measure synergy on test set => synergy_test_acc

        logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")
        
        # update best
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
