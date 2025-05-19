import torch
import torch.nn as nn
import torch.optim as optim
import copy
from tqdm import tqdm

from modules.kd_loss import kd_loss_fn, ce_loss_fn

@torch.no_grad()
def eval_synergy(teacher_wrappers, mbm, synergy_head, loader, device="cuda"):
    """
    Evaluate synergy logit => top-1 accuracy on a given loader.
    synergy = synergy_head( mbm(teacher1_feat, teacher2_feat) )
    """
    # eval mode
    for tw in teacher_wrappers:
        tw.eval()
    mbm.eval()
    synergy_head.eval()

    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 1) Teacher feats
        feats = []
        for tw in teacher_wrappers:
            # teacher_wrappers[i] returns (feat, logit, ce_loss)
            f, _, _ = tw(x)
            feats.append(f)

        # 2) MBM => synergy head
        if len(feats) == 1:
            fsyn = feats[0]
        else:
            fsyn = mbm(*feats)

        zsyn = synergy_head(fsyn)  # synergy logit => shape [N, #classes]
        pred = zsyn.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

    # 다시 train 모드로 복귀
    for tw in teacher_wrappers:
        tw.train()
    mbm.train()
    synergy_head.train()

    return 100.0 * correct / total


def teacher_adaptive_update(
    teacher_wrappers,       # list or tuple, e.g. [teacher1_wrapper, teacher2_wrapper]
    mbm, synergy_head,
    student_model,          # 고정 (student 로짓은 KD용으로만 사용)
    trainloader,            # Teacher를 학습할 때 쓰는 train loader
    testloader,             # [NEW] synergy 정확도 평가용 loader
    cfg,
    logger,
    teacher_init_state=None,
    teacher_init_state_2=None
):
    """
    - teacher_wrappers: [teacher1, teacher2] 형태 (requires_grad=True인 부분만 업데이트)
    - mbm, synergy_head도 학습
    - student_model은 고정 (KD를 위해 Student logit만 참고)
    - cfg 예시:
      {
        "teacher_lr": 1e-4,
        "mbm_lr_factor": 5.0,
        "teacher_weight_decay": 3e-4,
        "reg_lambda": 1e-5,   # teacher init 대비 L2
        "mbm_reg_lambda": 1e-4, # MBM 추가 정규화
        "synergy_ce_alpha": 0.3,
        "teacher_adapt_alpha_kd": 0.2,
        "temperature": 4.0,
        ...
      }
    - teacher_init_state, teacher_init_state_2: teacher 파라미터 초기 상태 (L2 reg 목적)
    """

    # -------------------------
    # (1) 학습할 파라미터 묶기
    # -------------------------
    teacher_params = []
    for tw in teacher_wrappers:
        for p in tw.parameters():
            if p.requires_grad:
                teacher_params.append(p)

    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    # 옵티마이저
    optimizer = optim.Adam([
        {"params": teacher_params, "lr": cfg["teacher_lr"]},
        {"params": mbm_params,     "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0)},
        {"params": syn_params,     "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0)},
    ], weight_decay=cfg["teacher_weight_decay"])

    # -------------------------
    # (2) best snapshot init
    # -------------------------
    best_synergy = -1
    best_state = {
        "teacher_wraps": [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers],
        "mbm": copy.deepcopy(mbm.state_dict()),
        "syn_head": copy.deepcopy(synergy_head.state_dict())
    }

    # -------------------------
    # (3) 메인 학습 루프
    # -------------------------
    for ep in range(cfg["teacher_adapt_epochs"]):
        teacher_loss_sum = 0.0
        count = 0
        for batch in tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]"):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            # (A) Student 로짓은 고정
            with torch.no_grad():
                s_out = student_model(x)

            # (B) Teacher => feats
            feats = []
            for tw in teacher_wrappers:
                f, _, _ = tw(x)  # (feat, logit, ce)
                feats.append(f)

            # (C) MBM => synergy logit
            if len(feats) == 1:
                fsyn = feats[0]
            else:
                fsyn = mbm(*feats)  
            zsyn = synergy_head(fsyn)  

            # (D) Loss 계산
            #  - KD (teacher syn vs student)
            loss_kd = kd_loss_fn(zsyn, s_out, T=cfg.get("temperature", 4.0))
            #  - Synergy CE
            loss_ce = ce_loss_fn(zsyn, y)
            synergy_ce_loss = cfg["synergy_ce_alpha"] * loss_ce
            #  - base
            total_loss = cfg["teacher_adapt_alpha_kd"] * loss_kd + synergy_ce_loss

            # (E) Teacher init L2 reg
            reg_loss = 0.0
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

            total_loss += cfg.get("reg_lambda", 0.0) * reg_loss

            # (F) MBM + synergy_head 추가 정규화
            mbm_reg_loss = 0.0
            for p in mbm_params:
                mbm_reg_loss += p.pow(2).sum()
            for p in syn_params:
                mbm_reg_loss += p.pow(2).sum()
            total_loss += cfg.get("mbm_reg_lambda", 0.0) * mbm_reg_loss

            # (G) Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            teacher_loss_sum += total_loss.item() * x.size(0)
            count += x.size(0)

        ep_loss = teacher_loss_sum / count

        # -------------------------
        # (4) synergy eval
        # -------------------------
        # testloader가 None이면 생략
        if "testloader" in cfg and cfg["testloader"] is not None:
            synergy_test_acc = eval_synergy(
                teacher_wrappers, mbm, synergy_head,
                loader=cfg["testloader"],
                device=cfg["device"]
            )
        else:
            synergy_test_acc = -1

        logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")

        # -------------------------
        # (5) best model update
        # -------------------------
        if synergy_test_acc > best_synergy:
            best_synergy = synergy_test_acc
            best_state["teacher_wraps"] = [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers]
            best_state["mbm"] = copy.deepcopy(mbm.state_dict())
            best_state["syn_head"] = copy.deepcopy(synergy_head.state_dict())

    # -------------------------
    # (6) best restore
    # -------------------------
    for i, tw in enumerate(teacher_wrappers):
        tw.load_state_dict(best_state["teacher_wraps"][i])
    mbm.load_state_dict(best_state["mbm"])
    synergy_head.load_state_dict(best_state["syn_head"])

    return best_synergy
