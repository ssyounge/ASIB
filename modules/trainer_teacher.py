# modules/trainer_teacher.py

import torch
import torch.nn as nn
import copy
from utils.progress import smart_tqdm

from modules.losses import kd_loss_fn, ce_loss_fn

def _cpu_state_dict(module: torch.nn.Module):
    """
    주어진 nn.Module의 state_dict() 값을 **CPU Tensor**로 복사해 반환한다.
    GPU 메모리 사용량을 줄이기 위해 스냅샷을 RAM에 저장할 때 사용.
    """
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

@torch.no_grad()
def eval_synergy(teacher_wrappers, mbm, synergy_head, loader, device="cuda"):
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        t1_dict = teacher_wrappers[0](x)
        t2_dict = teacher_wrappers[1](x)

        f1_2d = t1_dict["feat_2d"]
        f2_2d = t2_dict["feat_2d"]
        f1_4d = t1_dict.get("feat_4d")
        f2_4d = t2_dict.get("feat_4d")

        fsyn = mbm([f1_2d, f2_2d], [f1_4d, f2_4d])
        zsyn = synergy_head(fsyn)

        pred = zsyn.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

    acc = 100.0 * correct / total if total>0 else 0
    return acc

def teacher_adaptive_update(
    teacher_wrappers,
    mbm, synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler=None,
):
    """
    - teacher_wrappers: [teacher1, teacher2]
    - mbm, synergy_head: partial freeze 포함
    - student_model: 고정 (KD용)
    - testloader: (optional) evaluation loader for synergy accuracy
    """
    teacher_params = []
    for tw in teacher_wrappers:
        for p in tw.parameters():
            if p.requires_grad:
                teacher_params.append(p)
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]


    best_synergy = -1
    best_state = {
        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
        "mbm":  _cpu_state_dict(mbm),
        "syn_head": _cpu_state_dict(synergy_head)
    }



    # 1) teacher_iters 우선 => 없으면 teacher_adapt_epochs
    teacher_epochs = cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5))
    logger.info(f"[TeacherAdaptive] Using teacher_epochs={teacher_epochs}")

    for ep in range(teacher_epochs):
        teacher_loss_sum = 0.0
        count = 0

        for batch in smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]"):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            # (A) Student logit (고정)
            with torch.no_grad():
                _, s_logit, _ = student_model(x)

            # (B) Teacher features
            feats_2d = []
            feats_4d = []
            for tw in teacher_wrappers:
                t_dict = tw(x)
                feats_2d.append(t_dict["feat_2d"])
                feats_4d.append(t_dict.get("feat_4d"))

            # (C) MBM + synergy_head
            fsyn = mbm(feats_2d, feats_4d)
            zsyn = synergy_head(fsyn)

            # (D) loss 계산 (KL + synergyCE)
            loss_kd         = kd_loss_fn(zsyn, s_logit, T=cfg.get("temperature", 4.0))
            loss_ce         = ce_loss_fn(
                zsyn,
                y,
                label_smoothing=cfg.get("label_smoothing", 0.0),
            )
            synergy_ce_loss = cfg["synergy_ce_alpha"] * loss_ce

            # 기본 KD+CE
            total_loss = cfg["teacher_adapt_alpha_kd"] * loss_kd + synergy_ce_loss

            # ── 1) Teacher 파라미터 L2 정규화 ───────────────────────────
            reg_loss = torch.tensor(0.0, device=cfg["device"])
            for p in teacher_params:                 # 위에서 이미 모아 둠
                if p.requires_grad:
                    reg_loss = reg_loss + p.pow(2).sum()
            total_loss = total_loss + float(cfg.get("reg_lambda", 0.0)) * reg_loss
            # -----------------------------------------------------------

            # ── 2) MBM + Synergy-Head L2 정규화 ─────────────────────────
            mbm_reg_loss = torch.tensor(0.0, device=cfg["device"])
            for p in mbm_params + syn_params:
                if p.requires_grad:
                    mbm_reg_loss = mbm_reg_loss + p.pow(2).sum()
            total_loss = total_loss + float(cfg.get("mbm_reg_lambda", 0.0)) * mbm_reg_loss
            # -----------------------------------------------------------      
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            teacher_loss_sum += total_loss.item() * x.size(0)
            count += x.size(0)

        ep_loss = teacher_loss_sum / count

        # synergy_eval
        if testloader is not None:
            synergy_test_acc = eval_synergy(
                teacher_wrappers,
                mbm,
                synergy_head,
                loader=testloader,
                device=cfg["device"],
            )
        else:
            synergy_test_acc = -1

        logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"teacher_ep{ep+1}_loss", ep_loss)
        logger.update_metric(f"teacher_ep{ep+1}_synAcc", synergy_test_acc)

        if scheduler is not None:
            scheduler.step()

        # best snapshot
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
