"""
methods/asmb.py

Core logic for Adaptive Synergy Manifold Bridging (ASMB).
 - Multi-Stage Self-Training
 - Partial Freeze (Teacher & Student)
 - MBM-based synergy manifold
"""

import torch
import copy
from modules.trainer_teacher import teacher_adaptive_update
from modules.trainer_student import student_distillation_update
from modules.partial_freeze import freeze_teacher_params, freeze_student_params
from models.mbm import ManifoldBridgingModule
# 필요시 kd_loss, logger import

def run_asmb_training(
    teacher1_wrapper,  # e.g. TeacherResNetWrapper
    teacher2_wrapper,  # e.g. TeacherEfficientNetWrapper
    student_model,
    mbm: ManifoldBridgingModule,
    synergy_head,
    trainloader1,
    trainloader2,
    testloader,
    cfg,
    logger
):
    """
    Conduct ASMB multi-stage training:
      Stage s=1..S
       (A) teacher_adaptive_update => update teacher BN/Head + MBM
       (B) student_distillation_update => update student top layers
    Args:
      teacher1_wrapper, teacher2_wrapper: teacher net wrappers (with partial freeze applied)
      student_model: partial-frozen student model
      mbm: ManifoldBridgingModule instance
      synergy_head: small head to produce synergy logit from MBM
      trainloader1, trainloader2: for teacher1 / teacher2 updates
      testloader: evaluation
      cfg: dictionary with hyperparams (teacher_lr, synergy_ce_alpha, etc.)
      logger: logging
    Returns:
      best_student_acc, best_student_state
    """

    # 0) partial freeze
    #    (이미 partial_freeze.py에서 freeze_teacher_params, freeze_student_params 썼다면, 여기서 호출 가능)
    freeze_teacher_params(teacher1_wrapper, freeze_bn=True)
    freeze_teacher_params(teacher2_wrapper, freeze_bn=True)
    freeze_student_params(student_model, freeze_bn=True)

    # 1) multi-stage loop
    num_stages = cfg.get("num_stages", 2)
    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    for stage_idx in range(1, num_stages + 1):
        logger.info(f"\n=== [ASMB] Stage {stage_idx}/{num_stages} ===")

        # (A) Teacher Adaptive Update
        teacher1_init = copy.deepcopy(teacher1_wrapper.state_dict())
        teacher2_init = copy.deepcopy(teacher2_wrapper.state_dict())

        teacher_adaptive_update(
            teacher_wrappers=[teacher1_wrapper, teacher2_wrapper],
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=student_model,    # fixed student
            trainloader=(trainloader1, trainloader2),
            cfg=cfg,
            logger=logger,
            teacher_init_state=teacher1_init,
            teacher_init_state_2=teacher2_init
        )

        # Optional: Evaluate synergy after teacher update
        synergy_acc = eval_synergy_acc(
            [teacher1_wrapper, teacher2_wrapper],
            mbm, synergy_head, testloader, cfg["device"]
        )
        logger.info(f"[Stage {stage_idx}] synergy_acc= {synergy_acc:.2f}")

        # (B) Student Distillation
        acc = student_distillation_update(
            teacher_wrappers=[teacher1_wrapper, teacher2_wrapper],
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=student_model,
            trainloader=trainloader1,  # or combine loader1+loader2
            testloader=testloader,
            cfg=cfg,
            logger=logger
        )
        logger.info(f"[Stage {stage_idx}] Student Acc= {acc:.2f}")

        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(student_model.state_dict())

    # end for stage
    logger.info(f"[ASMB] Done => best student Acc= {best_acc:.2f}")
    return best_acc, best_state


@torch.no_grad()
def eval_synergy_acc(teacher_wrappers, mbm, synergy_head, loader, device="cuda"):
    """
    Evaluate synergy manifold's accuracy:
      z_syn = synergy_head( mbm( teacher1_feat, teacher2_feat ) )
    """
    for tw in teacher_wrappers:
        tw.eval()
    mbm.eval()
    synergy_head.eval()

    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        feats = []
        for tw in teacher_wrappers:
            f, _, _ = tw(x)  # (feat, logit, ce_loss)
            feats.append(f)
        if len(feats) == 1:
            fsyn = feats[0]
        else:
            fsyn = mbm(*feats)  # assume mbm is multi-input
        zsyn = synergy_head(fsyn)
        pred = zsyn.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return 100.0 * correct / total
