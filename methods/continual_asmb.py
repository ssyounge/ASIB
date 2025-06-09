"""
methods/continual_asmb.py

Demonstration of a "continual" KD scenario using ASMB:
  - Step 1: Teacher(1,2) => Student_A
  - Step 2: Student_A becomes Teacher_A => plus new Teacher3 => Student_B
  - Step 3: Student_B becomes Teacher_B => plus new Teacher4 => Student_C
  ... etc.

Partial Freeze & MBM can be reused. We rely on:
 - teacher_adaptive_update(...) from trainer_teacher.py
 - student_distillation_update(...) from trainer_student.py
"""

import copy
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from modules.losses import kd_loss_fn, ce_loss_fn
from modules.partial_freeze import freeze_teacher_params, freeze_student_with_adapter
from trainer_teacher import teacher_adaptive_update
from trainer_student import student_distillation_update

def run_asmb_continual(
    base_student,
    first_teachers,           # e.g. (teacher1, teacher2) for the first multi-teacher step
    teacher_sequence,         # list of new teachers that appear sequentially
    mbm, synergy_head,
    trainloader,
    testloader,
    cfg,
    logger
):
    """
    Continual ASMB:

    1) Use teacher1, teacher2 => distill => student_A
    2) student_A => rename to teacher_A
       plus next teacher => distill => student_B
    3) student_B => rename to teacher_B
       plus next teacher => distill => student_C
    ...
    
    Args:
      base_student: an initial Student model (nn.Module)
      first_teachers: tuple/list of length 2 => (teacher1, teacher2)
      teacher_sequence: [ T3, T4, T5, ... ] new teachers that show up in order
      mbm, synergy_head: MBM bridging module, synergy head
      trainloader, testloader: for distillation training & evaluation
      cfg: dict of hyperparams
      logger: for logging
    Returns:
      final_student: the last Student model
    """

    device = cfg["device"]
    current_student = base_student.to(device)

    # ---------------------------
    # Step 1) Multi-Teacher => Student_A
    # ---------------------------
    logger.info("[Continual] Stage 1: Multi-Teacher Distillation from teacher #1,#2 => student_A")

    # partial freeze Student, if needed
    if cfg.get("use_partial_freeze", False):
        freeze_teacher_params(
            first_teachers[0],
            teacher_name=cfg.get("teacher1_type", "resnet101"),
            freeze_bn=cfg.get("teacher1_freeze_bn", True),
            freeze_ln=cfg.get("teacher1_freeze_ln", True),
            freeze_scope=cfg.get("teacher1_freeze_scope", None),
        )
        freeze_teacher_params(
            first_teachers[1],
            teacher_name=cfg.get("teacher2_type", "efficientnet_b2"),
            freeze_bn=cfg.get("teacher2_freeze_bn", True),
            freeze_ln=cfg.get("teacher2_freeze_ln", True),
            freeze_scope=cfg.get("teacher2_freeze_scope", None),
        )
        freeze_student_with_adapter(
            current_student,
            student_name=cfg.get("student_type", "resnet_adapter"),
            freeze_bn=cfg.get("student_freeze_bn", True),
            freeze_ln=cfg.get("student_freeze_ln", True),
            freeze_scope=cfg.get("student_freeze_scope", None),
        )

    # build optimizers for first stage
    t_params = []
    for tw in first_teachers:
        for p in tw.parameters():
            if p.requires_grad:
                t_params.append(p)
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    t_opt = optim.Adam(
        [
            {"params": t_params, "lr": cfg["teacher_lr"]},
            {
                "params": mbm_params,
                "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
            },
            {
                "params": syn_params,
                "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
            },
        ],
        weight_decay=cfg["teacher_weight_decay"],
    )
    t_sched = StepLR(
        t_opt,
        step_size=cfg.get("teacher_step_size", 10),
        gamma=cfg.get("teacher_gamma", 0.1),
    )

    s_params = [p for p in current_student.parameters() if p.requires_grad]
    s_opt = optim.Adam(
        s_params,
        lr=cfg["student_lr"],
        weight_decay=cfg["student_weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    s_sched = StepLR(
        s_opt,
        step_size=cfg.get("student_step_size", 10),
        gamma=cfg.get("student_gamma", 0.1),
    )

    # (A) Teacher adaptive update (optional)
    teacher_adaptive_update(
        teacher_wrappers=first_teachers,
        mbm=mbm,
        synergy_head=synergy_head,
        student_model=current_student,  # student is fixed at this step
        trainloader=trainloader,
        cfg=cfg,
        logger=logger,
        optimizer=t_opt,
        scheduler=t_sched,
    )

    # (B) Student Distillation => actually, we can do logit-level distill 
    #    from "teacher synergy" to the Student.
    #    But if "teacher_adaptive_update" already performed the synergy update,
    #    we do a separate student update:
    student_distillation_update(
        teacher_wrappers=first_teachers,
        mbm=mbm,
        synergy_head=synergy_head,
        student_model=current_student,
        trainloader=trainloader,
        testloader=testloader,
        cfg=cfg,
        logger=logger,
        optimizer=s_opt,
        scheduler=s_sched,
    )

    student_A = copy.deepcopy(current_student)
    logger.info("[Continual] student_A created. Next, student_A => teacher_A for next step.")

    # ---------------------------
    # Step 2..N) For each new teacher in teacher_sequence
    # ---------------------------
    #  teacher_A + newTeacher => student_B
    #  teacher_B + nextTeacher => student_C
    # ...
    previous_student_as_teacher = student_A

    for idx, new_teacher in enumerate(teacher_sequence, start=2):
        logger.info(f"[Continual] Stage {idx}: oldStudent => teacher, + newTeacher => new Student")

        # rename
        teacherA = previous_student_as_teacher
        # create new blank Student
        new_student = copy.deepcopy(base_student)  # or create a new scratch model

        # partial freeze if needed
        if cfg.get("use_partial_freeze", False):
            freeze_student_with_adapter(
                new_student,
                student_name=cfg.get("student_type", "resnet_adapter"),
                freeze_bn=cfg.get("student_freeze_bn", True),
                freeze_ln=cfg.get("student_freeze_ln", True),
                freeze_scope=cfg.get("student_freeze_scope", None),
            )

        # 1) teacher adaptive update:
        teacher_wrappers = [teacherA, new_teacher]
        # build optimizers for this stage
        t_params = []
        for tw in teacher_wrappers:
            for p in tw.parameters():
                if p.requires_grad:
                    t_params.append(p)
        mbm_params = [p for p in mbm.parameters() if p.requires_grad]
        syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

        t_opt = optim.Adam(
            [
                {"params": t_params, "lr": cfg["teacher_lr"]},
                {
                    "params": mbm_params,
                    "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
                },
                {
                    "params": syn_params,
                    "lr": cfg["teacher_lr"] * cfg.get("mbm_lr_factor", 1.0),
                },
            ],
            weight_decay=cfg["teacher_weight_decay"],
        )
        t_sched = StepLR(
            t_opt,
            step_size=cfg.get("teacher_step_size", 10),
            gamma=cfg.get("teacher_gamma", 0.1),
        )

        s_params = [p for p in new_student.parameters() if p.requires_grad]
        s_opt = optim.Adam(
            s_params,
            lr=cfg["student_lr"],
            weight_decay=cfg["student_weight_decay"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        s_sched = StepLR(
            s_opt,
            step_size=cfg.get("student_step_size", 10),
            gamma=cfg.get("student_gamma", 0.1),
        )

        teacher_adaptive_update(
            teacher_wrappers=teacher_wrappers,
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=new_student,
            trainloader=trainloader,
            cfg=cfg,
            logger=logger,
            optimizer=t_opt,
            scheduler=t_sched,
        )

        # 2) student distillation
        student_distillation_update(
            teacher_wrappers=teacher_wrappers,
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=new_student,
            trainloader=trainloader,
            testloader=testloader,
            cfg=cfg,
            logger=logger,
            optimizer=s_opt,
            scheduler=s_sched,
        )

        logger.info(f"[Continual] => new student_{idx} obtained.")
        # rename
        previous_student_as_teacher = copy.deepcopy(new_student)

    logger.info("[Continual] All stages done. final student returned.")
    return previous_student_as_teacher
