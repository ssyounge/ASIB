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

from modules.kd_loss import kd_loss_fn, ce_loss_fn
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
    # freeze_student_with_adapter(current_student)  # example if you want partial freeze

    # (A) Teacher adaptive update (optional)
    teacher_adaptive_update(
        teacher_wrappers=first_teachers,
        mbm=mbm,
        synergy_head=synergy_head,
        student_model=current_student,  # student is fixed at this step
        trainloader=trainloader,
        cfg=cfg,
        logger=logger
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
        logger=logger
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
        # freeze_student_with_adapter(new_student)

        # 1) teacher adaptive update:
        teacher_wrappers = [teacherA, new_teacher]
        teacher_adaptive_update(
            teacher_wrappers=teacher_wrappers,
            mbm=mbm,
            synergy_head=synergy_head,
            student_model=new_student,
            trainloader=trainloader,
            cfg=cfg,
            logger=logger
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
            logger=logger
        )

        logger.info(f"[Continual] => new student_{idx} obtained.")
        # rename
        previous_student_as_teacher = copy.deepcopy(new_student)

    logger.info("[Continual] All stages done. final student returned.")
    return previous_student_as_teacher
