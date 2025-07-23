"""Hydra configuration helpers."""

from __future__ import annotations


def flatten_hydra_config(cfg: dict) -> dict:
    """Copy nested Hydra config entries to the top-level.

    Parameters
    ----------
    cfg : dict
        Configuration returned from ``OmegaConf.to_container``.

    Returns
    -------
    dict
        The same dict with additional top-level keys populated.
    """

    if "experiment" in cfg and isinstance(cfg["experiment"], dict):
        nested = flatten_hydra_config(dict(cfg["experiment"]))
        for k, v in nested.items():
            cfg.setdefault(k, v)

    dataset = cfg.get("dataset", {})
    if "dataset" in dataset and isinstance(dataset["dataset"], dict):
        dataset = dataset["dataset"]
    dataset_name = dataset.get("name")
    if dataset_name:
        cfg.setdefault("dataset_name", dataset_name)
    cfg.setdefault("data_root", dataset.get("root"))
    cfg.setdefault("small_input", dataset.get("small_input"))
    cfg.setdefault("data_aug", dataset.get("data_aug"))

    schedule = cfg.get("schedule", {})
    cfg.setdefault("lr_schedule", schedule.get("type"))
    cfg.setdefault("lr_warmup_epochs", schedule.get("lr_warmup_epochs"))
    cfg.setdefault("min_lr", schedule.get("min_lr"))
    cfg.setdefault("lr_step_size", schedule.get("lr_step_size"))
    cfg.setdefault("gamma", schedule.get("gamma"))

    model = cfg.get("model", {})
    teacher = model.get("teacher", {})
    if isinstance(teacher, dict) and "model" in teacher:
        teacher = teacher["model"]
    if isinstance(teacher, dict) and "teacher" in teacher:
        teacher = teacher["teacher"]
    cfg.setdefault("teacher_type", teacher.get("name"))
    cfg.setdefault("teacher_pretrained", teacher.get("pretrained"))
    cfg.setdefault("teacher_lr", teacher.get("lr"))
    cfg.setdefault("teacher_weight_decay", teacher.get("weight_decay"))
    teacher_fl = teacher.get("freeze_level")
    cfg.setdefault("teacher_freeze_level", -1 if teacher_fl is None else teacher_fl)
    cfg.setdefault("teacher_freeze_bn", teacher.get("freeze_bn"))
    cfg.setdefault("teacher_freeze_ln", teacher.get("freeze_ln"))
    cfg.setdefault("teacher_use_adapter", teacher.get("use_adapter"))
    cfg.setdefault("teacher_bn_head_only", teacher.get("bn_head_only"))

    student = model.get("student", {})
    if isinstance(student, dict) and "model" in student:
        student = student["model"]
    if isinstance(student, dict) and "student" in student:
        student = student["student"]
    cfg.setdefault("student_type", student.get("name"))
    cfg.setdefault("student_pretrained", student.get("pretrained"))
    cfg.setdefault("student_lr", student.get("lr"))
    cfg.setdefault("student_weight_decay", student.get("weight_decay"))
    student_fl = student.get("freeze_level")
    cfg.setdefault("student_freeze_level", -1 if student_fl is None else student_fl)
    cfg.setdefault("student_freeze_bn", student.get("freeze_bn"))
    cfg.setdefault("student_freeze_ln", student.get("freeze_ln"))
    cfg.setdefault("student_use_adapter", student.get("use_adapter"))
    cfg.setdefault("student_bn_head_only", student.get("bn_head_only"))

    wandb_cfg = cfg.get("wandb", {})
    cfg.setdefault("use_wandb", wandb_cfg.get("use"))
    cfg.setdefault("wandb_entity", wandb_cfg.get("entity"))
    cfg.setdefault("wandb_project", wandb_cfg.get("project"))
    cfg.setdefault("wandb_run_name", wandb_cfg.get("run_name"))
    cfg.setdefault("wandb_api_key", wandb_cfg.get("api_key"))


    log_cfg = cfg.get("log", {})
    cfg.setdefault("log_level", log_cfg.get("level"))
    cfg.setdefault("log_filename", log_cfg.get("filename"))

    # ------------------------------------------------------------------
    # distillation method parameters (ce_alpha, kd_alpha, etc.)
    # are nested under ``cfg['method']['method']`` in Hydra configs.
    # copy them to the top level if not already present.
    # ------------------------------------------------------------------
    method_cfg = cfg.get("method", {})
    if isinstance(method_cfg, dict) and "method" in method_cfg:
        method_cfg = method_cfg["method"]
    if isinstance(method_cfg, dict):
        for k, v in method_cfg.items():
            cfg.setdefault(k, v)

    return cfg
