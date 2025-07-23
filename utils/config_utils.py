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

    dataset = cfg.get("dataset", {})
    cfg.setdefault("dataset_name", dataset.get("name"))
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
    cfg.setdefault("teacher_type", teacher.get("name"))
    cfg.setdefault("teacher_pretrained", teacher.get("pretrained"))
    cfg.setdefault("teacher_lr", teacher.get("lr"))
    cfg.setdefault("teacher_weight_decay", teacher.get("weight_decay"))
    cfg.setdefault("teacher_freeze_level", teacher.get("freeze_level"))
    cfg.setdefault("teacher_freeze_bn", teacher.get("freeze_bn"))
    cfg.setdefault("teacher_freeze_ln", teacher.get("freeze_ln"))
    cfg.setdefault("teacher_use_adapter", teacher.get("use_adapter"))
    cfg.setdefault("teacher_bn_head_only", teacher.get("bn_head_only"))

    student = model.get("student", {})
    cfg.setdefault("student_type", student.get("name"))
    cfg.setdefault("student_pretrained", student.get("pretrained"))
    cfg.setdefault("student_lr", student.get("lr"))
    cfg.setdefault("student_weight_decay", student.get("weight_decay"))
    cfg.setdefault("student_freeze_level", student.get("freeze_level"))
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

    return cfg
