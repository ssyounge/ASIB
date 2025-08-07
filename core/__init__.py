# core/__init__.py

from .builder import (
    build_model,
    create_student_by_name,
    create_teacher_by_name,
    partial_freeze_teacher_auto,
    partial_freeze_student_auto,
)
from .trainer import (
    create_optimizers_and_schedulers,
    run_training_stages,
    run_continual_learning,
)
from .utils import (
    _renorm_ce_kd,
    renorm_ce_kd,
    setup_partial_freeze_schedule,
    setup_partial_freeze_schedule_with_cfg,
    setup_safety_switches,
    setup_safety_switches_with_cfg,
    auto_set_mbm_query_dim_with_model,
    cast_numeric_configs,
)

__all__ = [
    # Builder functions
    "build_model",
    "create_student_by_name",
    "create_teacher_by_name",
    "partial_freeze_teacher_auto",
    "partial_freeze_student_auto",
    
    # Trainer functions
    "create_optimizers_and_schedulers",
    "run_training_stages",
    "run_continual_learning",
    
    # Utility functions
    "renorm_ce_kd",
    "setup_partial_freeze_schedule",
    "setup_partial_freeze_schedule_with_cfg",
    "setup_safety_switches",
    "setup_safety_switches_with_cfg",
    "auto_set_mbm_query_dim_with_model",
    "cast_numeric_configs",
] 