# utils/metrics.py
import time
import torch

from utils.common import count_trainable_parameters


class StageMeter:
    """Collects key metrics for each training stage."""

    def __init__(self, stage_idx: int, logger, cfg: dict, student):
        self.stg = stage_idx
        self.logger = logger
        self.cfg = cfg
        self.student = student
        self.t0 = time.time()
        self.img_seen = 0

        # FLOPs (once) : thop 없으면 0
        macs = cfg.get("_flops_cache")
        if macs is None:
            try:
                from thop import profile

                dummy = torch.zeros(1, 3, 32, 32).to(cfg["device"])
                macs, _ = profile(student, inputs=(dummy,))
            except Exception:
                macs = 0
            cfg["_flops_cache"] = macs
        self.macs = macs

    def step(self, bs: int):
        """Accumulate images processed for the current stage."""
        self.img_seen += bs

    def finish(self, best_acc: float):
        """Finalize metrics for the stage and log them."""
        wall_min = (time.time() - self.t0) / 60.0
        gpu_h = wall_min / 60.0 * torch.cuda.device_count()
        gflops = self.macs / 1e9 * self.img_seen
        param_M = count_trainable_parameters(self.student) / 1e6

        pfx = f"stage{self.stg}"
        self.logger.update_metric(f"{pfx}_acc", best_acc)
        self.logger.update_metric(f"{pfx}_wall_min", wall_min)
        self.logger.update_metric(f"{pfx}_gpu_h", gpu_h)
        self.logger.update_metric(f"{pfx}_gflops", gflops)
        self.logger.update_metric(f"{pfx}_param_M", param_M)

        self.logger.info(
            "[%s] acc=%.2f | %.1f\u202fmin | GPU-h %.2f | %.1f\u202fGFLOPs | %.1f\u202fM params",
            pfx,
            best_acc,
            wall_min,
            gpu_h,
            gflops,
            param_M,
        )

