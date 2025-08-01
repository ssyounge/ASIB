# utils/metrics.py
import time
import torch

from utils.common import count_trainable_parameters
from modules.disagreement import compute_disagreement_rate


def compute_accuracy(outputs, targets):
    """Compute classification accuracy."""
    _, predicted = torch.max(outputs.data, 1)
    total = targets.size(0)
    correct = (predicted == targets).sum().item()
    return 100.0 * correct / total


class StageMeter:
    """Collects key metrics for each training stage."""

    def __init__(self, stage_idx: int, logger, cfg: dict, student):
        self.stg = stage_idx
        self.logger = logger
        self.cfg = cfg
        self.student = student
        self.t0 = time.time()
        self.img_seen = 0

                # FLOPs (once) : 비활성화됨
        macs = 0  # FLOPs 계산 비활성화로 안정성 확보
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

        # 간단한 정확도만 표시 (상세 metrics는 제거)
        self.logger.info("[%s] acc=%.2f%%", pfx, best_acc)
        
        # Return metrics for experiment summary
        return {
            'wall_min': wall_min,
            'gpu_h': gpu_h,
            'gflops': gflops,
            'acc': best_acc
        }


class ExperimentMeter:
    """Collects total metrics across all training stages."""
    
    def __init__(self, logger, cfg: dict, student):
        self.logger = logger
        self.cfg = cfg
        self.student = student
        self.total_wall_min = 0.0
        self.total_gpu_h = 0.0
        self.total_gflops = 0.0
        self.stage_accs = []
        self.param_M = count_trainable_parameters(student) / 1e6
        self.experiment_start_time = time.time()  # 전체 실험 시작 시간
    
    def add_stage_metrics(self, wall_min: float, gpu_h: float, gflops: float, acc: float):
        """Add metrics from a completed stage."""
        self.total_wall_min += wall_min
        self.total_gpu_h += gpu_h
        self.total_gflops += gflops
        self.stage_accs.append(acc)
    
    def finish_experiment(self):
        """Log total experiment metrics."""
        # 전체 실험 시간을 실제로 계산
        total_experiment_time = (time.time() - self.experiment_start_time) / 60.0  # 분 단위
        total_experiment_gpu_h = total_experiment_time / 60.0 * torch.cuda.device_count()  # GPU-h
        
        best_acc = max(self.stage_accs) if self.stage_accs else 0.0
        final_acc = self.stage_accs[-1] if self.stage_accs else 0.0
        
        self.logger.info("=" * 80)
        self.logger.info("🎯 EXPERIMENT SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(
            "📊 Performance: Best=%.2f%% | Final=%.2f%% | Stages=%d",
            best_acc, final_acc, len(self.stage_accs)
        )
        self.logger.info(
            "⏱️  Total Time: %.1f min (%.2f GPU-h)",
            total_experiment_time, total_experiment_gpu_h
        )
        self.logger.info(
            "🧮 Total Compute: %.1f GFLOPs | Model: %.1f M params",
            self.total_gflops, self.param_M
        )
        self.logger.info("=" * 80)
        
        # 단위 설명
        self.logger.info("📝 Units: min=wall clock time, GPU-h=GPU usage hours")
        self.logger.info("         GFLOPs=total operations, M params=trainable parameters")
        self.logger.info("=" * 80)

