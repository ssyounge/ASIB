# utils/logging/structured.py

"""Structured logging utilities for ASIB-KD."""

import logging
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TrainingMetrics:
    """Training metrics data class."""
    epoch: int
    stage: int
    loss: float
    accuracy: float
    learning_rate: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class StructuredLogger:
    """Structured logger for training metrics."""
    
    def __init__(self, log_file: str, level: str = "INFO"):
        """Initialize structured logger."""
        self.logger = logging.getLogger("structured")
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # File handler
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics as JSON."""
        self.logger.info(json.dumps(metrics.to_dict()))
    
    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log event as JSON."""
        event = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        self.logger.info(json.dumps(event))


class PerformanceLogger:
    """Performance monitoring logger."""
    
    def __init__(self):
        """Initialize performance logger."""
        self.start_time = time.time()
        self.metrics: Dict[str, float] = {}
    
    def start_timer(self, name: str) -> None:
        """Start a timer."""
        self.metrics[f"{name}_start"] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a timer and return duration."""
        if f"{name}_start" not in self.metrics:
            raise ValueError(f"Timer '{name}' was not started")
        
        duration = time.time() - self.metrics[f"{name}_start"]
        self.metrics[f"{name}_duration"] = duration
        return duration
    
    def log_memory_usage(self) -> Dict[str, float]:
        """Log memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_metrics = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
            
            self.metrics.update(memory_metrics)
            return memory_metrics
        except ImportError:
            logging.warning("psutil not available, skipping memory logging")
            return {}
    
    def log_gpu_memory(self) -> Dict[str, float]:
        """Log GPU memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_metrics = {}
                for i in range(torch.cuda.device_count()):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024 / 1024
                    gpu_metrics[f"gpu_{i}_allocated_mb"] = memory_allocated
                    gpu_metrics[f"gpu_{i}_reserved_mb"] = memory_reserved
                
                self.metrics.update(gpu_metrics)
                return gpu_metrics
        except Exception as e:
            logging.warning(f"Failed to log GPU memory: {e}")
            return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = time.time() - self.start_time
        summary = {
            "total_time_seconds": total_time,
            "metrics": self.metrics.copy()
        }
        return summary


def setup_structured_logging(
    log_dir: str,
    experiment_name: str,
    level: str = "INFO"
) -> StructuredLogger:
    """Setup structured logging."""
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{experiment_name}_structured.log")
    return StructuredLogger(log_file, level)


def log_training_step(
    logger: StructuredLogger,
    epoch: int,
    stage: int,
    loss: float,
    accuracy: float,
    learning_rate: float
) -> None:
    """Log training step metrics."""
    metrics = TrainingMetrics(
        epoch=epoch,
        stage=stage,
        loss=loss,
        accuracy=accuracy,
        learning_rate=learning_rate,
        timestamp=datetime.now().isoformat()
    )
    logger.log_metrics(metrics)


def log_experiment_start(
    logger: StructuredLogger,
    config: Dict[str, Any]
) -> None:
    """Log experiment start."""
    logger.log_event("experiment_start", {
        "config": config,
        "timestamp": datetime.now().isoformat()
    })


def log_experiment_end(
    logger: StructuredLogger,
    final_accuracy: float,
    total_time: float
) -> None:
    """Log experiment end."""
    logger.log_event("experiment_end", {
        "final_accuracy": final_accuracy,
        "total_time_seconds": total_time,
        "timestamp": datetime.now().isoformat()
    }) 