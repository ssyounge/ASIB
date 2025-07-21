# utils/misc.py

import os
import torch
import random
import numpy as np
import inspect
import logging

# ----------------------------------------------------------------------------
def sanitize_cfg(cfg: dict) -> dict:
    """Return a JSON-serializable copy of ``cfg``."""
    cfg_clean = {}
    for k, v in cfg.items():
        if isinstance(v, logging.Logger):
            cfg_clean[k] = str(v)
        else:
            cfg_clean[k] = v
    return cfg_clean

# ── NEW: backwards-compatible torch.load ──────────────────────
if "weights_only" not in inspect.signature(torch.load).parameters:
    _orig_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load
    print(
        "[utils.misc]  ⚠  PyTorch <2.1 detected → 'weights_only' patched for torch.load()."
    )
# ───────────────────────────────────────────────────────────────

def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed to use for ``random``, ``numpy`` and ``torch``.
    deterministic : bool, optional
        If ``True`` and CUDA is available, sets deterministic flags for
        ``cudnn`` backends. This also disables the benchmark mode for
        convolution layers. When using deterministic cuBLAS operations,
        PyTorch requires the ``CUBLAS_WORKSPACE_CONFIG`` environment
        variable to be set (e.g. to ``":16:8"`` or ``":4096:8"``) before
        importing ``torch``.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth"):
    """
    Save model & optimizer state_dict, plus metadata (epoch).
    """
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict()
    }
    torch.save(ckpt, save_path)

def load_checkpoint(model, optimizer, load_path):
    """
    Load model & optimizer from checkpoint file.
    """
    if not os.path.exists(load_path):
        print(f"[Warning] No checkpoint found at {load_path}")
        return 0  # or -1, indicating failure

    ckpt = torch.load(load_path, weights_only=True)
    model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optim_state"])
    start_epoch = ckpt["epoch"]
    return start_epoch

def cutmix_data(inputs, targets, alpha=1.0):
    """Apply CutMix augmentation.

    ``inputs`` and ``targets`` are NCHW tensors, i.e. ``[N, C, H, W]`` for
    ``inputs`` and ``[N]`` for ``targets``.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(N, C, H, W)``.
    targets : Tensor
        Labels tensor of shape ``(N,)``.
    alpha : float, optional
        Beta distribution parameter. ``alpha <= 0`` disables CutMix.
    """
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    batch_size = inputs.size(0)
    indices = torch.randperm(batch_size, device=inputs.device)
    lam = random.betavariate(alpha, alpha)

    H, W = inputs.size(2), inputs.size(3)
    cut_w = int(W * (1 - lam))
    cut_h = int(H * (1 - lam))
    cx = random.randint(0, W - 1)
    cy = random.randint(0, H - 1)

    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y2 = min(cy + cut_h // 2, H)

    inputs_clone = inputs.clone()
    inputs_clone[:, :, y1:y2, x1:x2] = inputs[indices, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    target_a = targets
    target_b = targets[indices]
    return inputs_clone, target_a, target_b, lam


def mixup_data(inputs, targets, alpha=1.0):
    """Apply MixUp augmentation.

    Parameters
    ----------
    inputs : Tensor
        Input tensor of shape ``(N, C, H, W)``.
    targets : Tensor
        Ground-truth labels of shape ``(N,)``.
    alpha : float, optional
        MixUp beta distribution parameter. ``alpha <= 0`` disables MixUp.

    Returns
    -------
    Tuple[Tensor, Tensor, Tensor, float]
        Mixed inputs, first set of labels, second set of labels and mix
        coefficient ``lam``.
    """
    if alpha <= 0.0:
        return inputs, targets, targets, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)

    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute the criterion for mixed targets."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def get_amp_components(cfg):
    """Return autocast context and GradScaler based on config."""
    use_amp = bool(cfg.get("use_amp", False))
    device = cfg.get("device", "cuda")
    if not use_amp:
        from contextlib import nullcontext
        return nullcontext(), None
    if device != "cuda" or not torch.cuda.is_available():
        from contextlib import nullcontext
        return nullcontext(), None
    amp_dtype = cfg.get("amp_dtype", "float16")
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    autocast_ctx = torch.autocast("cuda", dtype=dtype)
    scaler = torch.amp.GradScaler(
        "cuda", init_scale=int(cfg.get("grad_scaler_init_scale", 1024))
    )
    return autocast_ctx, scaler


def check_label_range(dataset, num_classes: int) -> None:
    """Validate that dataset labels are within ``[0, num_classes - 1]``.

    Parameters
    ----------
    dataset : Dataset
        Dataset object (must expose ``targets`` or ``labels`` attribute).
    num_classes : int
        Expected number of classes.

    Raises
    ------
    ValueError
        If any label falls outside the valid range.
    """
    labels = None
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):
        labels = dataset.labels

    if labels is None:
        return

    label_tensor = torch.as_tensor(labels)
    min_label = int(label_tensor.min())
    max_label = int(label_tensor.max())
    if min_label < 0 or max_label >= num_classes:
        raise ValueError(
            f"Dataset labels must be within [0, {num_classes - 1}], "
            f"got min={min_label}, max={max_label}"
        )


def get_model_num_classes(model):
    """Return the classifier output dimension for a variety of models."""
    module = getattr(model, "backbone", model)
    if hasattr(module, "fc"):
        return module.fc.out_features
    if hasattr(module, "classifier"):
        cls = module.classifier
        if isinstance(cls, torch.nn.Linear):
            return cls.out_features
        if isinstance(cls, torch.nn.Sequential):
            for layer in reversed(cls):
                if isinstance(layer, torch.nn.Linear):
                    return layer.out_features
    if hasattr(module, "head"):
        return module.head.out_features
    raise AttributeError("Unable to infer num_classes from model")
