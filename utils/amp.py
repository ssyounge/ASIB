# utils/amp.py

from contextlib import nullcontext
import torch


def autocast_off_like(tensor):
    """Return an autocast-disabled context matching the tensor's device type.

    Works across CUDA/CPU/MPS. Falls back to nullcontext when autocast is
    unavailable for the given device.
    """
    dev = getattr(tensor, "device", None)
    dev_type = getattr(dev, "type", None) or "cuda"
    try:
        return torch.autocast(device_type=dev_type, enabled=False)
    except Exception:
        return nullcontext()


