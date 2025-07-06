# utils/feature_hook.py

import torch.nn as nn


class FeatHook:
    """Capture intermediate stage outputs from arbitrary backbones."""

    def __init__(self, backbone: nn.Module, layer_ids):
        self.features, self.handles = {}, []

        def _get_block(net, k):
            """Return the ``k``-th stage from various backbone layouts."""
            # ① torchvision ConvNeXt / EfficientNet
            if hasattr(net, "features"):
                return net.features[k]
            # ② ResNet 계열(net.layer1‑4) → layer1=0, layer2=1 …
            if hasattr(net, f"layer{k+1}"):
                return getattr(net, f"layer{k+1}")
            # ③ fallback: first Sequential children
            blocks = [m for m in net.children() if isinstance(m, nn.Sequential)]
            return blocks[k]

        for i in layer_ids:
            blk = _get_block(backbone, i)
            h = blk.register_forward_hook(
                lambda m, x, y, idx=i: self.features.__setitem__(idx, y.detach())
            )
            self.handles.append(h)

    def clear(self):
        self.features.clear()

    def close(self):
        [h.remove() for h in self.handles]
