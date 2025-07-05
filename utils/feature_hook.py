import torch.nn as nn

class FeatHook:
    """hook.features[idx] → tensor(detached)"""
    def __init__(self, backbone: nn.Module, layer_ids):
        self.features, self.handles = {}, []
        for i in layer_ids:
            h = backbone.features[i].register_forward_hook(
                lambda m, x, y, idx=i: self.features.__setitem__(idx, y.detach())
            )
            self.handles.append(h)

    def clear(self):
        self.features.clear()

    def close(self):
        for h in self.handles:
            h.remove()
