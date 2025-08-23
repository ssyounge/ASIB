import copy
import torch
import torch.nn as nn


class EMATeacher(nn.Module):
    def __init__(self, student_wrapper: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = float(decay)
        # Copy the student wrapper to preserve forward format and heads
        self.model = copy.deepcopy(student_wrapper).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, student_wrapper: nn.Module):
        ms, mt = student_wrapper.state_dict(), self.model.state_dict()
        for k in mt.keys():
            if k in ms:
                mt[k].copy_(self.decay * mt[k] + (1.0 - self.decay) * ms[k])

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)


