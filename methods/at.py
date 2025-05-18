# methods/at.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def at_loss(teacher_feat, student_feat, p=2):
    """
    Attention Transfer loss (AT):
    - teacher_feat, student_feat: shape [N, C, H, W]
    - p=2 (Euclidean norm) or p=1
    - compute channel-wise norm => attention map => normalize => MSE
    Reference: Zagoruyko & Komodakis, "Paying More Attention to Attention," ICLR 2017
    """
    # 1) Teacher attention
    t_atten = teacher_feat.abs().pow(p).sum(dim=1)  # [N, H, W]
    t_atten = t_atten.pow(1./p)
    # 2) Student attention
    s_atten = student_feat.abs().pow(p).sum(dim=1)
    s_atten = s_atten.pow(1./p)

    # 3) Normalize
    t_norm = F.normalize(t_atten.view(t_atten.size(0), -1), dim=1)
    s_norm = F.normalize(s_atten.view(s_atten.size(0), -1), dim=1)

    # 4) MSE
    loss = F.mse_loss(s_norm, s_norm)
    return loss

class ATDistiller(nn.Module):
    """
    Example class using AT loss as a 'distiller.'
    - teacher_net, student_net: forward(x)-> (feat, logit, etc.)
    - For demonstration, we'll show how to compute at_loss in training step
    """
    def __init__(self, teacher_net, student_net, alpha=1.0, p=2):
        super().__init__()
        self.teacher = teacher_net
        self.student = student_net
        self.alpha = alpha
        self.p = p

    def forward(self, x, y=None):
        # teacher_feat, teacher_logit, ...
        with torch.no_grad():
            t_feat, t_logit, _ = self.teacher(x)

        s_feat, s_logit = self.student_forward(x)

        # AT loss
        loss_at = at_loss(t_feat, s_feat, p=self.p)

        # we can also combine with CE => total_loss
        ce_loss = F.cross_entropy(s_logit, y) if y is not None else 0.
        total_loss = ce_loss + self.alpha * loss_at

        return total_loss, s_logit

    def student_forward(self, x):
        """
        Suppose student_net returns (feat, logit) or something similar
        Adjust as needed
        """
        feat, logit, _ = self.student(x)  # or however your student returns
        return feat, logit
