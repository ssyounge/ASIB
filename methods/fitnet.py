"""
methods/fitnet.py

FitNet: "Hints for Thin Deep Nets" (Romero et al., 2015)
 - Uses an intermediate layer of Teacher as a "hint"
 - Student intermediate layer as "guided layer"
 - Minimizes L2 distance between these two feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FitNetLoss(nn.Module):
    """
    FitNet's hint-based loss between teacher_feat (hint) and student_feat (guided).
    Typically MSE.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        # or we can store something else if needed

    def forward(self, teacher_hint, student_guided):
        """
        teacher_hint, student_guided: shape [N, C, H, W] (if 2D feature map)
        or [N, D] if flattened. We do MSE on them.
        We'll assume they're the same shape already, or we do a conv1x1 to match shapes.
        """
        # If needed, we might do pooling or dimension matching, but let's keep it simple.
        loss_hint = F.mse_loss(student_guided, teacher_hint)
        return self.alpha * loss_hint


class FitNetDistiller(nn.Module):
    """
    Minimal Distiller class for FitNet:
      - We pick a teacher "hint layer" and a student "guided layer"
      - Minimize MSE + Student's CE (or KD) for classification
    """
    def __init__(self, teacher_net, student_net, hint_alpha=1.0, ce_alpha=1.0):
        super().__init__()
        self.teacher = teacher_net
        self.student = student_net
        self.hint_loss_fn = FitNetLoss(alpha=hint_alpha)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.ce_alpha = ce_alpha
        # optionally, we can store "hint_layer" indices

    def forward(self, x, y=None):
        """
        1) teacher forward => teacher_feat_hint
        2) student forward => student_feat_guided, student_logit
        3) L2 loss (hint vs guided)
        4) CE loss => total
        """

        with torch.no_grad():
            # We get teacher's hint feature + teacher's final logit if needed
            # Let's assume teacher(...) returns (hint_feat, final_logit) or something similar
            t_hint, t_logit, _ = self.teacher(x)

        # Student
        s_guided, s_logit, _ = self.student(x)

        # FitNet hint loss
        hint_loss_val = self.hint_loss_fn(t_hint, s_guided)

        # CE (classification) loss
        ce_loss_val = 0.0
        if y is not None:
            ce_loss_val = self.ce_loss_fn(s_logit, y)

        total_loss = hint_loss_val + self.ce_alpha * ce_loss_val
        return total_loss, s_logit
