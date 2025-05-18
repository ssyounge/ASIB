"""
methods/vanilla_kd.py

Vanilla KD (Hinton et al., 2015):
 - teacher_logits vs student_logits => KL divergence w/ temperature T
 - Also student does CE w.r.t ground truth label
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaKD(nn.Module):
    """
    Minimal distiller for "vanilla" KD:
      total_loss = alpha * CE(student, label) + (1-alpha) * KL(stu, tea) * T^2
    """
    def __init__(self, teacher_net, student_net, alpha=0.5, temperature=4.0):
        super().__init__()
        self.teacher = teacher_net
        self.student = student_net
        self.alpha = alpha
        self.T = temperature
        self.ce_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, y):
        """
        x: input
        y: label
        1) teacher forward => teacher_logit
        2) student forward => student_logit
        3) CE + KD => total_loss
        Return: total_loss, student_logit
        """

        with torch.no_grad():
            # teacher_logit: shape [N, num_classes]
            # if teacher returns (feat, logit), just use logit
            _, t_logit, _ = self.teacher(x)

        # student
        _, s_logit, _ = self.student(x)

        # 1) CE
        ce_loss = self.ce_loss_fn(s_logit, y)

        # 2) KD: KL-div with temperature T
        #   kd_loss = KL( softmax(stu/T), softmax(tea/T) ) * T^2
        kd_loss = self.kd_loss_fn(s_logit, t_logit, self.T)

        # total
        total_loss = self.alpha*ce_loss + (1.-self.alpha)*kd_loss

        return total_loss, s_logit

    def kd_loss_fn(self, stu_logit, tea_logit, T):
        """
        compute KL divergence between student & teacher,
        with softmax temperature T
        kd_loss = KL( softmax(stu/T), softmax(tea/T) ) * (T^2)
        """
        stu_log_probs = F.log_softmax(stu_logit / T, dim=1)
        tea_probs     = F.softmax(tea_logit / T, dim=1)
        kl_div = F.kl_div(stu_log_probs, tea_probs, reduction='batchmean') * (T*T)
        return kl_div
