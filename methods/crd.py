"""
methods/crd.py

Contrastive Representation Distillation (CRD) stub code.
Reference: Tian et al., "Contrastive Representation Distillation," ICLR 2020
https://github.com/HobbitLong/RepDistiller

NOTE:
 - Real CRD uses memory bank (negative samples) or large in-batch negatives
 - This is a simplified skeleton to illustrate the core structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRDLoss(nn.Module):
    """
    A simplified version of CRD, ignoring memory bank complexities.
    Typically, we need:
      - teacher_feat: [N, D]
      - student_feat: [N, D]
      - instance indices (for positives/negatives) or memory bank

    We'll assume we do in-batch sampling for negatives as a minimal demonstration.
    """

    def __init__(self, feat_dim=128, temperature=0.07):
        super().__init__()
        self.feat_dim = feat_dim
        self.temperature = temperature

    def forward(self, student_feat, teacher_feat):
        """
        student_feat, teacher_feat: shape [N, D]
        We'll pretend the entire batch is used for negatives (in-batch).
        Real CRD also requires 'instance index' or 'memory bank' to track positives.

        Return: contrastive loss
        """
        # 1) Normalize
        s_norm = F.normalize(student_feat, dim=1)  # [N, D]
        t_norm = F.normalize(teacher_feat, dim=1)  # [N, D]

        # 2) Compute similarity
        #   sim_ij = s_norm[i]*t_norm[j]
        #   For positives, i==j, for negatives, i!=j
        sim_matrix = torch.mm(s_norm, t_norm.t())  # [N, N]

        # 3) We'll treat diag elements (i==j) as positives, off-diagonal as negatives
        #   This is a naive approach, original CRD is more subtle.
        N = student_feat.size(0)
        pos_mask = torch.eye(N, device=student_feat.device).bool()  # [N, N]
        # pos scores
        pos_score = sim_matrix[pos_mask] / self.temperature  # shape [N]
        # neg scores
        neg_score = sim_matrix[~pos_mask].view(N, N - 1) / self.temperature

        # 4) InfoNCE-like loss
        #   log( exp(pos) / [exp(pos) + sum(exp(neg)) ] ) averaged
        pos_exp = pos_score.exp()  # [N]
        neg_exp = neg_score.exp().sum(dim=1)  # [N]
        contrastive_loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-7)).mean()

        return contrastive_loss


class CRDDistiller(nn.Module):
    """
    Minimal distiller class for CRD. Typically, we want to combine:
      - CE loss on student's logit (for classification)
      - CRD loss between teacher_feat & student_feat
    """

    def __init__(self, teacher_net, student_net, crd_dim=128, alpha=0.5, temperature=0.07):
        super().__init__()
        self.teacher = teacher_net
        self.student = student_net
        self.alpha = alpha
        self.crd_loss_fn = CRDLoss(feat_dim=crd_dim, temperature=temperature)
        self.ce_loss_fn = nn.CrossEntropyLoss()

        # Possibly need a projection layer for teacher/student to match dimension crd_dim
        # self.student_proj = nn.Linear(student_feat_dim, crd_dim)
        # self.teacher_proj = nn.Linear(teacher_feat_dim, crd_dim)

    def forward(self, x, y):
        """
        x: input
        y: label

        We'll do:
         - teacher_feat, teacher_logit
         - student_feat, student_logit
         - CRD loss
         - CE loss on student's logit
         - total => alpha*CRD + (1-alpha)*CE
        """
        with torch.no_grad():
            t_feat, t_logit, _ = self.teacher(x)   # teacher forward
            # if projecting
            # t_feat = self.teacher_proj(t_feat)

        s_feat, s_logit, _ = self.student(x)
        # s_feat = self.student_proj(s_feat)

        # CRD
        crd_loss_val = self.crd_loss_fn(s_feat, t_feat)

        # CE
        ce_loss_val = self.ce_loss_fn(s_logit, y)

        total_loss = self.alpha * crd_loss_val + (1. - self.alpha) * ce_loss_val

        return total_loss, s_logit
