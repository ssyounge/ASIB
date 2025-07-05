import torch.nn.functional as F
import torch


def _resize(src, tgt):
    if src.shape[-2:] == tgt.shape[-2:]:
        return tgt
    return F.interpolate(tgt, src.shape[-2:], mode='bilinear', align_corners=False)


def feat_mse(student_dict, teacher_dict, ids, weights):
    loss = 0.
    for i, w in zip(ids, weights):
        s = student_dict[i]
        t = _resize(s, teacher_dict[i])
        loss += w * F.mse_loss(s, t)
    return loss


def feat_mse_pair(student_dict, t1_dict, t2_dict, ids, weights):
    loss = 0.
    for i, w in zip(ids, weights):
        s  = student_dict[i]
        t1 = _resize(s, t1_dict[i])
        t2 = _resize(s, t2_dict[i])
        loss += w * 0.5 * (F.mse_loss(s, t1) + F.mse_loss(s, t2))
    return loss
