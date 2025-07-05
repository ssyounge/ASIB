import torch.nn.functional as F
import torch


def _resize(src, tgt):
    if src.shape[-2:] == tgt.shape[-2:]:
        return tgt
    return F.interpolate(tgt, src.shape[-2:], mode='bilinear', align_corners=False)

# ───────────────── channel matcher ─────────────────
def _match_channels(src, tgt):
    """Return *tgt* whose channel-dim matches *src* (simple crop / repeat)."""
    Cs, Ct = src.shape[1], tgt.shape[1]
    if Cs == Ct:
        return tgt
    if Ct > Cs:                           # teacher 채널이 더 크면 앞부분만 사용
        return tgt[:, :Cs]
    # teacher 채널이 더 작으면 반복-복제해서 맞춤
    rep = (Cs + Ct - 1) // Ct             # 최소 반복 횟수
    return tgt.repeat(1, rep, 1, 1)[:, :Cs]


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
        t1 = _match_channels(s, _resize(s, t1_dict[i]))
        t2 = _match_channels(s, _resize(s, t2_dict[i]))
        loss += w * 0.5 * (F.mse_loss(s, t1) + F.mse_loss(s, t2))
    return loss
