import torch.nn.functional as F
import torch


def _resize(src, tgt):
    if src.shape[-2:] == tgt.shape[-2:]:
        return tgt
    return F.interpolate(tgt, src.shape[-2:], mode='bilinear', align_corners=False)

# ────── channel matcher —-------------------------------------------------
def _match_channels(src, tgt):
    """
    src : 기준(feature from student)  —  shape = [N, C_s, H, W]
    tgt : 대응(feature from teacher)   —  shape = [N, C_t, H, W]

    ‑ If C_t > C_s … 앞 C_s‑개 채널만 사용
    ‑ If C_t < C_s … 채널을 반복‑복제해서 C_s 로 맞춤
    """
    Cs, Ct = src.shape[1], tgt.shape[1]
    if Cs == Ct:
        return tgt
    if Ct > Cs:                     # 자려내기
        return tgt[:, :Cs]
    rep = (Cs + Ct - 1) // Ct       # 최소 반복 횟수
    return tgt.repeat(1, rep, 1, 1)[:, :Cs]


def feat_mse(student_dict, teacher_dict, ids, weights):
    loss = 0.
    for i, w in zip(ids, weights):
        s = student_dict[i]                                             # [N, C_s, H, W]
        t = _match_channels(s, _resize(s, teacher_dict[i]))             # [N, C_s, H, W]
        loss += w * F.mse_loss(s, t)
    return loss


def feat_mse_pair(student_dict, t1_dict, t2_dict, ids, weights):
    loss = 0.
    for i, w in zip(ids, weights):
        s  = student_dict[i]                                            # student
        t1 = _match_channels(s, _resize(s, t1_dict[i]))                 # teacher‑1
        t2 = _match_channels(s, _resize(s, t2_dict[i]))                 # teacher‑2
        loss += w * 0.5 * (F.mse_loss(s, t1) + F.mse_loss(s, t2))
    return loss
