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


# ───────── feature scale 정규화 ────────────────────
def _norm_feat(f):
    """Channel-wise L2 normalization for feature maps."""
    n, c, h, w = f.shape
    f = f.reshape(n, c, -1)
    f = F.normalize(f, dim=2)
    return f.reshape(n, c, h, w)


def feat_mse(student_dict, teacher_dict, ids, weights):
    loss = 0.0
    for i, w in zip(ids, weights):
        s = _norm_feat(student_dict[i])
        t = _norm_feat(_match_channels(s, _resize(s, teacher_dict[i])))
        loss += w * F.mse_loss(s, t)
    return loss


def feat_mse_pair(student_dict, t1_dict, t2_dict, ids, weights):
    loss = 0.0
    for i, w in zip(ids, weights):
        s = _norm_feat(student_dict[i])
        t1 = _norm_feat(_match_channels(s, _resize(s, t1_dict[i])))
        t2 = _norm_feat(_match_channels(s, _resize(s, t2_dict[i])))
        loss += w * 0.5 * (F.mse_loss(s, t1) + F.mse_loss(s, t2))
    return loss
