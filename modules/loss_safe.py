import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

EPS = 1e-6  # log(0) 방지용 최소값


@torch.no_grad()
def _smooth_one_hot(y, n_cls: int, eps: float):
    """label smoothing"""
    smooth = torch.full((y.size(0), n_cls), eps / (n_cls - 1), device=y.device)
    smooth.scatter_(1, y.unsqueeze(1), 1.0 - eps)
    return smooth


def ce_safe(logits, target, ls_eps: float = 0.0):
    """
    * FP16 언더플로우 방지용 32-bit 계산
    * softmax-clamp 로 log(0) → nan 차단
    * 필요하면 label-smoothing 바로 사용
    """
    with autocast(False):
        logits = logits.float()
        if ls_eps > 0:
            tgt_prob = _smooth_one_hot(target, logits.size(1), ls_eps)
            log_prob = torch.log_softmax(logits, dim=1)
            return -(tgt_prob * log_prob).sum(dim=1).mean()

        prob = torch.softmax(logits, dim=1).clamp(min=EPS)
        return F.nll_loss(prob.log(), target, reduction="mean")


def kl_safe(p_logits, q_logits, tau: float = 1.0):
    with autocast(False):
        p = torch.softmax(p_logits.float() / tau, dim=1).clamp(EPS, 1.0)
        q = torch.softmax(q_logits.float() / tau, dim=1).clamp(EPS, 1.0)
        return F.kl_div(p.log(), q, reduction="batchmean") * (tau ** 2)
