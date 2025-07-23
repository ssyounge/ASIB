# modules/disagreement.py

import torch
import logging  # DEBUG 로그용
from utils.misc import get_amp_components

@torch.no_grad()
def compute_disagreement_rate(
    teacher1,
    teacher2,
    loader,
    device="cuda",
    cfg=None,
    mode: str = "both_wrong",
):
    """Return the disagreement rate between two teachers.

    The ``mode`` argument controls how disagreement is determined:

    - ``"pred"`` uses a simple prediction mismatch, i.e. ``pred1 != pred2``.
    - ``"both_wrong"`` (default) measures *cross-error*, counting samples on
      which **both** teachers are incorrect.

    teacher1, teacher2: nn.Module (teacher wrappers)
        Their forward(x) should return a dict containing at least the
        keys "feat_2d", "feat_4d" and "logit".
    loader: DataLoader
    device: "cuda" or "cpu"

    Parameters
    ----------
    teacher1, teacher2 : nn.Module
        Teacher wrappers. ``forward(x)`` must return a dict with ``"logit"``.
    loader : DataLoader
        Loader yielding ``(images, labels)``.
    device : str
        Device to run on.
    cfg : dict, optional
        Config used for AMP detection.
    mode : str, optional
        ``"pred"`` or ``"both_wrong"`` as described above.

    Returns
    -------
    float
        Disagreement rate in ``[0, 100]``.
    """
    # record current training modes to restore them later
    teacher1_train_state = teacher1.training
    teacher2_train_state = teacher2.training

    teacher1.eval()
    teacher2.eval()
    total_samples = 0
    disagree_count = 0

    autocast_ctx, _ = get_amp_components(cfg or {})
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with autocast_ctx:
            logit1 = teacher1(x)["logit"]
            logit2 = teacher2(x)["logit"]

        # argmax predictions
        pred1 = logit1.argmax(dim=1)
        pred2 = logit2.argmax(dim=1)

        correct1 = pred1.eq(y)
        correct2 = pred2.eq(y)

        # ---- 마스크 결정 -------------------------------------------------
        if mode == "none":  # (옵션) 항상 OFF
            disagree_mask = torch.zeros_like(correct1, dtype=torch.bool)
        elif mode == "pred":
            disagree_mask = pred1 != pred2
        elif mode == "both_wrong":
            disagree_mask = ~correct1 & ~correct2
        elif mode == "any_wrong":
            disagree_mask = ~(correct1 & correct2)
        else:
            raise ValueError(f"Unknown disagree_mode: {mode}")

        disagree_count += disagree_mask.sum().item()
        total_samples += y.size(0)

    dis_rate = 100.0 * disagree_count / total_samples if total_samples > 0 else 0.0

    # ---- DEBUG 로그 --------------------------------------------------
    lvl = (cfg.get("log_level") if cfg is not None else None) or "INFO"
    if cfg is not None and lvl.upper() == "DEBUG":
        logging.debug(
            "[DisagreeDBG] mode=%s | selected=%d / %d (%.1f %%)",
            mode,
            disagree_count,
            total_samples,
            100.0 * disagree_count / total_samples if total_samples > 0 else 0.0,
        )

    # restore original training modes
    teacher1.train(teacher1_train_state)
    teacher2.train(teacher2_train_state)

    return dis_rate


@torch.no_grad()
def sample_weights_from_disagreement(
    logit1: torch.Tensor,
    logit2: torch.Tensor,
    labels: torch.Tensor,
    mode: str = "pred",
    lambda_high: float = 1.0,
    lambda_low: float = 1.0,
) -> torch.Tensor:
    """Return per-sample weights based on teacher disagreement.

    Parameters
    ----------
    logit1 : Tensor
        Teacher #1 logits of shape ``(N, C)``.
    logit2 : Tensor
        Teacher #2 logits of shape ``(N, C)``.
    labels : Tensor
        Ground truth labels of shape ``(N,)``.
    mode : str, optional
        ``"pred"`` to assign ``lambda_high`` when teacher predictions differ.
        ``"both_wrong"`` to assign ``lambda_high`` when both teachers are
        incorrect on the sample. Defaults to ``"pred"``.
    lambda_high : float, optional
        Weight for samples that satisfy the condition.
    lambda_low : float, optional
        Weight for the remaining samples.

    Returns
    -------
    Tensor
        1D tensor of shape ``(N,)`` containing non-negative sample weights.
    """

    pred1 = logit1.argmax(dim=1)
    pred2 = logit2.argmax(dim=1)

    if mode == "pred":
        mask = pred1 != pred2
    elif mode == "both_wrong":
        mask = (pred1 != labels) & (pred2 != labels)
    else:
        raise ValueError(f"Unknown disagree_mode: {mode}")

    high = torch.tensor(lambda_high, dtype=torch.float32, device=logit1.device)
    low = torch.tensor(lambda_low, dtype=torch.float32, device=logit1.device)
    weights = torch.where(mask, high, low)
    return weights
