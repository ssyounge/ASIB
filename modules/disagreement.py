# modules/disagreement.py

import torch

@torch.no_grad()
def compute_disagreement_rate(teacher1, teacher2, loader, device="cuda"):
    """
    Compute the *cross-error rate*, i.e. the percentage of samples on which
    ``teacher1`` and ``teacher2`` both make an incorrect prediction.

    teacher1, teacher2: nn.Module (teacher wrappers)
        Their forward(x) should return a dict containing at least the
        keys "feat_2d", "feat_4d" and "logit".
    loader: DataLoader
    device: "cuda" or "cpu"

    Returns:
        float (0~100) indicating the cross-error rate.
    """
    teacher1.eval()
    teacher2.eval()
    total_samples = 0
    both_wrong = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # forward each teacher, get their logits from the returned dict
        logit1 = teacher1(x)["logit"]
        logit2 = teacher2(x)["logit"]

        # argmax predictions
        pred1 = logit1.argmax(dim=1)
        pred2 = logit2.argmax(dim=1)

        # mask of "both teacher preds are wrong"
        wrong_mask = (pred1 != y) & (pred2 != y)
        both_wrong += wrong_mask.sum().item()
        total_samples += y.size(0)

    cross_err_rate = 100.0 * both_wrong / total_samples if total_samples > 0 else 0.0
    return cross_err_rate


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
