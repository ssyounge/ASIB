# modules/disagreement.py

import torch

@torch.no_grad()
def compute_disagreement_rate(teacher1, teacher2, loader, device="cuda"):
    """
    Compute the *cross-error rate*, i.e. the percentage of samples on which
    ``teacher1`` and ``teacher2`` both make an incorrect prediction.

    teacher1, teacher2: nn.Module (teacher wrappers)
        Their forward(x) should return (feat, logit, loss) or similar.
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

        # forward each teacher, get their logits
        # assume forward(...) => (feat, logit, ce_loss)
        _, logit1, _ = teacher1(x)
        _, logit2, _ = teacher2(x)

        # argmax predictions
        pred1 = logit1.argmax(dim=1)
        pred2 = logit2.argmax(dim=1)

        # mask of "both teacher preds are wrong"
        wrong_mask = (pred1 != y) & (pred2 != y)
        both_wrong += wrong_mask.sum().item()
        total_samples += y.size(0)

    cross_err_rate = 100.0 * both_wrong / total_samples if total_samples > 0 else 0.0
    return cross_err_rate
