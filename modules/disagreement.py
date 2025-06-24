# modules/disagreement.py

import torch

@torch.no_grad()
def compute_disagreement_rate(teacher1, teacher2, loader, device="cuda"):
    """
    Compute the percentage of samples on which teacher1 and teacher2
    predict *different* labels.

    teacher1, teacher2: nn.Module (teacher wrappers)
        Their forward(x) should return (feat, logit, loss) or similar.
    loader: DataLoader
    device: "cuda" or "cpu"

    Returns:
        float (0~100) indicating the disagreement percentage.
    """
    teacher1.eval()
    teacher2.eval()
    total_samples = 0
    disagree_count = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # forward each teacher, get their logits
        # assume forward(...) => (feat, logit, ce_loss)
        _, logit1, _ = teacher1(x)
        _, logit2, _ = teacher2(x)

        # argmax predictions
        pred1 = logit1.argmax(dim=1)
        pred2 = logit2.argmax(dim=1)

        # mask of samples where the two teachers disagree
        disagree_mask = pred1 != pred2
        disagree_count += disagree_mask.float().sum().item()
        total_samples += disagree_mask.numel()

    # percentage of samples with different predictions
    disagree_rate = (
        100.0 * disagree_count / total_samples if total_samples > 0 else 0.0
    )
    return disagree_rate
