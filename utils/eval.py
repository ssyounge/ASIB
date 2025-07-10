# utils/eval.py
from __future__ import annotations
import torch

@torch.no_grad()
def evaluate_acc(
    model,
    loader,
    device: str = "cuda",
    mixup_active: bool = False,   # NEW – 호출 처리를 위한 더미 플래그
    classes: list[int] | None = None,
):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # labels from mixup/cutmix loaders may be one-hot encoded
        # or have an extra singleton dimension. Squeeze first and
        # only apply argmax when a class dimension is present.
        y = y.squeeze()
        if mixup_active and y.ndim > 1:
            y = y.argmax(dim=1)
        # 학습-평가 공통 호출을 위해 mixup_active 플래그만 받는다.
        # 평가 시엔 실제 MixUp 입력이 없으므로 로직 변화 없음.
        out = model(x)
        if isinstance(out, tuple):
            logits = out[1]
        elif isinstance(out, dict):
            logits = out.get("logit", out)
        else:
            logits = out

        # --------------------------------------------------------------
        # Continual-learning: slice logits/labels for a subset of classes
        # --------------------------------------------------------------
        if classes is not None:                     # ← 클래스‑하위 집합 평가
            cls_tensor = torch.tensor(
                classes, dtype=torch.long, device=logits.device
            )
            logits = logits.index_select(1, cls_tensor)
            y = torch.searchsorted(cls_tensor, y, right=False)

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_mbm_acc(teacher1, teacher2, vib_mbm, loader, device="cuda"):
    """Evaluate classification accuracy of ``vib_mbm`` using frozen teachers."""
    vib_mbm.eval()
    teacher1.eval()
    teacher2.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out1 = teacher1(x)
        out2 = teacher2(x)
        t1_dict = out1[0] if isinstance(out1, tuple) else out1
        t2_dict = out2[0] if isinstance(out2, tuple) else out2
        logits = vib_mbm(t1_dict["feat_2d"], t2_dict["feat_2d"])[1]
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
