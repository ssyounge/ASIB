# tests/test_simple_finetune.py

import pytest; pytest.importorskip("torch")
import torch
from torch.utils.data import DataLoader, TensorDataset

from ASMB_KD import simple_finetune


def test_simple_finetune_runs(tmp_path):
    model = torch.nn.Linear(4, 2)
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=2)
    ckpt = tmp_path / "ft.pth"
    simple_finetune(
        model,
        loader,
        lr=0.1,
        epochs=1,
        device="cpu",
        weight_decay=0.1,
        cfg={},
        ckpt_path=str(ckpt),
    )
    assert model.training
