from torch import randn
import torch

def test_mse_nonzero():
    a = randn(8, 128, requires_grad=True)
    b = randn(8, 128)
    loss = torch.nn.functional.mse_loss(a, b)
    assert loss.item() > 0
