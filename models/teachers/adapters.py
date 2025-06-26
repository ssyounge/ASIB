import torch
import torch.nn as nn

class DistillationAdapter(nn.Module):
    """Simple MLP used to refine teacher features for distillation."""

    def __init__(self, in_dim: int, hidden_dim: int | None = None, out_dim: int | None = None) -> None:
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(1, in_dim // 2)
        if out_dim is None:
            out_dim = max(1, in_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
