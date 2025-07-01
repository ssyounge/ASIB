import torch.nn as nn

class Discriminator(nn.Module):
    """Simple MLP discriminator for adversarial feature learning."""

    def __init__(self, in_dim: int, hidden_dim_factor: int = 2):
        super().__init__()
        hidden_dim = max(128, in_dim // hidden_dim_factor)
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        if features.dim() > 2:
            features = features.flatten(1)
        return self.model(features)
