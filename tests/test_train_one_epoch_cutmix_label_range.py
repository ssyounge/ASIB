import pytest

torch = pytest.importorskip("torch")

from modules.cutmix_finetune_teacher import train_one_epoch_cutmix


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 2)

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))
        return {"logit": self.fc(x)}


def test_train_one_epoch_cutmix_label_range():
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(1, 3, 4, 4)
    y = torch.tensor([2])  # out of range for 2 classes
    loader = [(x, y)]
    with pytest.raises(ValueError, match=r"Dataset labels must be within.*min=2, max=2"):
        train_one_epoch_cutmix(
            model,
            loader,
            optimizer,
            alpha=0.0,
            device="cpu",
            num_classes=2,
        )

