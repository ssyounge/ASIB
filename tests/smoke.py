import torch
from models.common.base_wrapper import MODEL_REGISTRY
from data.cifar100 import get_cifar100_loaders


def main():
    device = torch.device("cpu")
    train_loader, _ = get_cifar100_loaders(batch_size=1, num_workers=0, augment=False)
    model = MODEL_REGISTRY["resnet101_student"](num_classes=100)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            model(x.to(device))
            if i >= 9:
                break


if __name__ == "__main__":
    main()
