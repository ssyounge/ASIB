import pytest

torch = pytest.importorskip("torch")

from utils.misc import check_label_range


class DummyDataset:
    def __init__(self, labels):
        self.targets = labels


def test_invalid_labels_raise_error():
    dataset = DummyDataset([-1, 0, 1])
    with pytest.raises(ValueError):
        check_label_range(dataset, num_classes=2)


def test_valid_labels_pass():
    dataset = DummyDataset([0, 1, 0])
    check_label_range(dataset, num_classes=2)
