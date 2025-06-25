import pytest

torch = pytest.importorskip("torch")

from models.mbm import build_from_teachers, SynergyHead

class DummyTeacher(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def get_feat_dim(self):
        return self.dim
    def get_feat_channels(self):
        return self.dim

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_classes):
        self.classes = list(range(num_classes))
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return torch.zeros(3), 0

def get_loader(num_classes):
    dataset = DummyDataset(num_classes)
    return torch.utils.data.DataLoader(dataset, batch_size=1)

@pytest.mark.parametrize("n_cls", [3, 7])
def test_synergy_head_output_matches_dataset_class_count(n_cls):
    loader = get_loader(n_cls)
    cfg = {"mbm_out_dim": 8}
    # mimic logic in main.py/eval.py
    num_classes = len(loader.dataset.classes)
    cfg["num_classes"] = num_classes
    teachers = [DummyTeacher(4), DummyTeacher(4)]
    _, head = build_from_teachers(teachers, cfg, query_dim=None)
    assert isinstance(head, SynergyHead)
    assert head[-1].out_features == num_classes
