import torch
from modules.disagreement import compute_disagreement_rate


def test_disagreement_rate_range(dummy_teachers):
    t1, t2 = dummy_teachers
    loader = [(torch.zeros(1, 3), torch.tensor([0]))]
    rate = compute_disagreement_rate(t1, t2, loader, device="cpu")
    assert 0.0 <= rate <= 100.0
