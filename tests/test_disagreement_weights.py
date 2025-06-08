import torch
from modules.disagreement import sample_weights_from_disagreement


def test_weight_shape_and_nonneg():
    logit1 = torch.randn(4, 10)
    logit2 = torch.randn(4, 10)
    labels = torch.randint(0, 10, (4,))
    w = sample_weights_from_disagreement(logit1, logit2, labels)
    assert w.shape == labels.shape
    assert torch.all(w >= 0)
