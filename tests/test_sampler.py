# tests/test_sampler.py

import pytest; pytest.importorskip("torch")
import torch

from utils.dataloader import BalancedReplaySampler


@pytest.mark.parametrize(
    "batch_size,ratio,n_cur",
    [
        (4, 0.5, 9),
        (5, 0.3, 7),
        (3, 0.7, 8),
        (6, 0.0, 5),
    ],
)
def test_balanced_replay_sampler_len(batch_size, ratio, n_cur):
    cur_idx = list(range(n_cur))
    rep_idx = list(range(100))
    sampler = BalancedReplaySampler(cur_idx, rep_idx, batch_size, ratio, shuffle=False)
    items = list(iter(sampler))
    assert len(items) == len(sampler)

