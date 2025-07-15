# utils/dataloader.py

from __future__ import annotations

import torch
import random
import math

class BalancedReplaySampler(torch.utils.data.Sampler):
    """Sample from replay and current indices to keep a given ratio per batch."""

    def __init__(self, cur_indices, rep_indices, batch_size,
                 ratio: float = 0.5, shuffle: bool = True):
        # ── sanity-check: 0 ≤ ratio < 1  (ratio=1 → cc = 0 ⇒ 무한 루프 위험)
        if not (0.0 <= ratio < 1.0):
            raise ValueError("ratio must be in [0, 1).")
        self.cur = list(cur_indices)
        self.rep = list(rep_indices)
        self.bs  = batch_size
        self.rc  = int(batch_size * ratio)           # replay per batch
        # 최소 1 개의 current 샘플을 보장
        self.cc  = max(1, batch_size - self.rc)      # current per batch
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.cur)
            random.shuffle(self.rep)
        rep_ptr = cur_ptr = 0
        while cur_ptr < len(self.cur):
            rep_batch = self.rep[rep_ptr:rep_ptr + self.rc]
            cur_batch = self.cur[cur_ptr:cur_ptr + self.cc]
            if len(rep_batch) < self.rc:
                if self.shuffle:
                    random.shuffle(self.rep)
                rep_ptr = 0
                rep_batch = self.rep[rep_ptr:rep_ptr + self.rc]
            yield from rep_batch + cur_batch
            rep_ptr += self.rc
            cur_ptr += self.cc

    def __len__(self):
        """Return the number of samples yielded by the sampler."""
        cur_samples = len(self.cur)
        replay_samples = math.ceil(cur_samples / self.cc) * self.rc
        return cur_samples + replay_samples
