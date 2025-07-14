# utils/dataloader.py

from __future__ import annotations

import torch
import random
import math

class BalancedReplaySampler(torch.utils.data.Sampler):
    """Sample from replay and current indices to keep a given ratio per batch."""

    def __init__(self, cur_indices, rep_indices, batch_size, ratio=0.5, shuffle=True):
        self.cur = list(cur_indices)
        self.rep = list(rep_indices)
        self.bs = batch_size
        self.rc = int(batch_size * ratio)
        self.cc = batch_size - self.rc
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
        # number of batches processed based on current data
        batches = math.ceil(len(self.cur) / max(1, self.cc))
        # total samples = current samples + replay samples per batch
        return len(self.cur) + batches * self.rc
