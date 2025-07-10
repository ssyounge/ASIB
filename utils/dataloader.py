# utils/dataloader.py

import torch
import random

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
        # 전체 샘플 수 (replay + current)
        return len(self.cur) + len(self.rep)
