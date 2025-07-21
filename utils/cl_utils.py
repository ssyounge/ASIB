# utils/cl_utils.py
"""Simplified continual-learning utilities."""

import random
from typing import List, Tuple

import torch


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, int, int]] = []

    def add(self, dataset, task_id: int) -> None:
        self.buffer.extend([(x, y, task_id) for x, y in dataset])
        if len(self.buffer) > self.capacity:
            self.buffer = random.sample(self.buffer, self.capacity)

    def sample(self, n: int, device="cpu"):
        batch = random.sample(self.buffer, min(n, len(self.buffer)))
        if not batch:
            return [], [], []
        xs, ys, tids = zip(*batch)
        xs = torch.stack(xs).to(device)
        ys = torch.tensor(ys, device=device)
        return xs, ys, tids


class EWC:
    def __init__(self, lambda_ewc: float = 0.4) -> None:
        self.lmbd = lambda_ewc
        self.params = {}
        self.fisher = {}

    def update_fisher(self, model, loader, device="cuda") -> None:
        model.eval()
        self.params = {
            n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad
        }
        self.fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)[1] if isinstance(model(x), tuple) else model(x)
            loss = torch.nn.functional.cross_entropy(out, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.grad is not None:
                    self.fisher[n] += (p.grad.detach() ** 2) / len(loader)

    def penalty(self, model) -> torch.Tensor:
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.params:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return self.lmbd * loss

