# utils/cl_utils.py

"""Continual learning utilities."""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import random


class ReplayBuffer:
    """Simple replay buffer for continual learning."""
    
    def __init__(self, capacity: int):
        """Initialize replay buffer.
        
        Parameters
        ----------
        capacity : int
            Maximum number of samples to store.
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def update(self, dataset):
        """Update buffer with new dataset."""
        # Simple implementation: store random samples from dataset
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        
        for idx in indices[:self.capacity]:
            if len(self.buffer) < self.capacity:
                self.buffer.append(dataset[idx])
            else:
                self.buffer[self.position] = dataset[idx]
                self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample batch from buffer.
        
        Parameters
        ----------
        batch_size : int
            Number of samples to return.
        device : str
            Device to place tensors on.
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (data, labels) batch.
        """
        if len(self.buffer) == 0:
            return torch.empty(0), torch.empty(0)
        
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        data = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.tensor([item[1] for item in batch]).to(device)
        
        return data, labels


class EWC:
    """Elastic Weight Consolidation for continual learning."""
    
    def __init__(self, lambda_ewc: float = 0.4):
        """Initialize EWC.
        
        Parameters
        ----------
        lambda_ewc : float
            EWC regularization strength.
        """
        self.lambda_ewc = lambda_ewc
        self.fisher_info = {}
        self.optpar = {}
    
    def compute_fisher_info(self, model: nn.Module, dataloader, device: str):
        """Compute Fisher information matrix."""
        model.eval()
        fisher_info = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            model.zero_grad()
            output = model(data)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2 / len(dataloader)
        
        self.fisher_info = fisher_info
        self.optpar = {name: param.data.clone() for name, param in model.named_parameters()}
    
    def compute_ewc_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC regularization loss."""
        if not self.fisher_info:
            return torch.tensor(0.0, device=next(model.parameters()).device)
        
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_info and name in self.optpar:
                ewc_loss += (self.fisher_info[name] * (param - self.optpar[name]) ** 2).sum()
        
        return self.lambda_ewc * ewc_loss 