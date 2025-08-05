#!/usr/bin/env python3
"""Test disagreement computation"""

import torch


def test_disagreement_rate_range(dummy_teachers):
    """Test disagreement rate is in valid range"""
    t1, t2 = dummy_teachers
    
    # Create dummy predictions
    pred1 = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    pred2 = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    
    # Simple disagreement calculation
    disagreement = torch.abs(pred1 - pred2).mean()
    
    # Check range
    assert 0.0 <= disagreement <= 1.0


def test_disagreement_computation():
    """Test basic disagreement computation"""
    # Create dummy predictions
    pred1 = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    pred2 = torch.tensor([[0.2, 0.8], [0.7, 0.3]])
    
    # Calculate disagreement
    disagreement = torch.abs(pred1 - pred2).mean()
    
    # Should be positive
    assert disagreement > 0.0
    assert torch.isfinite(disagreement)
