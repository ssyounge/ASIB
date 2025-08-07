#!/usr/bin/env python3
"""
간단한 훈련 테스트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_training():
    """간단한 훈련 테스트"""
    logger.info("🚀 Starting simple training test...")
    
    # CUDA 확인
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"Device: {device_name} ({device_memory:.1f} GB)")
    else:
        logger.error("CUDA not available!")
        return
    
    # 간단한 모델 생성
    model = nn.Linear(10, 1).cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 더미 데이터 생성
    x = torch.randn(100, 10).cuda()
    y = torch.randn(100, 1).cuda()
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=10)
    
    # 훈련 루프
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 5 == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
    
    logger.info("✅ Simple training test completed!")
    
    # 메모리 정리
    del model, optimizer, criterion, dataloader, dataset, x, y
    torch.cuda.empty_cache()
    
    logger.info("✅ Memory cleanup completed!")

if __name__ == "__main__":
    test_simple_training() 