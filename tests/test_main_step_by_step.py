#!/usr/bin/env python3
"""
main.py 단계별 실행 테스트
"""

import torch
import logging
import sys
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_main_steps():
    """main.py 단계별 실행 테스트"""
    logger.info("🚀 Testing main.py step by step...")
    
    try:
        # 1단계: 기본 import 확인
        logger.info("Step 1: Testing basic imports...")
        import torch
        import torch.nn as nn
        logger.info("✅ Basic imports successful")
        
        # 2단계: CUDA 확인
        logger.info("Step 2: Testing CUDA...")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name()
            device_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"✅ CUDA available: {device_name} ({device_memory:.1f} GB)")
        else:
            logger.error("❌ CUDA not available!")
            return
        
        # 3단계: 프로젝트 모듈 import 확인
        logger.info("Step 3: Testing project imports...")
        sys.path.insert(0, os.getcwd())
        
        try:
            from core import create_student_by_name, create_teacher_by_name
            logger.info("✅ Core imports successful")
        except Exception as e:
            logger.error(f"❌ Core imports failed: {e}")
            return
        
        # 4단계: 모델 생성 테스트
        logger.info("Step 4: Testing model creation...")
        try:
            student_model = create_student_by_name(
                "resnet50_scratch",
                pretrained=False,
                small_input=True,
                num_classes=100
            )
            logger.info("✅ Student model creation successful")
            
            teacher_model = create_teacher_by_name(
                "resnet152",
                num_classes=100,
                pretrained=False,
                small_input=True
            )
            logger.info("✅ Teacher model creation successful")
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            return
        
        # 5단계: 데이터 로딩 테스트
        logger.info("Step 5: Testing data loading...")
        try:
            from data.cifar100 import get_cifar100_loaders
            train_loader, test_loader = get_cifar100_loaders(
                batch_size=4,
                num_workers=0,
                augment=False
            )
            logger.info("✅ Data loading successful")
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return
        
        # 6단계: IB‑MBM 생성 테스트
        logger.info("Step 6: Testing MBM creation...")
        try:
            from models import build_ib_mbm_from_teachers as build_from_teachers
            cfg = {"ib_mbm_query_dim": 2048}
            mbm, synergy_head = build_from_teachers([teacher_model, teacher_model], cfg)
            logger.info("✅ MBM creation successful")
        except Exception as e:
            logger.error(f"❌ MBM creation failed: {e}")
            return
        
        logger.info("✅ All steps completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_steps() 