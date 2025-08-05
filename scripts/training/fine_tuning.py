# scripts/fine_tuning.py
"""
Example: Fine-tuning Teacher (ResNet/EfficientNet/Swin) on either CIFAR-100 or ImageNet100
using optional CutMix or standard CE training.

Usage:
  python scripts/fine_tuning.py --config-name base \
      +teacher_type=resnet152 +finetune_epochs=100 +finetune_lr=0.0005

Change datasets with a ``dataset.name`` override or a dataset YAML file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import logging
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
from utils.logging import init_logger
from utils.common import (
    set_random_seed,
    check_label_range,
    get_model_num_classes,
    count_trainable_parameters,
    smart_tqdm,
    get_amp_components,
)

# data loaders
from data.cifar100 import get_cifar100_loaders
from data.imagenet32 import get_imagenet32_loaders

# teacher factories
from core.builder import build_model, create_teacher_by_name

# partial freeze
from modules.partial_freeze import (
    partial_freeze_teacher_resnet,
    partial_freeze_teacher_efficientnet,
)

# cutmix finetune
from modules.cutmix_finetune_teacher import finetune_teacher_cutmix, eval_teacher


def run_fine_tuning(config):
    """
    Run fine-tuning experiment with given configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing training parameters
        
    Returns:
    --------
    dict
        Training results and metrics
    """
    # Extract config parameters
    logging.info(f"Config keys: {list(config.keys())}")
    
    # config가 중첩되어 있는 경우 처리
    if 'finetune' in config:
        finetune_config = config['finetune']
        logging.info(f"Using nested finetune config")
    else:
        finetune_config = config
        logging.info(f"Using flat config")
    
    logging.info(f"Finetune config keys: {list(finetune_config.keys())}")
    
    teacher_type = finetune_config.get('teacher_type', 'resnet152')
    dataset_name = finetune_config.get('dataset_name', 'cifar100')
    batch_size = finetune_config.get('batch_size', 128)
    epochs = finetune_config.get('finetune_epochs', 100)
    lr = finetune_config.get('finetune_lr', 0.0005)
    weight_decay = finetune_config.get('finetune_weight_decay', 1e-4)
    device = finetune_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    logging.info(f"Extracted teacher_type: {teacher_type}")
    logging.info(f"Extracted dataset_name: {dataset_name}")
    
    # Set random seed
    seed = config.get('seed', 42)
    set_random_seed(seed)
    
    # Get data loaders
    try:
        data_root = finetune_config.get('data_root', './data')
        train_loader, test_loader = get_data_loaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=0,  # 데이터 로더 문제 방지
            augment=True,
            root=data_root
        )
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return {'error': f'Data loading failed: {e}'}
    
    # Create teacher model
    try:
        # CIFAR-100은 100개 클래스, ImageNet32는 1000개 클래스
        if dataset_name == "cifar100":
            num_classes = 100
        elif dataset_name == "imagenet32":
            num_classes = 1000
        else:
            raise ValueError(f"Unknown dataset_name={dataset_name}")
            
        teacher = create_teacher_by_name(
            teacher_name=teacher_type,
            num_classes=num_classes,
            pretrained=finetune_config.get('teacher_pretrained', True),
            small_input=finetune_config.get('small_input', False),
            cfg=config
        )
        teacher = teacher.to(device)
    except Exception as e:
        logging.error(f"Failed to create teacher model: {e}")
        return {'error': f'Model creation failed: {e}'}
    
    # Training results
    results = {
        'teacher_type': teacher_type,
        'dataset': dataset_name,
        'epochs': epochs,
        'final_accuracy': 0.0,
        'final_loss': float('inf'),
        'training_history': []
    }
    
    # 실제 훈련 실행
    logging.info(f"Starting fine-tuning for {teacher_type} on {dataset_name}")
    logging.info(f"Training for {epochs} epochs with lr={lr}")
    
    # 체크포인트 경로 설정
    ckpt_path = finetune_config.get('finetune_ckpt_path', f'checkpoints/{teacher_type}_{dataset_name}.pth')
    
    # 실제 훈련 실행
    training_results = standard_ce_finetune(
        model=teacher,
        train_loader=train_loader,
        test_loader=test_loader,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        device=device,
        ckpt_path=ckpt_path,
        label_smoothing=finetune_config.get('label_smoothing', 0.0),
        cfg=config
    )
    
    # 결과 업데이트
    results.update(training_results)
    
    logging.info(f"Fine-tuning completed. Final accuracy: {results['final_accuracy']:.4f}")
    
    return results


def get_data_loaders(
    dataset_name, batch_size=128, num_workers=2, augment=True, root=None
):
    """
    Returns train_loader, test_loader based on dataset_name.
    """
    if dataset_name == "cifar100":
        return get_cifar100_loaders(
            root=root or "./data",
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    elif dataset_name == "imagenet32":
        return get_imagenet32_loaders(
            root=root or "./data/imagenet32",
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}")



def partial_freeze_teacher_auto(
    model,
    teacher_type,
    freeze_bn=True,
    freeze_ln=True,
    use_adapter=False,
    bn_head_only=False,
    freeze_level=1,
):
    """Automatically apply partial freezing based on teacher type."""
    if "resnet" in teacher_type.lower():
        return partial_freeze_teacher_resnet(
            model,
            freeze_bn=freeze_bn,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    elif "efficientnet" in teacher_type.lower():
        return partial_freeze_teacher_efficientnet(
            model,
            freeze_bn=freeze_bn,
            bn_head_only=bn_head_only,
            freeze_level=freeze_level,
        )
    else:
        # Default: no freezing
        return model


def standard_ce_finetune(
    model,
    train_loader,
    test_loader,
    lr,
    weight_decay,
    epochs,
    device,
    ckpt_path,
    label_smoothing: float = 0.0,
    cfg=None,
): 
    """Standard cross-entropy fine-tuning with advanced scheduling."""
    import torch.optim as optim
    from torch.nn import CrossEntropyLoss
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 손실 함수 설정
    if label_smoothing > 0:
        criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    else:
        criterion = CrossEntropyLoss()
    
    # 고급 스케줄링 설정
    scheduler_type = cfg.get('scheduler_type', 'cosine') if cfg else 'cosine'
    warmup_epochs = cfg.get('warmup_epochs', 0) if cfg else 0
    min_lr = cfg.get('min_lr', 1e-6) if cfg else 1e-6
    
    # Warm-up 방어구문
    if warmup_epochs >= epochs:
        logging.warning(f"warmup_epochs({warmup_epochs}) >= epochs({epochs}) → warmup_epochs = {max(0, epochs - 1)}")
        warmup_epochs = max(0, epochs - 1)
    
    # 스케줄러 선택 및 설정
    scheduler = None
    if scheduler_type == 'onecycle':
        # OneCycleLR: 가장 효과적인 스케줄링
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=warmup_epochs/epochs if warmup_epochs > 0 else 0.1,
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr/25
            final_div_factor=1e4,  # min_lr = initial_lr/1e4
        )
        logging.info(f"Using OneCycleLR scheduler (max_lr={lr}, epochs={epochs})")
        
    elif scheduler_type == 'reduce_on_plateau':
        # ReduceLROnPlateau: 검증 성능 기반
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=True
        )
        logging.info(f"Using ReduceLROnPlateau scheduler (patience=5, factor=0.5)")
        
    elif scheduler_type == 'multistep':
        # MultiStepLR: 명시적 스케줄링
        milestones = cfg.get('lr_milestones', [epochs//3, 2*epochs//3]) if cfg else [epochs//3, 2*epochs//3]
        gamma = cfg.get('lr_gamma', 0.1) if cfg else 0.1
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        logging.info(f"Using MultiStepLR scheduler (milestones={milestones}, gamma={gamma})")
        
    elif scheduler_type == 'cosine_warm_restarts':
        # CosineAnnealingWarmRestarts: 주기적 재시작
        T_0 = cfg.get('restart_period', epochs//3) if cfg else epochs//3
        T_mult = cfg.get('restart_multiplier', 2) if cfg else 2
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=min_lr
        )
        logging.info(f"Using CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
        
    else:
        # 기본: Cosine Annealing (warm-up 포함)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr
        )
        logging.info(f"Using CosineAnnealingLR scheduler (T_max={epochs-warmup_epochs})")
    
    # 훈련 루프
    model.train()
    best_acc = 0.0
    
    # Early stopping 설정
    early_stopping_patience = cfg.get('early_stopping_patience', 10) if cfg else 10
    early_stopping_min_delta = cfg.get('early_stopping_min_delta', 0.1) if cfg else 0.1
    patience_counter = 0
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Warm-up 적용 (OneCycleLR이 아닌 경우에만)
        if scheduler_type != 'onecycle' and epoch < warmup_epochs:
            warm_lr = (epoch / warmup_epochs) * lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = warm_lr
            logging.info(f"Epoch {epoch:3d}: Warm-up LR = {warm_lr:.6f}")
        elif scheduler_type != 'onecycle' and scheduler is not None:
            # OneCycleLR이 아닌 경우 스케줄러 적용
            if scheduler_type == 'reduce_on_plateau':
                # ReduceLROnPlateau는 검증 후에 적용
                pass
            else:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch {epoch:3d}: {scheduler_type} LR = {current_lr:.6f}")
        
        # 훈련
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # BaseKDModel은 (feat_dict, logit, aux) 튜플을 반환
            if isinstance(output, tuple):
                if len(output) == 3:
                    output = output[1]  # logit (두 번째 요소)
                elif len(output) == 2:
                    output = output[1]  # logits는 보통 두 번째 요소
                else:
                    output = output[0]  # 첫 번째 요소 사용
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # OneCycleLR은 매 step마다 업데이트
            if scheduler_type == 'onecycle' and scheduler is not None:
                scheduler.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        # 검증
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # BaseKDModel은 (feat_dict, logit, aux) 튜플을 반환
                if isinstance(output, tuple):
                    if len(output) == 3:
                        output = output[1]  # logit (두 번째 요소)
                    elif len(output) == 2:
                        output = output[1]  # logits는 보통 두 번째 요소
                    else:
                        output = output[0]  # 첫 번째 요소 사용
                
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        # 메트릭 계산
        train_acc = 100.0 * correct / total
        val_acc = 100.0 * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        
        # 로깅 - 매 에포크마다 출력
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Epoch {epoch:3d}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%, LR={current_lr:.6f}")
        
        # ReduceLROnPlateau 스케줄러 업데이트 (검증 후)
        if scheduler_type == 'reduce_on_plateau' and scheduler is not None:
            scheduler.step(val_acc)
        
        # 최고 성능 모델 저장
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"New best model saved: {val_acc:.2f}%")
        
        # Early stopping 체크
        if val_acc > best_val_acc + early_stopping_min_delta:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered at epoch {epoch}. Best accuracy: {best_acc:.2f}%")
            break
    
    return {
        'final_accuracy': best_acc,
        'final_loss': avg_val_loss,
        'epochs_trained': epoch + 1,
        'best_accuracy': best_acc
    }


@hydra.main(config_path="../../configs", config_name="base", version_base="1.3")
def main(cfg: DictConfig):
    """Main function for fine-tuning."""
    # Convert config to dict for compatibility
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # Run fine-tuning
    results = run_fine_tuning(config)
    
    # Log results
    if 'error' in results:
        logging.error(f"Fine-tuning failed: {results['error']}")
    else:
        logging.info(f"Fine-tuning completed successfully: {results}")


if __name__ == "__main__":
    main()
