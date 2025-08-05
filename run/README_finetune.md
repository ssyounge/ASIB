# Finetune 스크립트 사용법

## 📁 **스크립트 파일들**

### 🚀 **전체 자동 실행**
```bash
# 모든 teacher들을 순차적으로 finetune
./run/run_finetune_all_teachers.sh
```

### 🎯 **개별 실행**
```bash
# 특정 teacher만 finetune
./run/run_finetune_single.sh convnext_s_cifar100
./run/run_finetune_single.sh convnext_s_imagenet32
./run/run_finetune_single.sh convnext_l_cifar100
./run/run_finetune_single.sh convnext_l_imagenet32
./run/run_finetune_single.sh efficientnet_l2_cifar100
./run/run_finetune_single.sh efficientnet_l2_imagenet32
./run/run_finetune_single.sh resnet152_cifar100
./run/run_finetune_single.sh resnet152_imagenet32
```

## 📊 **Teacher 모델들**

| Teacher | Parameters | Features | Config File |
|---------|------------|----------|-------------|
| ConvNeXt-Small (CIFAR-100) | 49.6M | 768 | `convnext_s_cifar100` |
| ConvNeXt-Small (ImageNet-32) | 49.6M | 768 | `convnext_s_imagenet32` |
| ConvNeXt-Large (CIFAR-100) | 197M | 1536 | `convnext_l_cifar100` |
| ConvNeXt-Large (ImageNet-32) | 197M | 1536 | `convnext_l_imagenet32` |
| EfficientNet-L2 (CIFAR-100) | 480M | 1408 | `efficientnet_l2_cifar100` |
| EfficientNet-L2 (ImageNet-32) | 480M | 1408 | `efficientnet_l2_imagenet32` |
| ResNet152 (CIFAR-100) | 60M | 2048 | `resnet152_cifar100` |
| ResNet152 (ImageNet-32) | 60M | 2048 | `resnet152_imagenet32` |

## 📁 **출력 경로**

### Checkpoints
- `experiments/checkpoints/convnext_s_cifar100.pth`
- `experiments/checkpoints/convnext_s_imagenet32.pth`
- `experiments/checkpoints/convnext_l_cifar100.pth`
- `experiments/checkpoints/convnext_l_imagenet32.pth`
- `experiments/checkpoints/efficientnet_l2_cifar100.pth`
- `experiments/checkpoints/efficientnet_l2_imagenet32.pth`
- `experiments/checkpoints/resnet152_cifar100.pth`
- `experiments/checkpoints/resnet152_imagenet32.pth`

### Results
- `experiments/outputs/finetune/convnext_s_cifar100_ft/`
- `experiments/outputs/finetune/convnext_s_imagenet32_ft/`
- `experiments/outputs/finetune/convnext_l_cifar100_ft/`
- `experiments/outputs/finetune/convnext_l_imagenet32_ft/`
- `experiments/outputs/finetune/efficientnet_l2_cifar100_ft/`
- `experiments/outputs/finetune/efficientnet_l2_imagenet32_ft/`
- `experiments/outputs/finetune/resnet152_cifar100_ft/`
- `experiments/outputs/finetune/resnet152_imagenet32_ft/`

## ⚙️ **설정 파일들**
- `configs/finetune/convnext_s_cifar100.yaml`
- `configs/finetune/convnext_s_imagenet32.yaml`
- `configs/finetune/convnext_l_cifar100.yaml`
- `configs/finetune/convnext_l_imagenet32.yaml`
- `configs/finetune/efficientnet_l2_cifar100.yaml`
- `configs/finetune/efficientnet_l2_imagenet32.yaml`
- `configs/finetune/resnet152_cifar100.yaml`
- `configs/finetune/resnet152_imagenet32.yaml`

## 🧪 **테스트**
```bash
# 설정 파일들이 올바른지 테스트
python scripts/test_finetune_configs.py
``` 