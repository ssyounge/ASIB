# eval.py
import argparse
import yaml
import torch

from data.cifar100 import get_cifar100_loaders
from modules.trainer_student import eval_student
# 모델 생성 함수를 임포트(예: teacher_resnet, student_resnet 등)
from models.teacher_resnet import create_resnet101_for_cifar100
# 혹은 필요에 맞게 student 모델을 불러오세요
# from models.student_resnet_adapter import create_resnet101_with_extended_adapter

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script for KD model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the checkpoint .pth file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=None)
    # 필요하면 teacher_lr, synergy_ce_alpha 등도 override할 수 있음.
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 명령줄 인자로 batch_size를 바꾸고 싶으면 override
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device = args.device
    print(f"[Eval] Using device={device}, checkpoint={args.ckpt_path}")

    # -------------------------------------------
    # 1) 데이터 로더 생성
    # -------------------------------------------
    train_loader, test_loader = get_cifar100_loaders(
        batch_size=cfg["batch_size"], 
        root=cfg["data_root"]
    )

    # -------------------------------------------
    # 2) 모델 생성 + checkpoint 로드
    # -------------------------------------------
    # 예시: ResNet-101 Student 라고 가정.
    # 필요에 따라 "create_resnet101_for_cifar100" 나 다른 student 함수를 import.
    model = create_resnet101_for_cifar100(pretrained=False)
    model = model.to(device)

    # 체크포인트 로드
    ckpt = torch.load(args.ckpt_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"[Eval] Loaded model_state from {args.ckpt_path}, epoch={ckpt.get('epoch', '?')}")
    else:
        # 혹은 바로 state_dict 형태라면
        model.load_state_dict(ckpt)
        print(f"[Eval] Loaded state_dict from {args.ckpt_path}")

    # -------------------------------------------
    # 3) 테스트 정확도 측정
    # -------------------------------------------
    test_acc = eval_student(model, test_loader, device)
    print(f"[Eval] Test Accuracy= {test_acc:.2f}%")

if __name__ == "__main__":
    main()
