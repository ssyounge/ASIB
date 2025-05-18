import argparse
import yaml
from modules import trainer_teacher, trainer_student
from models import teacher_resnet, teacher_efficientnet, student_resnet_adapter, mbm
from data.cifar100 import get_cifar100_loaders
from utils.logger import setup_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    # 다른 인자들도 원하는 대로 추가
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 로거 설정
    logger = setup_logger(cfg["log_filename"])
    logger.info("Starting ASMB-KD experiment...")

    # 1) Data loader
    trainloader1, trainloader2, testloader = get_cifar100_loaders(cfg)

    # 2) Teacher/Student 모델 생성
    teacher1 = teacher_resnet.create_resnet101_for_cifar100(cfg["teacher1_pretrained"])
    teacher2 = teacher_efficientnet.create_efficientnet_b2_for_cifar100(cfg["teacher2_pretrained"])
    student  = student_resnet_adapter.create_resnet101_with_extended_adapter()

    # 3) partial freeze 등등
    # ...

    # 4) Multi-stage training
    for stage in range(1, cfg["num_stages"]+1):
        # A) teacher_adaptive_update
        trainer_teacher.teacher_adaptive_update(...)
        
        # B) student_distillation_update
        trainer_student.student_distillation_update(...)
    
    logger.info("All done.")

if __name__ == "__main__":
    main()
