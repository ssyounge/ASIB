# main.py

import argparse
import torch
from utils.logger import ExperimentLogger

def parse_args():
    parser = argparse.ArgumentParser()
    # 아래는 예시. 본인 프로젝트 옵션에 맞게 추가/수정
    parser.add_argument("--method", type=str, default="asmb")
    parser.add_argument("--teacher1", type=str, default="resnet50")
    parser.add_argument("--teacher2", type=str, default="efficientnet_b2")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--stage", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    return args


def train_one_epoch(model, train_loader, optimizer, device="cuda"):
    model.train()
    # 예시: 그냥 로스만 대충 계산
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)  # (feat, logit, None) 같은 식
        # ...
        loss = torch.randn(1).item()  # 임의로 더미 loss
        total_loss += loss
        # optimizer.step() ...
    return total_loss / len(train_loader)


def eval_model(model, test_loader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    # ...
    acc = 80.0 + torch.randn(1).abs().item()  # 임의로 80% 근처로 가정
    return acc


def main():
    args = parse_args()

    # 1) ExperimentLogger 초기화
    logger = ExperimentLogger(args)
    # => 내부에서 exp_id를 만들고, args를 dict로 변환

    # 2) 환경 준비 (seed 등)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3) 데이터 로더 (예시)
    train_loader = None  # load train data
    test_loader = None   # load test data

    # 4) 모델 생성 (Teacher, Student... 여기서는 간단히 student만 예시)
    student_model = torch.nn.Linear(128, 100).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)

    # 5) 학습 루프
    best_acc = 0.0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(student_model, train_loader, optimizer, device)
        test_acc = eval_model(student_model, test_loader, device)

        print(f"Epoch {epoch}: loss={train_loss:.3f}, test_acc={test_acc:.2f}")

        # 필요하면 logger에도 중간 결과 기록할 수 있음
        # logger.update_metric(f"epoch_{epoch}_loss", train_loss)
        # logger.update_metric(f"epoch_{epoch}_acc", test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

    # 6) 최종 결과를 logger에 기록
    logger.update_metric("best_acc", best_acc)
    logger.update_metric("final_loss", train_loss)

    # 7) logger.finalize()로 JSON & summary.csv 저장
    logger.finalize()
    print("[Main] Experiment finished.")


if __name__ == "__main__":
    main()
