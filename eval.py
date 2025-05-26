# eval.py

import argparse
import yaml
import torch
import torch.nn as nn
import copy
import os
from datetime import datetime
import csv

########################################
# (A) Data loader
########################################
from data.cifar100 import get_cifar100_loaders
# or from data.imagenet100 import get_imagenet100_loaders

########################################
# Teacher / Student / MBM
########################################
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.mbm import ManifoldBridgingModule, SynergyHead

# single model example
# from models.student_resnet_adapter import create_resnet101_with_extended_adapter

########################################
# parse_args, load_config
########################################
def parse_args():
    parser = argparse.ArgumentParser(description="Eval script (Train/Test Acc) + CSV logging")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--eval_mode", type=str, default="single", choices=["single","synergy"])
    
    # single model
    parser.add_argument("--ckpt_path", type=str, default=None, help="Single model ckpt path")

    # synergy
    parser.add_argument("--teacher1_ckpt", type=str, default=None)
    parser.add_argument("--teacher2_ckpt", type=str, default=None)
    parser.add_argument("--mbm_ckpt", type=str, default=None)
    parser.add_argument("--head_ckpt", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="./results", help="Directory to save CSV logs")
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

########################################
# (B) Accuracy Evaluate (Train/Test)
########################################
@torch.no_grad()
def evaluate_acc(model, loader, device="cuda"):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

########################################
# Synergy Wrapper
########################################
class SynergyEnsemble(nn.Module):
    """
    Teacher1, Teacher2, MBM, synergyHead => synergy logit
    """
    def __init__(self, teacher1, teacher2, mbm, synergy_head):
        super().__init__()
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.mbm = mbm
        self.synergy_head = synergy_head

    def forward(self, x):
        with torch.no_grad():
            f1, _, _ = self.teacher1(x)
            f2, _, _ = self.teacher2(x)
        fsyn = self.mbm(f1, f2)
        zsyn = self.synergy_head(fsyn)
        return zsyn

########################################
# (C) Main
########################################
def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    device = args.device
    print(f"[Eval] eval_mode={args.eval_mode}, device={device}, config={args.config}")

    # make sure save_dir exists
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) Dataloader
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg["batch_size"])
    n_classes = 100

    # 2) eval_mode branch
    if args.eval_mode == "single":
        # single model
        from models.student_resnet_adapter import create_resnet101_with_extended_adapter
        model = create_resnet101_with_extended_adapter(pretrained=False)
        model.to(device)

        # ckpt load
        if args.ckpt_path:
            ckpt = torch.load(args.ckpt_path, map_location=device)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                print(f"[Eval single] loaded model_state")
            else:
                model.load_state_dict(ckpt)
            print("[Eval single] ckpt loaded.")
        else:
            print("[Eval single] no ckpt => random init")

        # Evaluate
        train_acc = evaluate_acc(model, train_loader, device)
        test_acc  = evaluate_acc(model, test_loader, device)
        print(f"[SingleModel] TrainAcc={train_acc:.2f}%, TestAcc={test_acc:.2f}%")

        # (D) Save results => CSV
        save_results_csv(
            save_dir=args.save_dir,
            eval_mode="single",
            train_acc=train_acc,
            test_acc=test_acc,
            batch_size=cfg["batch_size"],
            config_path=args.config
        )

    else:
        # synergy
        teacher1 = create_resnet101(num_classes=n_classes, pretrained=False).to(device)
        teacher2 = create_efficientnet_b2(num_classes=n_classes, pretrained=False).to(device)
        mbm = ManifoldBridgingModule(in_dim=2048+1408, hidden_dim=512, out_dim=256).to(device)
        synergy_head = SynergyHead(in_dim=256, num_classes=n_classes).to(device)

        if args.teacher1_ckpt:
            t1_ck = torch.load(args.teacher1_ckpt, map_location=device)
            teacher1.load_state_dict(t1_ck)
        if args.teacher2_ckpt:
            t2_ck = torch.load(args.teacher2_ckpt, map_location=device)
            teacher2.load_state_dict(t2_ck)
        if args.mbm_ckpt:
            mbm_ck = torch.load(args.mbm_ckpt, map_location=device)
            mbm.load_state_dict(mbm_ck)
        if args.head_ckpt:
            head_ck = torch.load(args.head_ckpt, map_location=device)
            synergy_head.load_state_dict(head_ck)

        synergy_model = SynergyEnsemble(teacher1, teacher2, mbm, synergy_head).to(device)

        train_acc = evaluate_acc(synergy_model, train_loader, device)
        test_acc  = evaluate_acc(synergy_model, test_loader, device)
        print(f"[SynergyModel] TrainAcc={train_acc:.2f}%, TestAcc={test_acc:.2f}%")

        save_results_csv(
            save_dir=args.save_dir,
            eval_mode="synergy",
            train_acc=train_acc,
            test_acc=test_acc,
            batch_size=cfg["batch_size"],
            config_path=args.config
        )

########################################
# (E) Save CSV Results
########################################
def save_results_csv(
    save_dir,
    eval_mode,
    train_acc,
    test_acc,
    batch_size,
    config_path
):
    """
    - Generates unique filename: results_{eval_mode}_{timestamp}.csv
    - Writes a single row with columns: timestamp, eval_mode, train_acc, test_acc, batch_size, config_path
    """
    from datetime import datetime
    import csv

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{eval_mode}_{ts}.csv"
    filepath = os.path.join(save_dir, filename)

    # For example, we create a new file each run (one row).
    # If you want to append multiple runs in the same file, you'd do 'a' mode + check header existence.
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        # write header
        writer.writerow(["timestamp", "eval_mode", "train_acc", "test_acc", "batch_size", "config"])
        # write one row
        writer.writerow([ts, eval_mode, f"{train_acc:.2f}", f"{test_acc:.2f}", batch_size, config_path])

    print(f"[CSV] Results saved => {filepath}")


if __name__ == "__main__":
    main()
