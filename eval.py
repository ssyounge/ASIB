# eval.py
"""
Evaluates either a single model or a synergy model (Teacher1+2 + MBM + synergy head),
and logs the results (train_acc, test_acc, etc.) using ExperimentLogger.
"""

import argparse
import yaml
import torch
import torch.nn as nn
import os

########################################
# (A) Data, Models
########################################
from data.cifar100 import get_cifar100_loaders
# or from data.imagenet100 import get_imagenet100_loaders

from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.mbm import ManifoldBridgingModule, SynergyHead

# single model example:
# from models.student_resnet_adapter import create_resnet101_with_extended_adapter

########################################
# (B) Logger, misc
########################################
from utils.logger import ExperimentLogger
from utils.misc import set_random_seed

########################################
# parse_args, load_config
########################################
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script (Train/Test Acc) with ExperimentLogger")

    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--eval_mode", type=str, default="single", choices=["single","synergy"],
                        help="Evaluate single model or synergy model")
    # single model
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Single model checkpoint path")

    # synergy
    parser.add_argument("--teacher1_ckpt", type=str, default=None)
    parser.add_argument("--teacher2_ckpt", type=str, default=None)
    parser.add_argument("--mbm_ckpt", type=str, default=None)
    parser.add_argument("--head_ckpt", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory to store logs (JSON+CSV via logger)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

########################################
# (C) Evaluate function
########################################
@torch.no_grad()
def evaluate_acc(model, loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

########################################
# Synergy Wrapper
########################################
class SynergyEnsemble(nn.Module):
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
# (D) Main
########################################
def main():
    args = parse_args()

    # 1) optionally load YAML config and merge
    config = {}
    if os.path.exists(args.config):
        config = load_config(args.config)

    # merge
    cfg = {**config, **vars(args)}

    # 2) set seed
    set_random_seed(cfg["seed"])

    # 3) create logger
    logger = ExperimentLogger(cfg, exp_name="eval_experiment")

    # 4) data
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg["batch_size"])
    device = cfg["device"]
    n_classes = 100

    if cfg["eval_mode"] == "single":
        # single model
        from models.student_resnet_adapter import create_resnet101_with_extended_adapter
        model = create_resnet101_with_extended_adapter(pretrained=False).to(device)

        # load ckpt
        if cfg["ckpt_path"]:
            ckpt = torch.load(cfg["ckpt_path"], map_location=device)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
            print(f"[Eval single] loaded from {cfg['ckpt_path']}")
        else:
            print("[Eval single] no ckpt => random init")

        train_acc = evaluate_acc(model, train_loader, device)
        test_acc  = evaluate_acc(model, test_loader, device)
        print(f"[Single] Train={train_acc:.2f}, Test={test_acc:.2f}")

        # logger update
        logger.update_metric("eval_mode", "single")
        logger.update_metric("train_acc", train_acc)
        logger.update_metric("test_acc", test_acc)

    else:
        # synergy
        teacher1 = create_resnet101(num_classes=n_classes, pretrained=False).to(device)
        teacher2 = create_efficientnet_b2(num_classes=n_classes, pretrained=False).to(device)
        mbm = ManifoldBridgingModule(in_dim=2048+1408, hidden_dim=512, out_dim=256).to(device)
        synergy_head = SynergyHead(in_dim=256, num_classes=n_classes).to(device)

        if cfg["teacher1_ckpt"]:
            t1_ck = torch.load(cfg["teacher1_ckpt"], map_location=device)
            teacher1.load_state_dict(t1_ck)
        if cfg["teacher2_ckpt"]:
            t2_ck = torch.load(cfg["teacher2_ckpt"], map_location=device)
            teacher2.load_state_dict(t2_ck)
        if cfg["mbm_ckpt"]:
            mbm_ck = torch.load(cfg["mbm_ckpt"], map_location=device)
            mbm.load_state_dict(mbm_ck)
        if cfg["head_ckpt"]:
            head_ck = torch.load(cfg["head_ckpt"], map_location=device)
            synergy_head.load_state_dict(head_ck)

        synergy_model = SynergyEnsemble(teacher1, teacher2, mbm, synergy_head).to(device)
        train_acc = evaluate_acc(synergy_model, train_loader, device)
        test_acc  = evaluate_acc(synergy_model, test_loader, device)
        print(f"[Synergy] Train={train_acc:.2f}, Test={test_acc:.2f}")

        logger.update_metric("eval_mode", "synergy")
        logger.update_metric("train_acc", train_acc)
        logger.update_metric("test_acc", test_acc)

    # finalize => saves JSON + appends summary.csv
    logger.finalize()

if __name__ == "__main__":
    main()
