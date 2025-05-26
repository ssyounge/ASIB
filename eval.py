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

from data.cifar100 import get_cifar100_loaders
from models.mbm import ManifoldBridgingModule, SynergyHead
from utils.logger import ExperimentLogger
from utils.misc import set_random_seed

# ============== Teacher Factory ==============
# Import the three teacher creation functions:
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t

def create_teacher_by_name(teacher_name, num_classes=100, pretrained=False):
    """Creates a teacher model based on teacher_name."""
    if teacher_name == "resnet101":
        return create_resnet101(num_classes=num_classes, pretrained=pretrained)
    elif teacher_name == "efficientnet_b2":
        return create_efficientnet_b2(num_classes=num_classes, pretrained=pretrained)
    elif teacher_name == "swin_tiny":
        return create_swin_t(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"[eval.py] Unknown teacher_name={teacher_name}")

# ============== Argparse, YAML ==============
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
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

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

# ============== Synergy Ensemble ==============
class SynergyEnsemble(nn.Module):
    def __init__(self, teacher1, teacher2, mbm, synergy_head):
        super().__init__()
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.mbm = mbm
        self.synergy_head = synergy_head

    def forward(self, x):
        with torch.no_grad():
            f1_dict, _, _ = self.teacher1(x)
            f2_dict, _, _ = self.teacher2(x)

        # assume we use "feat_2d" for synergy
        f1_2d = f1_dict["feat_2d"]
        f2_2d = f2_dict["feat_2d"]

        fsyn = self.mbm(f1_2d, f2_2d)
        zsyn = self.synergy_head(fsyn)
        return zsyn

def main():
    # 1) parse + load config
    args = parse_args()
    base_cfg = load_config(args.config)
    cfg = {**base_cfg, **vars(args)}

    # 2) set seed
    set_random_seed(cfg["seed"])

    # 3) Logger
    logger = ExperimentLogger(cfg, exp_name="eval_experiment")

    # 4) Data
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg["batch_size"])
    device = cfg["device"]
    n_classes = 100

    if cfg["eval_mode"] == "single":
        # single model eval
        from models.students.student_resnet_adapter import create_resnet101_with_extended_adapter
        model = create_resnet101_with_extended_adapter(pretrained=False).to(device)

        # load single model ckpt
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

        logger.update_metric("eval_mode", "single")
        logger.update_metric("train_acc", train_acc)
        logger.update_metric("test_acc", test_acc)

    else:
        # synergy mode
        # 1) YAML: teacher1_type, teacher2_type
        teacher1_type = cfg.get("teacher1_type", "resnet101")
        teacher2_type = cfg.get("teacher2_type", "efficientnet_b2")

        # 2) create teachers
        teacher1 = create_teacher_by_name(teacher1_type, num_classes=n_classes, pretrained=False).to(device)
        teacher2 = create_teacher_by_name(teacher2_type, num_classes=n_classes, pretrained=False).to(device)

        # 3) load teacher ckpts
        if cfg["teacher1_ckpt"]:
            t1_ck = torch.load(cfg["teacher1_ckpt"], map_location=device)
            teacher1.load_state_dict(t1_ck)
        if cfg["teacher2_ckpt"]:
            t2_ck = torch.load(cfg["teacher2_ckpt"], map_location=device)
            teacher2.load_state_dict(t2_ck)

        # 4) MBM => in_dim from teacher dims
        t1_dim = teacher1.get_feat_dim()
        t2_dim = teacher2.get_feat_dim()
        mbm_in_dim = t1_dim + t2_dim

        mbm = ManifoldBridgingModule(
            in_dim=mbm_in_dim,
            hidden_dim=512,
            out_dim=256
        ).to(device)
        synergy_head = SynergyHead(in_dim=256, num_classes=n_classes).to(device)

        # load MBM, synergy head
        if cfg["mbm_ckpt"]:
            mbm_ck = torch.load(cfg["mbm_ckpt"], map_location=device)
            mbm.load_state_dict(mbm_ck)
        if cfg["head_ckpt"]:
            head_ck = torch.load(cfg["head_ckpt"], map_location=device)
            synergy_head.load_state_dict(head_ck)

        # synergy ensemble
        synergy_model = SynergyEnsemble(teacher1, teacher2, mbm, synergy_head).to(device)

        # evaluate
        train_acc = evaluate_acc(synergy_model, train_loader, device)
        test_acc  = evaluate_acc(synergy_model, test_loader, device)
        print(f"[Synergy] Train={train_acc:.2f}, Test={test_acc:.2f}")

        logger.update_metric("eval_mode", "synergy")
        logger.update_metric("train_acc", train_acc)
        logger.update_metric("test_acc", test_acc)

    logger.finalize()

if __name__ == "__main__":
    main()
