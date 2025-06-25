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
from models.mbm import ManifoldBridgingModule, SynergyHead, build_from_teachers
from models.la_mbm import LightweightAttnMBM
from utils.logger import ExperimentLogger
from utils.misc import set_random_seed, get_amp_components
from main import create_student_by_name

# Teacher Factory
# Import the three teacher creation functions:
from models.teachers.teacher_resnet import create_resnet101
from models.teachers.teacher_efficientnet import create_efficientnet_b2
from models.teachers.teacher_swin import create_swin_t

def create_teacher_by_name(teacher_name, num_classes=100, pretrained=False, small_input=False):
    """Creates a teacher model based on teacher_name."""
    if teacher_name == "resnet101":
        return create_resnet101(num_classes=num_classes, pretrained=pretrained, small_input=small_input)
    elif teacher_name == "efficientnet_b2":
        return create_efficientnet_b2(
            num_classes=num_classes,
            pretrained=pretrained,
            small_input=small_input,
        )
    elif teacher_name == "swin_tiny":
        return create_swin_t(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"[eval.py] Unknown teacher_name={teacher_name}")

# Argparse, YAML
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
    parser.add_argument("--student_type", type=str, default="resnet_adapter")
    parser.add_argument("--student_ckpt", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--results_dir", type=str, default="./results",
                        help="Directory to store logs (JSON+CSV via logger)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--small_input", type=int,
                        help="1 for CIFAR-style conv stem in teachers")
    parser.add_argument("--use_amp", type=int)
    parser.add_argument("--amp_dtype", type=str)

    # MBM options (overrides YAML)
    parser.add_argument("--mbm_type", type=str)
    parser.add_argument("--mbm_r", type=int)
    parser.add_argument("--mbm_n_head", type=int)
    parser.add_argument("--mbm_learnable_q", type=int)
    return parser.parse_args()

def load_config(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    return {}

@torch.no_grad()
def evaluate_acc(model, loader, device="cuda", cfg=None):
    autocast_ctx, _ = get_amp_components(cfg or {})
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            out = model(x)
            if isinstance(out, tuple):
                logits = out[1]
            elif isinstance(out, dict):
                logits = out["logit"]
            else:
                logits = out
            preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / total

# Synergy Ensemble
class SynergyEnsemble(nn.Module):
    def __init__(self, teacher1, teacher2, mbm, synergy_head, student=None, cfg=None):
        super().__init__()
        self.teacher1 = teacher1
        self.teacher2 = teacher2
        self.mbm = mbm
        self.synergy_head = synergy_head
        self.student = student
        self.cfg = cfg or {}
        self.la_mode = isinstance(mbm, LightweightAttnMBM) or self.cfg.get("mbm_type") == "LA"

    def forward(self, x):
        with torch.no_grad():
            f1_dict = self.teacher1(x)
            f2_dict = self.teacher2(x)

        f1_2d = f1_dict["feat_2d"]
        f2_2d = f2_dict["feat_2d"]
        f1_4d = f1_dict.get("feat_4d")
        f2_4d = f2_dict.get("feat_4d")

        if self.la_mode:
            assert self.student is not None, "student required for LA MBM"
            feat_dict, _, _ = self.student(x)
            key = self.cfg.get("feat_kd_key", "feat_2d")
            s_feat = feat_dict[key]
            fsyn, _, _, _ = self.mbm(s_feat, [f1_2d, f2_2d])
        else:
            fsyn = self.mbm([f1_2d, f2_2d], [f1_4d, f2_4d])

        zsyn = self.synergy_head(fsyn)
        return zsyn

def main():
    # 1) parse + load config
    args = parse_args()
    base_cfg = load_config(args.config)
    cfg = {**base_cfg, **vars(args)}

    # 2) set seed
    deterministic = cfg.get("deterministic", True)
    set_random_seed(cfg["seed"], deterministic=deterministic)

    # 3) Logger
    logger = ExperimentLogger(cfg, exp_name="eval_experiment")
    logger.update_metric("use_amp", cfg.get("use_amp", False))
    logger.update_metric("amp_dtype", cfg.get("amp_dtype", "float16"))
    logger.update_metric("mbm_type", cfg.get("mbm_type", "MLP"))
    logger.update_metric("mbm_r", cfg.get("mbm_r"))
    logger.update_metric("mbm_n_head", cfg.get("mbm_n_head"))
    logger.update_metric("mbm_learnable_q", cfg.get("mbm_learnable_q"))

    # 4) Data
    dataset_name = cfg.get("dataset_name", "cifar100")
    train_loader, test_loader = get_cifar100_loaders(batch_size=cfg["batch_size"])
    device = cfg["device"]
    small_input = cfg.get("small_input")
    if small_input is None:
        small_input = dataset_name == "cifar100"
    n_classes = 100

    if cfg["eval_mode"] == "single":
        # single model eval
        from models.students.student_resnet_adapter import create_resnet101_with_extended_adapter
        model = create_resnet101_with_extended_adapter(pretrained=False).to(device)

        # load single model ckpt
        if cfg["ckpt_path"]:
            ckpt = torch.load(
                cfg["ckpt_path"], map_location=device, weights_only=True
            )
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
            else:
                model.load_state_dict(ckpt)
            print(f"[Eval single] loaded from {cfg['ckpt_path']}")
        else:
            print("[Eval single] no ckpt => random init")

        train_acc = evaluate_acc(model, train_loader, device, cfg)
        test_acc  = evaluate_acc(model, test_loader, device, cfg)
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
        teacher1 = create_teacher_by_name(
            teacher1_type,
            num_classes=n_classes,
            pretrained=False,
            small_input=small_input,
        ).to(device)
        teacher2 = create_teacher_by_name(
            teacher2_type,
            num_classes=n_classes,
            pretrained=False,
            small_input=small_input,
        ).to(device)

        # 3) load teacher ckpts
        if cfg["teacher1_ckpt"]:
            t1_ck = torch.load(
                cfg["teacher1_ckpt"], map_location=device, weights_only=True
            )
            teacher1.load_state_dict(t1_ck)
        if cfg["teacher2_ckpt"]:
            t2_ck = torch.load(
                cfg["teacher2_ckpt"], map_location=device, weights_only=True
            )
            teacher2.load_state_dict(t2_ck)

        # 4) MBM and synergy head
        mbm_query_dim = cfg.get("mbm_query_dim")
        mbm, synergy_head = build_from_teachers(
            [teacher1, teacher2], cfg, query_dim=mbm_query_dim
        )
        mbm = mbm.to(device)
        synergy_head = synergy_head.to(device)

        # load MBM, synergy head
        if cfg["mbm_ckpt"]:
            mbm_ck = torch.load(
                cfg["mbm_ckpt"], map_location=device, weights_only=True
            )
            mbm.load_state_dict(mbm_ck)
        if cfg["head_ckpt"]:
            head_ck = torch.load(
                cfg["head_ckpt"], map_location=device, weights_only=True
            )
            synergy_head.load_state_dict(head_ck)

        # 5) student for LA MBM or optional synergy
        student_name = cfg.get("student_type", "resnet_adapter")
        student = create_student_by_name(
            student_name,
            pretrained=False,
            small_input=small_input,
            num_classes=n_classes,
        ).to(device)

        if cfg.get("student_ckpt"):
            s_ck = torch.load(
                cfg["student_ckpt"], map_location=device, weights_only=True
            )
            if "model_state" in s_ck:
                student.load_state_dict(s_ck["model_state"])
            else:
                student.load_state_dict(s_ck)

        # synergy ensemble
        synergy_model = SynergyEnsemble(
            teacher1,
            teacher2,
            mbm,
            synergy_head,
            student=student,
            cfg=cfg,
        ).to(device)

        # evaluate
        train_acc = evaluate_acc(synergy_model, train_loader, device, cfg)
        test_acc  = evaluate_acc(synergy_model, test_loader, device, cfg)
        print(f"[Synergy] Train={train_acc:.2f}, Test={test_acc:.2f}")

        logger.update_metric("eval_mode", "synergy")
        logger.update_metric("train_acc", train_acc)
        logger.update_metric("test_acc", test_acc)

    logger.finalize()

if __name__ == "__main__":
    main()
