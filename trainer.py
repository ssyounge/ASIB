# trainer.py

import os
import torch
import torch.nn.functional as F
from utils.schedule import cosine_lr_scheduler
from utils.misc import get_amp_components
from utils.eval import evaluate_acc


def simple_finetune(
    model,
    loader,
    lr,
    epochs,
    device,
    weight_decay=0.0,
    cfg=None,
    ckpt_path="finetuned_best.pth",
):
    """Fine-tune ``model`` using cross-entropy loss with basic reporting.

    The best model (by training accuracy) is saved to ``ckpt_path`` whenever
    improved. After all epochs finish, the final state is written to a
    ``*_last.pth`` file.
    """
    if epochs <= 0:
        return

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=float(weight_decay)
    )
    autocast_ctx, scaler = get_amp_components(cfg or {})
    criterion = torch.nn.CrossEntropyLoss()

    eval_loader = loader
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        running_loss = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast_ctx:
                out = model(x)
                logit = out[1] if isinstance(out, tuple) else out
                loss = criterion(logit, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            count += x.size(0)

        acc = evaluate_acc(model, eval_loader, device=device)
        avg_loss = running_loss / max(count, 1)
        # return to train mode after evaluation
        model.train()

        tag = ""
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            tag = "\u2605 best"

        print(
            f"[FineTune] ep {ep:03d}/{epochs}  loss {avg_loss:.4f}  acc {acc:.2f}%  best {best_acc:.2f}% {tag}"
        )

    last_path = ckpt_path.replace(".pth", "_last.pth")
    torch.save(model.state_dict(), last_path)
    print(
        f"[FineTune] done \u2192 best={best_acc:.2f}% ({ckpt_path}), last={last_path}"
    )
    model.train()


def teacher_vib_update(teacher1, teacher2, vib_mbm, loader, cfg, optimizer):
    """Train the VIB module using frozen teachers.

    Args:
        teacher1: First teacher network used for feature extraction.
        teacher2: Second teacher network used for feature extraction.
        vib_mbm: Bottleneck module to update.
        loader: Data loader providing input images and labels.
        cfg: Configuration dictionary with training options.
        optimizer: Optimizer for ``vib_mbm`` parameters.

    Returns:
        None.
    """
    device = cfg.get("device", "cuda")
    beta = cfg.get("beta_bottleneck", 0.003)
    clip = cfg.get("grad_clip_norm", 0)
    autocast_ctx, scaler = get_amp_components(cfg)
    vib_mbm.train()
    teacher1.eval()
    teacher2.eval()
    scheduler = cosine_lr_scheduler(optimizer, cfg.get("teacher_iters", 1))
    for ep in range(cfg.get("teacher_iters", 1)):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out1 = teacher1(x)
                out2 = teacher2(x)
                t1_dict = out1[0] if isinstance(out1, tuple) else out1
                t2_dict = out2[0] if isinstance(out2, tuple) else out2
            f1 = t1_dict["feat_2d"]
            f2 = t2_dict["feat_2d"]
            with autocast_ctx:
                z, logit_syn, kl_z, _ = vib_mbm(
                    f1,
                    f2,
                    log_kl=cfg.get("log_kl", False),
                )
                loss = F.cross_entropy(logit_syn, y) + beta * kl_z.mean()
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(vib_mbm.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(vib_mbm.parameters(), clip)
                optimizer.step()
        scheduler.step()


def student_vib_update(teacher1, teacher2, student_model, vib_mbm, student_proj, loader, cfg, optimizer):
    """Update the student network to mimic the VIB representation.

    Args:
        teacher1: First teacher network providing target features.
        teacher2: Second teacher network providing target features.
        student_model: Student model being trained.
        vib_mbm: Pre-trained VIB module used to generate targets.
        student_proj: Projection head mapping student features to the latent space.
        loader: Data loader supplying input images and labels.
        cfg: Configuration dictionary with training options.
        optimizer: Optimizer for student parameters.

    Returns:
        None.
    """
    device = cfg.get("device", "cuda")
    alpha = cfg.get("alpha_kd", 0.7)
    ce_alpha = cfg.get("ce_alpha", 1.0)
    clip = cfg.get("grad_clip_norm", 0)
    autocast_ctx, scaler = get_amp_components(cfg)
    vib_mbm.eval()
    student_model.train()
    scheduler = cosine_lr_scheduler(optimizer, cfg.get("student_iters", 1))
    for ep in range(cfg.get("student_iters", 1)):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out1 = teacher1(x)
                out2 = teacher2(x)
                t1_dict = out1[0] if isinstance(out1, tuple) else out1
                t2_dict = out2[0] if isinstance(out2, tuple) else out2
                f1 = t1_dict["feat_2d"]
                f2 = t2_dict["feat_2d"]
                _, _, _, mu = vib_mbm(
                    f1,
                    f2,
                    log_kl=cfg.get("log_kl", False),
                )
                z_target = mu.detach()
            with autocast_ctx:
                feat_dict, s_logit, _ = student_model(x)
                s_feat = feat_dict["feat_2d"]
                z_pred = student_proj(s_feat)
                kd = F.mse_loss(z_pred, z_target)
                loss = ce_alpha * F.cross_entropy(s_logit, y) + alpha * kd
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(student_model.parameters()) + list(student_proj.parameters()),
                        clip,
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(student_model.parameters()) + list(student_proj.parameters()),
                        clip,
                    )
                optimizer.step()
        scheduler.step()


