# trainer.py

import os
import copy, torch
from utils.model_factory import create_student_by_name   # fallback 생성용
import torch.nn.functional as F   # loss 함수(F.cross_entropy 등)용
from utils.schedule import cosine_lr_scheduler
from utils.misc import get_amp_components
from utils.eval import evaluate_acc
from tqdm.auto import tqdm


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

    If ``ckpt_path`` already exists, the saved weights are loaded and
    fine-tuning is skipped. Pass ``overwrite`` via ``cfg`` to ignore the
    existing checkpoint.
    """
    if os.path.exists(ckpt_path) and not (cfg or {}).get("overwrite", False):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"[FineTune] loaded checkpoint → {ckpt_path}")
        model.train()
        return

    if epochs <= 0:
        return

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    autocast_ctx, scaler = get_amp_components(cfg or {})
    criterion = torch.nn.CrossEntropyLoss()

    eval_loader = loader
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
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
        model.train()
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


def teacher_vib_update(teacher1, teacher2, vib_mbm, loader, cfg, optimizer, test_loader=None, logger=None):
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
        running_loss = 0.0
        running_kl = 0.0
        correct = 0
        count = 0
        epoch_loader = tqdm(
            loader,
            desc=f"[Teacher] epoch {ep + 1}",
            leave=False,
            disable=cfg.get("disable_tqdm", False),
        )
        for x, y in epoch_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                out1 = teacher1(x)
                out2 = teacher2(x)
                t1_dict = out1[0] if isinstance(out1, tuple) else out1
                t2_dict = out2[0] if isinstance(out2, tuple) else out2

                # ────────── DEBUG ② feature key 확인 ──────────
                assert "feat_2d" in t1_dict and "feat_2d" in t2_dict, (
                    "feat_2d key not found in teacher outputs"
                )
                feat1, feat2 = t1_dict["feat_2d"], t2_dict["feat_2d"]
                assert (
                    feat1.shape[0] == x.size(0) and feat2.shape[0] == x.size(0)
                ), "feature batch size mismatch"

            f1 = feat1
            f2 = feat2
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
            running_loss += loss.item() * x.size(0)
            running_kl += kl_z.mean().item() * x.size(0)
            correct += (logit_syn.argmax(1) == y).sum().item()
            count += x.size(0)
        scheduler.step()
        avg_loss = running_loss / max(count, 1)
        avg_kl = running_kl / max(count, 1)
        train_acc = 100.0 * correct / max(count, 1)
        test_acc = 0.0
        if test_loader is not None:
            from utils.eval import evaluate_mbm_acc

            test_acc = evaluate_mbm_acc(teacher1, teacher2, vib_mbm, test_loader, device)
        print(
            f"[Teacher] ep {ep + 1:03d}/{cfg.get('teacher_iters', 1)} "
            f"loss {avg_loss:.4f} kl {avg_kl:.4f} train_acc {train_acc:.2f}% test_acc {test_acc:.2f}%"
        )
        if logger is not None:
            logger.update_metric(f"teacher_ep{ep + 1}_train_acc", float(train_acc))
            logger.update_metric(f"teacher_ep{ep + 1}_test_acc", float(test_acc))

        # ────────── DEBUG ④ synergy acc 첫 epoch 후 한 번 출력 ──────────
        if ep == 0:
            from utils.eval import evaluate_mbm_acc
            dbg_acc = evaluate_mbm_acc(
                teacher1,
                teacher2,
                vib_mbm,
                test_loader,
                device=next(vib_mbm.parameters()).device,
            )
            print(f"[DEBUG] synergy_acc_after_ep1: {dbg_acc:.2f}%")


def student_vib_update(teacher1, teacher2, student_model, vib_mbm, student_proj, loader, cfg, optimizer, test_loader=None, logger=None):
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
    # ────── 하이퍼 ─────────────────────────────────────────────
    T = cfg.get("kd_temperature", 4)          #  KD 온도
    alpha_kd = cfg.get("alpha_kd", 0.5)       #  KD 가중치
    ce_alpha = cfg.get("ce_alpha", 1.0)       #  CE 가중치
    latent_w = cfg.get("latent_alpha", 1.0)   #  잠재 정렬
    clip = cfg.get("grad_clip_norm", 0)
    autocast_ctx, scaler = get_amp_components(cfg)
    vib_mbm.eval()
    student_model.train()
    ema_model = None
    scheduler = cosine_lr_scheduler(optimizer, cfg.get("student_iters", 1))
    for ep in range(cfg.get("student_iters", 1)):
        running_loss = 0.0
        correct = 0
        count = 0
        epoch_loader = tqdm(
            loader,
            desc=f"[Student] epoch {ep + 1}",
            leave=False,
            disable=cfg.get("disable_tqdm", False),
        )
        for x, y in epoch_loader:
            x, y = x.to(device), y.to(device)

            # ─ Teacher feature → synergy target ─────────────────
            with torch.no_grad():
                out1 = teacher1(x)
                out2 = teacher2(x)
                t1_dict = out1[0] if isinstance(out1, tuple) else out1
                t2_dict = out2[0] if isinstance(out2, tuple) else out2
                feat1, feat2 = t1_dict["feat_2d"], t2_dict["feat_2d"]
                z_t, logit_t, _, _ = vib_mbm(feat1, feat2)

            # ─ Student forward ─────────────────────────────────
            s_out = student_model(x)

            # ConvNeXt adapter: (feat_dict, logits, aux)  ← 3‑tuple
            if isinstance(s_out, tuple) and len(s_out) == 3:
                feat_dict, logit_s, _ = s_out
                feat_s = feat_dict["feat_2d"]
            elif isinstance(s_out, tuple) and len(s_out) == 2:
                feat_s, logit_s = s_out
            else:  # 단일 tensor 리턴 모델
                logit_s = s_out
                feat_s = student_model.get_feat()       # 필요 시 구현
            z_s = student_proj(feat_s)

            # ─ Losses ──────────────────────────────────────────
            ce = F.cross_entropy(logit_s, y)
            kd = F.kl_div(
                F.log_softmax(logit_s / T, dim=1),
                F.softmax(logit_t.detach() / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            latent = F.mse_loss(z_s, z_t.detach())

            loss = ce_alpha * ce + alpha_kd * kd + latent_w * latent
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
            running_loss += loss.item() * x.size(0)
            correct += (logit_s.argmax(1) == y).sum().item()
            count += x.size(0)
        scheduler.step()
        avg_loss = running_loss / max(count, 1)
        train_acc = 100.0 * correct / max(count, 1)
        # ─ EMA 추적 ─────────────────────────────────
        if cfg.get("use_ema", False):
            if ep == 0:                             # 초기 스냅샷
                try:
                    # 빠르고 간단하지만, weight_norm 모듈이 있을 땐 실패할 수 있음
                    ema_model = copy.deepcopy(student_model).eval()
                except RuntimeError:
                    # fallback: 새 인스턴스 생성 후 state_dict 복사
                    stype = cfg.get("student_type", "convnext_tiny")
                    n_cls = getattr(
                        student_model.backbone.classifier[2], "out_features", 100
                    )
                    ema_model = create_student_by_name(
                        stype,
                        num_classes=n_cls,
                        pretrained=False,
                        small_input=True,
                        cfg=cfg,
                    ).to(device)
                    ema_model.load_state_dict(student_model.state_dict(), strict=True)
                    ema_model.eval()
                # BN·Dropout 모두 eval 고정
                for m in ema_model.modules():
                    m.training = False
            with torch.no_grad():
                d = cfg.get("ema_decay", 0.995)     # 반응성 ↑
                for p_ema, p in zip(ema_model.parameters(),
                                    student_model.parameters()):
                    p_ema.data.mul_(d).add_(p.data, alpha=1 - d)
                # BN running_mean / var 동기화
                for b_ema, b in zip(ema_model.buffers(),
                                    student_model.buffers()):
                    if b_ema.dtype == torch.float32:
                        b_ema.copy_(b)

        # ─ 테스트 정확도 ────────────────────────────
        test_acc = 0.0
        if test_loader is not None:
            model_eval = ema_model if cfg.get("use_ema", False) else student_model
            test_acc = evaluate_acc(
                model_eval,
                test_loader,
                device=device,
                mixup_active=(cfg.get("mixup_alpha", 0) > 0 or cfg.get("cutmix_alpha_distill", 0) > 0),
            )
        print(
            f"[Student] ep {ep + 1:03d}/{cfg.get('student_iters', 1)} "
            f"loss {avg_loss:.4f} train_acc {train_acc:.2f}% test_acc {test_acc:.2f}%"
        )
        if logger is not None:
            logger.update_metric(f"student_ep{ep + 1}_train_acc", float(train_acc))
            logger.update_metric(f"student_ep{ep + 1}_test_acc", float(test_acc))

# ─ 최종 EMA 성능 저장 ──────────────────────────────
    if cfg.get("use_ema", False) and test_loader is not None:
        final_ema_acc = evaluate_acc(
            ema_model, test_loader, device=device,
            mixup_active=(cfg.get("mixup_alpha", 0) > 0 or cfg.get("cutmix_alpha_distill", 0) > 0),
        )
        logger.update_metric("test_acc", float(final_ema_acc))
        print(f"Final student EMA accuracy: {final_ema_acc:.2f}%")


