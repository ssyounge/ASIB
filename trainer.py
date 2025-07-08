# trainer.py

import os
import copy

import torch
import torch.nn.functional as F   # loss 함수(F.cross_entropy 등)용
from tqdm.auto import tqdm

from utils.model_factory import create_student_by_name   # fallback 생성용
from utils.schedule import cosine_lr_scheduler
from utils.misc import get_amp_components, mixup_data, mixup_criterion
from utils.eval import evaluate_acc
from utils.distill_loss import feat_mse_pair
from modules.losses import compute_vib_loss


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
    # warm-up + cosine
    scheduler = cosine_lr_scheduler(
        optimizer,
        epochs,
        warmup_epochs=(cfg or {}).get("finetune_warmup", 0),
        min_lr_ratio=(cfg or {}).get("min_lr_ratio_finetune", 0.1),
    )
    autocast_ctx, scaler = get_amp_components(cfg or {})

    mixup_alpha = (cfg or {}).get("finetune_mixup_alpha", 0.0)
    label_smooth = (cfg or {}).get("finetune_label_smoothing", 0.0)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # if caller supplies a separate validation loader, use it
    eval_loader = (cfg or {}).get("finetune_eval_loader", loader)
    best_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if mixup_alpha > 0.0:
                x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
            optimizer.zero_grad()
            with autocast_ctx:
                out = model(x)
                logit_s = out[1] if isinstance(out, tuple) else out

                if mixup_alpha > 0.0:
                    ce = mixup_criterion(criterion, logit_s, y_a, y_b, lam)
                else:
                    ce = criterion(logit_s, y)

                if (cfg or {}).get("mbm_type") == "VIB":
                    z = model.get_latent() if hasattr(model, "get_latent") else None
                    if z is not None:
                        vib_loss = compute_vib_loss(z)
                        loss = ce + (cfg or {}).get("latent_alpha", 1.0) * vib_loss
                    else:
                        loss = ce
                else:
                    loss = ce
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

        if scheduler is not None:
            scheduler.step()

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
    clip = cfg.get("grad_clip_norm", 0)
    autocast_ctx, scaler = get_amp_components(cfg)
    vib_mbm.train()
    teacher1.eval()
    teacher2.eval()
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
        for batch_idx, (x, y) in enumerate(epoch_loader):
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
                kl = kl_z.mean() if kl_z.dim() > 0 else kl_z
                loss = F.cross_entropy(logit_syn, y) + kl
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
            running_kl += kl.item() * x.size(0)
            correct += (logit_syn.argmax(1) == y).sum().item()
            count += x.size(0)
        avg_loss = running_loss / max(count, 1)
        avg_kl = running_kl / max(count, 1)
        train_acc = 100.0 * correct / max(count, 1)
        test_acc = 0.0
        if test_loader is not None:
            from utils.eval import evaluate_mbm_acc

            test_acc = evaluate_mbm_acc(teacher1, teacher2, vib_mbm, test_loader, device)
            vib_mbm.train()        # ← 평가‑모드 해제
        msg = (
            f"[Teacher] ep {ep + 1:03d}/{cfg.get('teacher_iters', 1)} "
            f"loss {avg_loss:.4f} kl {avg_kl:.4f} train_acc {train_acc:.2f}%  "
            f"test_acc {test_acc:.2f}%"
        )
        print(msg)
        if logger is not None:
            logger.update_metric(f"teacher_ep{ep + 1}_train_acc", float(train_acc))
            logger.update_metric(f"teacher_ep{ep + 1}_test_acc",    float(test_acc))

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
            vib_mbm.train()
            print(f"[DEBUG] synergy_acc_after_ep1: {dbg_acc:.2f}%")


def student_vib_update(
    teacher1,
    teacher2,
    student_model,
    vib_mbm,
    student_proj,
    loader,
    cfg,
    optimizer,
    test_loader=None,
    logger=None,
    scheduler=None,
    cur_classes=None,
):
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
        cur_classes: Optional sequence of class indices to slice student logits
            before computing distillation loss.

    Returns:
        None.
    """
    device = cfg.get("device", "cuda")
    best_acc = 0.0
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_best = os.path.join(ckpt_dir, "student_best.pth")
    ckpt_last = os.path.join(ckpt_dir, "student_last.pth")
    # ────── KD 스케줄 파라미터 ────────────────────────────────
    init_alpha = cfg.get("kd_alpha_init", cfg.get("alpha_kd", 0.5))
    final_alpha = cfg.get("kd_alpha_final", 0.3)
    init_T     = cfg.get("kd_T_init",   cfg.get("kd_temperature", 4))
    final_T    = cfg.get("kd_T_final",  3)
    warmup     = cfg.get("kd_warmup_frac", 0.0)
    gran       = cfg.get("kd_schedule_granularity", "step").lower()
    sched_pow  = cfg.get("kd_sched_pow", 1.0)

    alpha_kd = init_alpha
    T = init_T
    ce_alpha = cfg.get("ce_alpha", 1.0)       #  CE 가중치
    latent_w = cfg.get("latent_alpha", 1.0)   #  잠재 정렬
    latent_mse_weight = cfg.get("latent_mse_weight", 0.7)
    latent_angle_weight = cfg.get("latent_angle_weight", 0.3)
    clip = cfg.get("grad_clip_norm", 0)
    autocast_ctx, scaler = get_amp_components(cfg)

    # ───── MixUp / CutMix 설정 ─────
    mix_alpha = cfg.get("mixup_alpha", 0.0)
    cutmix_alpha = cfg.get("cutmix_alpha_distill", 0.0)
    do_mix = (mix_alpha > 0) or (cutmix_alpha > 0)

    vib_mbm.eval()
    student_model.train()
    ema_model = None
    total_epochs = cfg.get("student_iters", 1)
    if scheduler is None:
        scheduler = cosine_lr_scheduler(
            optimizer,
            total_epochs,
            warmup_epochs=cfg.get("student_warmup_epochs", 3),
            min_lr_ratio=cfg.get("min_lr_ratio_student", 0.05),
        )

    # 총 업데이트 횟수 (스케줄 계산용)
    total_steps  = total_epochs * len(loader)

    # --- Feature hook 세팅 ---------------------------
    from utils.feature_hook import FeatHook
    from utils.distill_loss import feat_mse_pair

    layer_ids  = cfg.get("feat_layers", [1, 2])      # ex) [1,2]
    layer_w    = cfg.get("feat_weights", [0.5, 0.5]) # 합 = 1

    _gamma_cfg = cfg.get("feat_loss_weight", 1.0)
    if isinstance(_gamma_cfg, (list, tuple)):
        gamma_schedule = list(_gamma_cfg)
    else:
        gamma_schedule = None
        gamma_feat = float(_gamma_cfg)

    hook_s  = FeatHook(student_model.backbone, layer_ids)
    hook_t1 = FeatHook(teacher1.backbone, layer_ids)
    hook_t2 = FeatHook(teacher2.backbone, layer_ids)

    for ep in range(total_epochs):
        running_loss = 0.0
        correct = 0
        count = 0
        epoch_loader = tqdm(
            loader,
            desc=f"[Student] epoch {ep + 1}",
            leave=False,
            disable=cfg.get("disable_tqdm", False),
        )
        for batch_idx, (x, y) in enumerate(epoch_loader):
            x, y = x.to(device), y.to(device)

            if do_mix:
                lam_alpha = mix_alpha if mix_alpha > 0 else cutmix_alpha
                x, y_a, y_b, lam = mixup_data(x, y, alpha=lam_alpha)

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

            # ─ KD 스케줄 (progress ∈ [0,1]) ────────────────────
            if gran == "epoch":
                raw_prog = ep / max(total_epochs - 1, 1)
            else:  # "step"
                global_step = ep * len(loader) + batch_idx
                raw_prog = global_step / max(total_steps - 1, 1)

            # warm‑up 구간 제외 후, p‑power 스케일 적용
            prog = max(0.0, raw_prog - warmup) / max(1e-6, 1.0 - warmup)
            prog_p = prog ** sched_pow

            alpha_kd = init_alpha * (1 - prog_p) + final_alpha * prog_p
            T        = init_T     * (1 - prog_p) + final_T     * prog_p

            # ─────────────── DEBUG: 스케줄 값 모니터링 ───────────────
            if batch_idx == 0 and ep in {0, 5, 10, 20, total_epochs - 1}:
                print(
                    f"[KD-sched] ep{ep:02d} prog={prog_p:.2f}, "
                    f"kd={alpha_kd:.3f}, T={T:.2f}, latent_w={latent_w}"
                )

            # ─ Losses ──────────────────────────────────────────
            if do_mix:
                ce = mixup_criterion(F.cross_entropy, logit_s, y_a, y_b, lam)
            else:
                ce = F.cross_entropy(logit_s, y)
            logit_kd = logit_s
            if cur_classes is not None:
                logit_kd = logit_s[:, cur_classes]

            kd = F.kl_div(
                F.log_softmax(logit_kd / T, dim=1),
                F.softmax(logit_t.detach() / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            # ─ Latent & Angle Loss 병행 ─
            latent_mse   = F.mse_loss(z_s, z_t.detach())
            latent_angle = 1 - F.cosine_similarity(z_s, z_t.detach(), dim=1).mean()
            latent       = latent_mse_weight * latent_mse + latent_angle_weight * latent_angle

            feat_loss = feat_mse_pair(
                hook_s.features,
                hook_t1.features,
                hook_t2.features,
                layer_ids,
                layer_w,
            )

            if gamma_schedule is not None:
                # 3‑단계 스케줄 (구간 길이가 같지 않아도 OK)
                seg = len(gamma_schedule)
                cur_seg = int(ep / (total_epochs / seg))
                gamma_feat = gamma_schedule[min(cur_seg, seg - 1)]
            loss = (
                ce_alpha*ce + alpha_kd*kd + latent_w*latent
                + gamma_feat*feat_loss
            )

            if batch_idx == 0 and ep % 10 == 0:
                print(f"[DEBUG] γ={gamma_feat:.3f}  feat_loss={feat_loss.item():.4f}")
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
            hook_s.clear(); hook_t1.clear(); hook_t2.clear()
            running_loss += loss.item() * x.size(0)
            correct += (logit_s.argmax(1) == y).sum().item()
            count += x.size(0)
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
                # warm-up: 초기에는 빠르게, 점점 느리게
                base = cfg.get("ema_decay", 0.995)     # 최종 목표치
                warm = cfg.get("ema_warmup_iters", 5)  # 앞 N epoch
                init = cfg.get("ema_initial_decay", 0.90)  # 초기 decay
                cur  = min(ep, warm) / warm
                d = base * cur + (1 - cur) * init      # init → base 로 선형 전환
                for p_ema, p in zip(ema_model.parameters(),
                                    student_model.parameters()):
                    p_ema.data.mul_(d).add_(p.data, alpha=1 - d)
                # BN running_mean / var 동기화
                for b_ema, b in zip(ema_model.buffers(),
                                    student_model.buffers()):
                    if b_ema.dtype == torch.float32:
                        b_ema.copy_(b)

        # ─ 테스트 정확도 ────────────────────────────
        # ───────── epoch‑end 평가 ─────────
        student_acc = 0.0
        ema_acc = None
        if test_loader is not None:
            student_acc = evaluate_acc(student_model, test_loader, device=device)
            if ema_model is not None:
                ema_acc = evaluate_acc(ema_model, test_loader, device=device)

        msg = (
            f"[Student] ep {ep + 1:03d}/{cfg.get('student_iters', 1)} "
            f"loss {avg_loss:.4f} train_acc {train_acc:.2f}%  "
            f"test_acc {student_acc:.2f}%"
        )
        if ema_acc is not None:
            msg += f"  ema_acc {ema_acc:.2f}%"
        print(msg)

        if scheduler is not None:
            scheduler.step()

        # --------- Check-pointing ----------
        if student_acc > best_acc:
            best_acc = student_acc
            torch.save(student_model.state_dict(), ckpt_best)
            if logger is not None:
                logger.info(f"[CKPT] ↑ best student acc={best_acc:.2f}%  → {ckpt_best}")

        # epoch 끝날 때마다 최신 가중치 덮어쓰기
        torch.save(student_model.state_dict(), ckpt_last)

        if logger is not None:
            logger.update_metric(
                f"student_ep{ep + 1}_train_acc", float(train_acc)
            )
            logger.update_metric(
                f"student_ep{ep + 1}_test_acc",  # ← 에폭 구분용 별도 키
                float(student_acc),
                step=ep + 1,
            )
            if ema_acc is not None:
                logger.update_metric("ema_acc", float(ema_acc), step=ep + 1)
            logger.update_metric("lr", optimizer.param_groups[0]["lr"])

# ─ 최종 EMA 성능 저장 ──────────────────────────────
    if cfg.get("use_ema", False) and test_loader is not None:
        final_ema_acc = evaluate_acc(
            ema_model, test_loader, device=device,
            mixup_active=(cfg.get("mixup_alpha", 0) > 0 or cfg.get("cutmix_alpha_distill", 0) > 0),
        )
        logger.update_metric("final_test_acc", float(final_ema_acc))
        print(f"Final student EMA accuracy: {final_ema_acc:.2f}%")

    hook_s.close(); hook_t1.close(); hook_t2.close()


