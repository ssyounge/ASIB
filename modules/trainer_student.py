# modules/trainer_student.py

import torch
import torch.nn.functional as F
import copy
from utils.progress import smart_tqdm
from models.la_mbm import LightweightAttnMBM
from modules.ib_mbm import IB_MBM

from modules.losses import kd_loss_fn, ce_loss_fn, ib_loss
from modules.disagreement import sample_weights_from_disagreement
from utils.misc import mixup_data, cutmix_data, mixup_criterion, get_amp_components
from utils.schedule import get_tau

def student_distillation_update(
    teacher_wrappers,
    mbm, synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler,
    global_ep: int = 0
):
    """Train the student model via knowledge distillation.

    The teachers and MBM are frozen to generate synergy logits while the
    student is optimized using a combination of cross-entropy and KD losses.
    When a :class:`LightweightAttnMBM` is used, the optional feature-level KD
    term aligns the student query with the teacher attention output in the
    MBM latent space ("latent space alignment").
    """
    # 1) freeze teacher + mbm
    teacher_reqgrad_states = []
    teacher_train_states = []
    for tw in teacher_wrappers:
        teacher_train_states.append(tw.training)
        states = []
        for p in tw.parameters():
            states.append(p.requires_grad)
            p.requires_grad = False
        teacher_reqgrad_states.append(states)
        tw.eval()

    mbm_reqgrad_states = []
    mbm_train_state = mbm.training
    for p in mbm.parameters():
        mbm_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    mbm.eval()

    syn_reqgrad_states = []
    syn_train_state = synergy_head.training
    for p in synergy_head.parameters():
        syn_reqgrad_states.append(p.requires_grad)
        p.requires_grad = False
    synergy_head.eval()

    student_epochs = cfg.get("student_iters", cfg.get("student_epochs_per_stage", 15))

    best_acc = 0.0
    best_state = copy.deepcopy(student_model.state_dict())

    logger.info(f"[StudentDistill] Using student_epochs={student_epochs}")

    autocast_ctx, scaler = get_amp_components(cfg)
    # ---------------------------------------------------------
    # MBM type check: LA or IB (query required) vs. baseline MLP
    # ---------------------------------------------------------
    la_mode = isinstance(mbm, (LightweightAttnMBM, IB_MBM)) \
              or cfg.get("mbm_type", "").lower() in ("la", "ib_mbm")
    for ep in range(student_epochs):
        cur_tau = get_tau(cfg, global_ep + ep)
        distill_loss_sum = 0.0
        cnt = 0
        feat_kd_sum = 0.0
        student_model.train()
        feat_kd_warned = False

        mix_mode = (
            "cutmix"
            if cfg.get("cutmix_alpha_distill", 0.0) > 0.0
            else "mixup" if cfg.get("mixup_alpha", 0.0) > 0.0
            else "none"
        )

        attn_sum = 0.0
        for x, y in smart_tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]"):
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            if mix_mode == "cutmix":
                x_mixed, y_a, y_b, lam = cutmix_data(
                    x, y, alpha=cfg["cutmix_alpha_distill"]
                )
            elif mix_mode == "mixup":
                x_mixed, y_a, y_b, lam = mixup_data(x, y, alpha=cfg["mixup_alpha"])
            else:
                x_mixed, y_a, y_b, lam = x, y, y, 1.0

            with autocast_ctx:
                # (A) Student forward (query)
                feat_dict, s_logit, _ = student_model(x_mixed)

                with torch.no_grad():
                    t1_dict = teacher_wrappers[0](x_mixed)
                    t2_dict = teacher_wrappers[1](x_mixed)

                    feat_key = (
                        "distill_feat"
                        if cfg.get("use_distillation_adapter", False)
                        else "feat_2d"
                    )
                    f1_2d = t1_dict[feat_key]
                    f2_2d = t2_dict[feat_key]
                    f1_4d = t1_dict.get("feat_4d")
                    f2_4d = t2_dict.get("feat_4d")

                if la_mode:
                    s_feat = feat_dict[cfg.get("feat_kd_key", "feat_2d")]
                    if isinstance(mbm, IB_MBM):
                        # IB-MBM returns z, mu, logvar
                        syn_feat, mu, logvar = mbm(
                            s_feat, torch.stack([f1_2d, f2_2d], dim=1)
                        )
                        attn = None
                        # optional IB loss
                        if cfg.get("use_ib", False):
                            ib_beta = cfg.get("ib_beta", 1e-2)
                            ib_loss_val = ib_loss(
                                syn_feat, mu, logvar, y, decoder=synergy_head, beta=ib_beta
                            )
                        else:
                            ib_loss_val = torch.tensor(0.0, device=cfg["device"])
                    else:  # LA MBM
                        syn_feat, attn, student_q_proj, teacher_attn_out = mbm(
                            s_feat, [f1_2d, f2_2d]
                        )
                        ib_loss_val = torch.tensor(0.0, device=cfg["device"])
                    fsyn = syn_feat
                    attn_sum += (attn.mean().item() if attn is not None else 0.0) * x.size(0)
                else:
                    fsyn = mbm([f1_2d, f2_2d], [f1_4d, f2_4d])
                    ib_loss_val = torch.tensor(0.0, device=cfg["device"])
                zsyn = synergy_head(fsyn)

                if mix_mode != "none":
                    ce_obj = lambda pred, target: ce_loss_fn(
                        pred,
                        target,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                        reduction="none",
                    )
                    ce_vec = mixup_criterion(ce_obj, s_logit, y_a, y_b, lam)
                else:
                    ce_vec = ce_loss_fn(
                        s_logit,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                        reduction="none",
                    )
                kd_vec = kd_loss_fn(
                    s_logit, zsyn, T=cur_tau, reduction="none"
                ).sum(dim=1)

            # (B1) disagreement-based sample weights
            if cfg.get("use_disagree_weight", False):
                weights = sample_weights_from_disagreement(
                    t1_dict["logit"],
                    t2_dict["logit"],
                    y,
                    mode=cfg.get("disagree_mode", "pred"),
                    lambda_high=cfg.get("disagree_lambda_high", 1.0),
                    lambda_low=cfg.get("disagree_lambda_low", 1.0),
                )
            else:
                weights = torch.ones_like(y, dtype=torch.float32, device=y.device)

            # apply sample weights to CE and KD losses computed above
            ce_loss_val = (weights * ce_vec).mean()
            kd_loss_val = (weights * kd_vec).mean()

            feat_kd_val = torch.tensor(0.0, device=cfg["device"])
            if cfg.get("feat_kd_alpha", 0) > 0:
                if la_mode and not isinstance(mbm, IB_MBM):
                    feat_kd_val = F.mse_loss(student_q_proj, teacher_attn_out.detach())
                else:
                    key = cfg.get("feat_kd_key", "feat_2d")

                    s_feat = feat_dict[key]
                    fsyn_use = fsyn

                    if fsyn_use.dim() == 4 and s_feat.dim() == 2:
                        fsyn_use = torch.nn.functional.adaptive_avg_pool2d(
                            fsyn_use, (1, 1)
                        ).flatten(1)

                    if cfg.get("feat_kd_norm", "none") == "l2":
                        s_feat = torch.nn.functional.normalize(
                            s_feat.view(s_feat.size(0), -1), dim=1
                        )
                        fsyn_use = torch.nn.functional.normalize(
                            fsyn_use.view(fsyn_use.size(0), -1), dim=1
                        )

                    s_flat = s_feat.view(s_feat.size(0), -1)
                    f_flat = fsyn_use.detach().view(fsyn_use.size(0), -1)
                    if s_flat.size(1) == f_flat.size(1):
                        feat_kd_val = torch.nn.functional.mse_loss(s_flat, f_flat)
                    else:
                        if not feat_kd_warned:
                            logger.info(
                                f"[StudentDistill] skip feat KD: s_feat={s_flat.size(1)}"
                                f" vs fsyn={f_flat.size(1)}"
                            )
                            feat_kd_warned = True

            loss = (
                cfg["ce_alpha"] * ce_loss_val
                + cfg["kd_alpha"] * kd_loss_val
                + cfg.get("feat_kd_alpha", 0.0) * feat_kd_val
                + ib_loss_val
            )

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), cfg["grad_clip_norm"]
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        student_model.parameters(), cfg["grad_clip_norm"]
                    )
                optimizer.step()

            bs = x.size(0)
            distill_loss_sum += loss.item()*bs
            feat_kd_sum += feat_kd_val.item() * bs
            cnt += bs

        ep_loss = distill_loss_sum / cnt
        avg_feat_kd = feat_kd_sum / cnt if cnt > 0 else 0.0
        attn_avg = attn_sum / cnt if la_mode and cnt > 0 else 0.0

        # (C) validate
        test_acc = eval_student(student_model, testloader, cfg["device"], cfg)

        logger.info(f"[StudentDistill ep={ep+1}] loss={ep_loss:.4f}, testAcc={test_acc:.2f}, best={best_acc:.2f}")

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"student_ep{ep+1}_acc", test_acc)
        logger.update_metric(f"student_ep{ep+1}_loss", ep_loss)
        if la_mode:
            logger.update_metric(f"student_ep{ep+1}_attn", attn_avg)
        logger.update_metric(f"ep{ep+1}_feat_kd", avg_feat_kd)
        logger.update_metric(f"ep{ep+1}_mix_mode", mix_mode)
        logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)

        if scheduler is not None:
            scheduler.step()

        # (E) best snapshot
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(student_model.state_dict())

    student_model.load_state_dict(best_state)

    # restore original requires_grad and training states
    for tw, states, train_flag in zip(teacher_wrappers, teacher_reqgrad_states, teacher_train_states):
        for p, rg in zip(tw.parameters(), states):
            p.requires_grad = rg
        tw.train(train_flag)
    for p, rg in zip(mbm.parameters(), mbm_reqgrad_states):
        p.requires_grad = rg
    mbm.train(mbm_train_state)
    for p, rg in zip(synergy_head.parameters(), syn_reqgrad_states):
        p.requires_grad = rg
    synergy_head.train(syn_train_state)

    logger.info(f"[StudentDistill] bestAcc={best_acc:.2f}")
    return best_acc

@torch.no_grad()
def eval_student(model, loader, device, cfg=None):
    autocast_ctx, _ = get_amp_components(cfg or {})
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            feat_dict, s_logit, _ = model(x)
        pred = s_logit.argmax(dim=1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return 100.*correct/total
