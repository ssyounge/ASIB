# modules/trainer_student.py

import torch
import copy
import logging
from utils.progress import smart_tqdm
from models.mbm import IB_MBM

from modules.losses import (
    kd_loss_fn, ce_loss_fn, ib_loss, certainty_weights
)
from modules.disagreement import sample_weights_from_disagreement
from utils.misc import mixup_data, cutmix_data, mixup_criterion, get_amp_components
from utils.schedule import get_tau, get_beta

try:
    import wandb
except ModuleNotFoundError:
    wandb = None

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
    # query-based MBM? (only IB_MBM remains)
    # ---------------------------------------------------------
    query_mode = isinstance(mbm, IB_MBM) \
                 or cfg.get("mbm_type", "").lower() == "ib_mbm"
    for ep in range(student_epochs):
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
            cur_tau = get_tau(cfg, min(global_ep + ep, total_epochs - 1))
        else:
            cur_tau = get_tau(cfg, global_ep + ep)
        distill_loss_sum = 0.0
        cnt = 0
        feat_kd_sum = 0.0
        student_model.train()

        mix_mode = (
            "cutmix"
            if cfg.get("cutmix_alpha_distill", 0.0) > 0.0
            else "mixup" if cfg.get("mixup_alpha", 0.0) > 0.0
            else "none"
        )

        attn_sum = 0.0
        for step, (x, y) in enumerate(
            smart_tqdm(trainloader, desc=f"[StudentDistill ep={ep+1}]")
        ):
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

                    feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) \
                               else "feat_2d"
                    f1_2d = t1_dict[feat_key]
                    f2_2d = t2_dict[feat_key]
                    f1_4d = t1_dict.get("feat_4d")
                    f2_4d = t2_dict.get("feat_4d")

                if query_mode:
                    s_feat = feat_dict[cfg.get("feat_kd_key", "feat_2d")]
                    if isinstance(mbm, IB_MBM):
                        # IB-MBM returns z, mu, logvar
                        syn_feat, mu, logvar = mbm(
                            s_feat, torch.stack([f1_2d, f2_2d], dim=1)
                        )
                        attn = None
                        # optional IB loss
                        if cfg.get("use_ib", False):
                            ib_beta = get_beta(cfg, global_ep + ep)
                            mu, logvar = mu.float(), logvar.float()
                            ib_loss_val = ib_loss(mu, logvar, beta=ib_beta)
                        else:
                            ib_loss_val = torch.tensor(0.0, device=cfg["device"])
                    fsyn = syn_feat
                    if "attn" in locals() and attn is not None:
                        attn_sum += attn.mean().item() * x.size(0)
                else:
                    fsyn = mbm([f1_2d, f2_2d], [f1_4d, f2_4d])
                    ib_loss_val = torch.tensor(0.0, device=cfg["device"])
                zsyn = synergy_head(fsyn)

                if mix_mode != "none":
                    def ce_obj(pred, target):
                        return ce_loss_fn(
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

                # ---- DEBUG: 첫 batch 모양 확인 ----
                if ep == 0 and step == 0:
                    print(
                        "[DBG/student] x",
                        tuple(x.shape),
                        "s_logit",
                        tuple(s_logit.shape),
                        "zsyn",
                        tuple(zsyn.shape),
                    )

            # ── (B1) sample-weights (always same dtype as losses) ───────────
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
                weights = torch.ones_like(y, dtype=s_logit.dtype, device=y.device)

            weights = weights.to(s_logit.dtype)

            if cfg.get("use_ib", False) and isinstance(mbm, IB_MBM):
                cw = certainty_weights(logvar).mean(dim=1).to(s_logit.dtype)
                weights = weights * cw

            # apply sample weights to CE and KD losses computed above
            ce_loss_val = (weights * ce_vec).mean()
            kd_loss_val = (weights * kd_vec).mean()

            feat_kd_val = torch.tensor(0.0, device=cfg["device"])
            if cfg.get("feat_kd_alpha", 0) > 0:
                key = cfg.get("feat_kd_key", "feat_2d")
                s_feat = feat_dict[key]
                fsyn_use = fsyn
                if fsyn_use.dim() == 4 and s_feat.dim() == 2:
                    fsyn_use = torch.nn.functional.adaptive_avg_pool2d(
                        fsyn_use, (1, 1)
                    ).flatten(1)

                feat_kd_val = feat_mse_loss(
                    s_feat, fsyn_use,
                    norm=cfg.get("feat_kd_norm", "none")
                )

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
                print(f"[StudentDistill] batch loss={loss.item():.4f}")
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
        attn_avg = attn_sum / cnt if query_mode and cnt > 0 else 0.0

        # (C) validate
        test_acc = eval_student(student_model, testloader, cfg["device"], cfg)

        logging.info(
            "[StudentDistill ep=%d] loss=%.4f testAcc=%.2f best=%.2f",
            ep + 1,
            ep_loss,
            test_acc,
            best_acc,
        )
        if wandb and wandb.run:
            wandb.log({
                "student/loss": ep_loss,
                "student/acc": test_acc,
                "student/epoch": global_ep + ep + 1,
            })

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"student_ep{ep+1}_acc", test_acc)
        logger.update_metric(f"student_ep{ep+1}_loss", ep_loss)
        if query_mode:
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
