# modules/trainer_teacher.py

import torch
import copy
from utils.progress import smart_tqdm
from models.la_mbm import LightweightAttnMBM
from modules.ib_mbm import IB_MBM

from modules.losses import kd_loss_fn, ce_loss_fn, ib_loss, certainty_weights
from utils.schedule import get_tau, get_beta
from utils.misc import get_amp_components

def _cpu_state_dict(module: torch.nn.Module):
    """Return a copy of ``module.state_dict()`` on the CPU.

    Useful when saving snapshots to RAM to reduce GPU memory usage.
    """
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

@torch.no_grad()
def eval_synergy(
    teacher_wrappers,
    mbm,
    synergy_head,
    loader,
    device="cuda",
    cfg=None,
    student_model=None,
):
    """Evaluate synergy accuracy.

    When ``mbm`` is a :class:`LightweightAttnMBM`, ``student_model`` must be
    provided so that the student features can be used as the attention query.
    """

    autocast_ctx, _ = get_amp_components(cfg or {})
    la_mode = isinstance(mbm, (LightweightAttnMBM, IB_MBM)) \
              or (cfg or {}).get("mbm_type", "").lower() in ("la", "ib_mbm")
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            t1_dict = teacher_wrappers[0](x)
            t2_dict = teacher_wrappers[1](x)

            key = "distill_feat" if (cfg or {}).get("use_distillation_adapter", False) else "feat_2d"
            f1_2d = t1_dict[key]
            f2_2d = t2_dict[key]
            f1_4d = t1_dict.get("feat_4d")
            f2_4d = t2_dict.get("feat_4d")

            if la_mode:
                assert student_model is not None, "student_model required for LA MBM"
                s_feat = student_model(x)[0][cfg.get("feat_kd_key", "feat_2d")]
                if isinstance(mbm, IB_MBM):
                    fsyn, _, _ = mbm(s_feat, torch.stack([f1_2d, f2_2d], dim=1))
                else:
                    fsyn, _, _, _ = mbm(s_feat, [f1_2d, f2_2d])
            else:
                fsyn = mbm([f1_2d, f2_2d], [f1_4d, f2_4d])
            zsyn = synergy_head(fsyn)

        pred = zsyn.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

    acc = 100.0 * correct / total if total>0 else 0
    return acc

def teacher_adaptive_update(
    teacher_wrappers,
    mbm, synergy_head,
    student_model,
    trainloader,
    testloader,
    cfg,
    logger,
    optimizer,
    scheduler,
    global_ep: int = 0,
):
    """
    - ``teacher_wrappers``: list containing ``teacher1`` and ``teacher2``.
    - ``mbm`` and ``synergy_head``: assume partial freezing has been applied.
    - ``student_model``: kept fixed for knowledge distillation.
    - ``testloader``: optional loader used to evaluate synergy accuracy.
    """
    teacher_params = []
    use_da = cfg.get("use_distillation_adapter", False)
    for tw in teacher_wrappers:
        param_src = (
            tw.distillation_adapter.parameters()
            if use_da and hasattr(tw, "distillation_adapter")
            else tw.parameters()
        )
        for p in param_src:
            if p.requires_grad:
                teacher_params.append(p)
    mbm_params = [p for p in mbm.parameters() if p.requires_grad]
    syn_params = [p for p in synergy_head.parameters() if p.requires_grad]

    teacher_epochs = cfg.get("teacher_iters", cfg.get("teacher_adapt_epochs", 5))


    best_synergy = -1
    best_state = {
        "teacher_wraps": [_cpu_state_dict(tw) for tw in teacher_wrappers],
        "mbm": _cpu_state_dict(mbm),
        "syn_head": _cpu_state_dict(synergy_head),
    }

    logger.info(f"[TeacherAdaptive] Using teacher_epochs={teacher_epochs}")

    autocast_ctx, scaler = get_amp_components(cfg)
    la_mode = isinstance(mbm, (LightweightAttnMBM, IB_MBM)) \
              or cfg.get("mbm_type", "").lower() in ("la", "ib_mbm")
    for ep in range(teacher_epochs):
        for tw in teacher_wrappers:
            tw.train()
        mbm.train()
        synergy_head.train()
        if student_model is not None:
            student_model.eval()
        cur_tau = get_tau(cfg, global_ep + ep)
        teacher_loss_sum = 0.0
        count = 0
        attn_sum = 0.0
        feat_kd_warned = False

        for batch in smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]"):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            with autocast_ctx:
                # (A) Student features and logits (kept fixed)
                with torch.no_grad():
                    feat_dict, s_logit, _ = student_model(x)
                    if la_mode:
                        key = cfg.get("feat_kd_key", "feat_2d")
                        s_feat = feat_dict[key]

                # (B) Teacher features
                feats_2d = []
                feats_4d = []
                feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) else "feat_2d"
                for tw in teacher_wrappers:
                    t_dict = tw(x)
                    feats_2d.append(t_dict[feat_key])
                    feats_4d.append(t_dict.get("feat_4d"))

                # (C) MBM + synergy_head
                if la_mode:
                    if isinstance(mbm, IB_MBM):
                        syn_feat, mu, logvar = mbm(
                            s_feat, torch.stack(feats_2d, dim=1)
                        )
                        attn = None
                        ib_loss_val = 0.0
                        if cfg.get("use_ib", False):
                            mu, logvar = mu.float(), logvar.float()
                            ib_beta = get_beta(cfg, global_ep + ep)
                            ib_loss_val = ib_loss(mu, logvar, beta=ib_beta)
                    else:
                        syn_feat, attn, _, _ = mbm(s_feat, feats_2d)
                        ib_loss_val = 0.0
                    fsyn = syn_feat
                else:
                    fsyn = mbm(feats_2d, feats_4d)
                    attn = None
                    ib_loss_val = 0.0
                zsyn = synergy_head(fsyn)

                # (D) compute loss (KL + synergyCE)
                if cfg.get("use_ib", False) and isinstance(mbm, IB_MBM):
                    ce_vec = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                        reduction="none",
                    )
                    kd_vec = kd_loss_fn(
                        zsyn, s_logit, T=cur_tau, reduction="none"
                    ).sum(dim=1)
                    cw = certainty_weights(logvar).mean(dim=1).to(zsyn.dtype)
                    loss_ce = (cw * ce_vec).mean()
                    loss_kd = (cw * kd_vec).mean()
                else:
                    loss_kd = kd_loss_fn(zsyn, s_logit, T=cur_tau)
                    loss_ce = ce_loss_fn(
                        zsyn,
                        y,
                        label_smoothing=cfg.get("label_smoothing", 0.0),
                    )

                # ① 누락 시 기본값 0.6
                synergy_weight = cfg.get("synergy_ce_alpha", 0.6)
                synergy_ce_loss = synergy_weight * loss_ce

                if la_mode and attn is not None:
                    attn_sum += attn.mean().item() * x.size(0)

                feat_kd_loss = torch.tensor(0.0, device=cfg["device"])
                if la_mode and cfg.get("feat_kd_alpha", 0) > 0 and not isinstance(mbm, IB_MBM):
                    s_flat = s_feat.view(s_feat.size(0), -1)
                    f_flat = fsyn.detach().view(fsyn.size(0), -1).to(s_flat.dtype)
                    if s_flat.size(1) == f_flat.size(1):
                        feat_kd_loss = torch.nn.functional.mse_loss(s_flat, f_flat)
                    else:
                        if not feat_kd_warned:
                            logger.info(
                                f"[TeacherAdaptive] skip feat KD: s_feat={s_flat.size(1)}"
                                f" vs fsyn={f_flat.size(1)}"
                            )
                            feat_kd_warned = True

            # Standard KD + CE
            total_loss = (
                cfg["teacher_adapt_alpha_kd"] * loss_kd
                + synergy_ce_loss
                + cfg.get("feat_kd_alpha", 0) * feat_kd_loss
                + ib_loss_val
            )

            # --- 1) L2 regularization on teacher parameters ---
            reg_loss = torch.tensor(0.0, device=cfg["device"])
            for p in teacher_params:                 # collected above
                if p.requires_grad:
                    reg_loss = reg_loss + p.pow(2).sum()
            total_loss = total_loss + float(cfg.get("reg_lambda", 0.0)) * reg_loss
            # -----------------------------------------------------------

            # --- 2) L2 regularization on MBM and Synergy-Head ---
            mbm_reg_loss = torch.tensor(0.0, device=cfg["device"])
            for p in mbm_params + syn_params:
                if p.requires_grad:
                    mbm_reg_loss = mbm_reg_loss + p.pow(2).sum()
            total_loss = total_loss + float(cfg.get("mbm_reg_lambda", 0.0)) * mbm_reg_loss
            # -----------------------------------------------------------      
            
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        teacher_params + mbm_params + syn_params,
                        cfg["grad_clip_norm"],
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if cfg.get("grad_clip_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        teacher_params + mbm_params + syn_params,
                        cfg["grad_clip_norm"],
                    )
                optimizer.step()

            teacher_loss_sum += total_loss.item() * x.size(0)
            count += x.size(0)

        ep_loss = teacher_loss_sum / count
        attn_avg = attn_sum / count if la_mode and count > 0 else 0.0

        # synergy_eval
        if testloader is not None:
            synergy_test_acc = eval_synergy(
                teacher_wrappers,
                mbm,
                synergy_head,
                loader=testloader,
                device=cfg["device"],
                cfg=cfg,
                student_model=student_model if la_mode else None,
            )
        else:
            synergy_test_acc = -1

        logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")

        # ── NEW: per-epoch logging ───────────────────────────────
        logger.update_metric(f"teacher_ep{ep+1}_loss", ep_loss)
        logger.update_metric(f"teacher_ep{ep+1}_synAcc", synergy_test_acc)
        logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)
        if la_mode:
            logger.update_metric(f"teacher_ep{ep+1}_attn", attn_avg)

        if scheduler is not None:
            scheduler.step()

        # best snapshot
        if synergy_test_acc > best_synergy:
            best_synergy = synergy_test_acc
            best_state["teacher_wraps"] = [copy.deepcopy(tw.state_dict()) for tw in teacher_wrappers]
            best_state["mbm"] = copy.deepcopy(mbm.state_dict())
            best_state["syn_head"] = copy.deepcopy(synergy_head.state_dict())

    # restore best
    for i, tw in enumerate(teacher_wrappers):
        tw.load_state_dict(best_state["teacher_wraps"][i])
    mbm.load_state_dict(best_state["mbm"])
    synergy_head.load_state_dict(best_state["syn_head"])

    return best_synergy
