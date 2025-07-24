# modules/trainer_teacher.py

import torch
import copy
from utils.progress import smart_tqdm
from models.mbm import IB_MBM

from modules.losses import (
    kd_loss_fn, ce_loss_fn, ib_loss, certainty_weights, feat_mse_loss
)
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
    query_mode = isinstance(mbm, IB_MBM) \
                 or (cfg or {}).get("mbm_type", "").lower() == "ib_mbm"
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

            if query_mode:
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
    # cfg.train_distill_adapter_only == True → teacher 본체는 그대로 freeze
    teacher_params = []
    only_da = cfg.get("train_distill_adapter_only", False)
    for tw in teacher_wrappers:
        param_src = (
            tw.distillation_adapter.parameters()
            if only_da and hasattr(tw, "distillation_adapter")
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
    query_mode = isinstance(mbm, IB_MBM) \
                 or cfg.get("mbm_type", "").lower() == "ib_mbm"
    for ep in range(teacher_epochs):
        for tw in teacher_wrappers:
            tw.train()
        mbm.train()
        synergy_head.train()
        if student_model is not None:
            student_model.eval()
        if scheduler is not None and hasattr(scheduler, "T_max"):
            total_epochs = scheduler.T_max
            cur_tau = get_tau(cfg, min(global_ep + ep, total_epochs - 1))
        else:
            cur_tau = get_tau(cfg, global_ep + ep)
        teacher_loss_sum = 0.0
        count = 0
        attn_sum = 0.0

        for step, batch in enumerate(
            smart_tqdm(trainloader, desc=f"[TeacherAdaptive ep={ep+1}]")
        ):
            x, y = batch
            x, y = x.to(cfg["device"]), y.to(cfg["device"])

            with autocast_ctx:
                # (A) Student features and logits (kept fixed)
                with torch.no_grad():
                    feat_dict, s_logit, _ = student_model(x)
                    if query_mode:
                        key = cfg.get("feat_kd_key", "feat_2d")
                        s_feat = feat_dict[key]

                # (B) Teacher features
                feats_2d = []
                feats_4d = []
                feat_key = "distill_feat" if cfg.get("use_distillation_adapter", False) else "feat_2d"
                t1_dict = None
                for i, tw in enumerate(teacher_wrappers):
                    t_dict = tw(x)
                    if i == 0:
                        t1_dict = t_dict
                    feats_2d.append(t_dict[feat_key])
                    feats_4d.append(t_dict.get("feat_4d"))

                # (C) MBM + synergy_head
                if query_mode:
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

                    # ---- DEBUG: 첫 batch 모양 확인 ----
                    if ep == 0 and step == 0:
                        print(
                            "[DBG/teacher] t1_logit",
                            tuple(t1_dict["logit"].shape),
                            "s_logit",
                            tuple(s_logit.shape),
                            "zsyn",
                            tuple(zsyn.shape),
                        )
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

                if query_mode and attn is not None:
                    attn_sum += attn.mean().item() * x.size(0)

                feat_kd_loss = torch.tensor(0.0, device=cfg["device"])
                if query_mode and cfg.get("feat_kd_alpha", 0) > 0 and not isinstance(mbm, IB_MBM):
                    feat_kd_loss = feat_mse_loss(
                        s_feat, fsyn,
                        norm=cfg.get("feat_kd_norm", "none")
                    )

            # Standard KD + CE
            kd_weight = cfg.get(
                "teacher_adapt_alpha_kd",
                cfg.get("kd_alpha", 1.0),
            )
            total_loss = (
                kd_weight * loss_kd
                + synergy_ce_loss
                + cfg.get("feat_kd_alpha", 0) * feat_kd_loss
                + ib_loss_val
            )

    # --- 1) L2 regularization on teacher parameters ---
    if teacher_params:
        reg_loss = torch.stack([(p ** 2).mean() for p in teacher_params]).mean()
    else:  # protect empty list
        reg_loss = torch.tensor(0.0, device=zsyn.device)
        print("[TeacherAdaptive] teacher_params empty -> reg_loss=0")

    total_loss = total_loss + float(cfg.get("reg_lambda", 0.0)) * reg_loss
    # -----------------------------------------------------------

    # --- 2) L2 regularization on MBM and Synergy-Head ---
    mbm_reg_loss = torch.stack([
        (p ** 2).mean() for p in mbm_params + syn_params if p.requires_grad
    ]).mean()
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
        print(f"[TeacherAdaptive] batch loss={total_loss.item():.4f}")
        if cfg.get("grad_clip_norm", 0) > 0:
            torch.nn.utils.clip_grad_norm_(
                teacher_params + mbm_params + syn_params,
                cfg["grad_clip_norm"],
            )
        optimizer.step()

    teacher_loss_sum += total_loss.item() * x.size(0)
    count += x.size(0)

    ep_loss = teacher_loss_sum / count
    attn_avg = attn_sum / count if query_mode and count > 0 else 0.0

    # synergy_eval
    if testloader is not None:
        synergy_test_acc = eval_synergy(
            teacher_wrappers,
            mbm,
            synergy_head,
            loader=testloader,
            device=cfg["device"],
            cfg=cfg,
            student_model=student_model if query_mode else None,
        )
    else:
        synergy_test_acc = -1

    logger.info(f"[TeacherAdaptive ep={ep+1}] loss={ep_loss:.4f}, synergy={synergy_test_acc:.2f}")

    # ── NEW: per-epoch logging ───────────────────────────────
    logger.update_metric(f"teacher_ep{ep+1}_loss", ep_loss)
    logger.update_metric(f"teacher_ep{ep+1}_synAcc", synergy_test_acc)
    logger.update_metric(f"epoch{global_ep+ep+1}_tau", cur_tau)
    if query_mode:
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
