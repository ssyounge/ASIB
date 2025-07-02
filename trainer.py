# trainer.py

import torch
import torch.nn.functional as F
from utils.schedule import cosine_lr_scheduler


def teacher_vib_update(teacher1, teacher2, vib_mbm, loader, cfg, optimizer):
    device = cfg.get("device", "cuda")
    beta = cfg.get("beta_bottleneck", 0.003)
    clip = cfg.get("grad_clip_norm", 0)
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
            z, logit_syn, kl_z, _ = vib_mbm(f1, f2)
            loss = F.cross_entropy(logit_syn, y) + beta * kl_z.mean()
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(vib_mbm.parameters(), clip)
            optimizer.step()
        scheduler.step()


def student_vib_update(teacher1, teacher2, student_model, vib_mbm, student_proj, loader, cfg, optimizer):
    device = cfg.get("device", "cuda")
    alpha = cfg.get("alpha_kd", 0.7)
    ce_alpha = cfg.get("ce_alpha", 1.0)
    clip = cfg.get("grad_clip_norm", 0)
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
                _, _, _, mu = vib_mbm(f1, f2)
                z_target = mu.detach()
            feat_dict, s_logit, _ = student_model(x)
            s_feat = feat_dict["feat_2d"]
            z_pred = student_proj(s_feat)
            kd = F.mse_loss(z_pred, z_target)
            loss = ce_alpha * F.cross_entropy(s_logit, y) + alpha * kd
            optimizer.zero_grad()
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(student_model.parameters()) + list(student_proj.parameters()),
                    clip,
                )
            optimizer.step()
        scheduler.step()


