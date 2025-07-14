#!/usr/bin/env python3
from __future__ import annotations
# scripts/launcher.py
# ------------------------------------------------------------
# Unified launcher  - single run & hyper-parameter sweep
#   usage)
#       python scripts/launcher.py experiments/exp.yaml [--extra CLI...]
# ------------------------------------------------------------

import argparse, itertools, os, pathlib, subprocess, sys, tempfile, yaml, time

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root

# ------------------------------------------------------------------
# 모든 결과 폴더에 고유 식별자 부여: SLURM_JOB_ID 없으면 YYMMDDhhmmss
# ------------------------------------------------------------------
JOB_ID = os.getenv("SLURM_JOB_ID") or time.strftime("%y%m%d%H%M%S")

# ---------------------------------------------------------------------
# crash helper
# ---------------------------------------------------------------------

def _abort(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)

# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------

def _to_abs(p: str | pathlib.Path) -> str:
    p = pathlib.Path(p)
    return str(p) if p.is_absolute() else str(ROOT / p)


def _scenario_from_imports(imports: list[str]) -> str:
    for p in imports:
        if "/scenario/" in p.replace("\\", "/"):
            return pathlib.Path(p).stem.lower()
    return "standard"  # fallback (should not happen)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_yaml", help="experiments/ folder YAML config")
    ap.add_argument("--gpus", default="0",
                    help="Comma-separated GPU IDs (default 1 GPU)")
    ap.add_argument("--max_parallel", type=int, default=1,
                    help="Maximum parallel processes")
    # (선택) CLI 로도 강제할 수 있게 --mode 추가
    ap.add_argument("--mode", choices=["full", "single"], default=None,
                    help="Override sweep mode in YAML")
    ap.add_argument("--extra", nargs=argparse.REMAINDER,
                    help="Additional CLI args forwarded to main.py")
    args = ap.parse_args()

    with open(args.exp_yaml, "r") as f:
        exp_cfg = yaml.safe_load(f) or {}

    imports: list[str] = exp_cfg.get("imports", [])
    if not imports:
        _abort("'imports' list is empty - need at least one entry")

    imports = [_to_abs(p) for p in imports]
    scenario = _scenario_from_imports(imports)

    sweep_dict: dict[str, list] | None = exp_cfg.get("sweep")

    # ──────────────────────────────────────────────────────────────
    # 1) sweep_mode 결정 –  CLI(--mode) 가 YAML(sweep_mode) 보다 우선
    # ──────────────────────────────────────────────────────────────
    yaml_mode = (exp_cfg.get("sweep_mode") or "full").lower()
    if yaml_mode not in {"full", "single"}:
        _abort("sweep_mode must be 'full' or 'single'")
    sweep_mode = args.mode or yaml_mode  # 최종 모드

    # policy: sweep allowed only in standard scenario
    if sweep_dict and scenario != "standard":
        _abort(
            "Sweep supported only for standard scenario "
            f"(current='{scenario}')."
        )

    # ───── generate parameter sets ──────────────────────────────
    param_sets: list[dict[str, str | int | float]] = []
    if sweep_dict:
        if sweep_mode == "full":         # Cartesian product
            keys, vals = zip(*sweep_dict.items())
            for tup in itertools.product(*vals):
                param_sets.append(dict(zip(keys, tup)))
        else:                            # single‑var variation
            for k, v_list in sweep_dict.items():
                for v in v_list:
                    param_sets.append({k: v})
    else:
        param_sets.append({})            # single run

    # global overrides (exp_id, student_iters, ...)
    global_ovr = {k: v for k, v in exp_cfg.items() if k not in {"imports", "sweep"}}
    exp_id = global_ovr.pop("exp_id", pathlib.Path(args.exp_yaml).stem)

    base_results = ROOT / "outputs" / "results" / f"{exp_id}_{JOB_ID}"
    base_results.mkdir(parents=True, exist_ok=True)
    print(
        f"[LAUNCH] scenario={scenario} sweep={bool(sweep_dict)} "
        f"mode={sweep_mode} runs={len(param_sets)} exp_id='{exp_id}' job={JOB_ID}",
        flush=True,
    )

    gpu_ids = [g.strip() for g in args.gpus.split(',') if g.strip()]
    if len(gpu_ids) < 1:
        _abort("--gpus is empty (need at least 1 GPU)")

    procs = []
    for idx, param in enumerate(param_sets):
        tmp_yaml = tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        )
        yaml.safe_dump({**global_ovr, **param}, tmp_yaml)
        tmp_yaml.close()

        tag = "_".join(f"{k}{v}" for k, v in param.items()) or "single"
        results_dir = base_results / tag  # <exp>_<JOB_ID>/<tag>
        results_dir.mkdir(parents=True, exist_ok=True)

        cfg_chain = ",".join(imports + [tmp_yaml.name])
        cmd = [
            "python",
            str(ROOT / "main.py"),
            "--config",
            cfg_chain,
            "--results_dir",
            str(results_dir),
        ] + (args.extra or [])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[idx % len(gpu_ids)]

        log_path = results_dir / "run.log"
        print(
            f"[LAUNCH] ▶ {idx+1}/{len(param_sets)} param={param or '—'} GPU={env['CUDA_VISIBLE_DEVICES']} log={log_path}",
            flush=True,
        )
        # ✅ with‑context로 열고, fp 수명 = Popen 종료까지
        fp = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=fp,
            stderr=subprocess.STDOUT,
        )
        proc._log_fp = fp            # 종료 후 닫기 위해 보관
        procs.append(proc)

        while len([p for p in procs if p.poll() is None]) >= args.max_parallel:
            time.sleep(10)

    for p in procs:
        p.wait()
        if hasattr(p, "_log_fp") and not p._log_fp.closed:
            p._log_fp.close()        # 🧹 열려 있던 로그 핸들 확실히 닫기
    print("[LAUNCH] all subprocesses finished ✅", flush=True)


if __name__ == "__main__":
    main()
