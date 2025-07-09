#!/usr/bin/env python
# scripts/run_sweep.py
# ------------------------------------------------------------
#SBATCH --job-name=kd_sweep
#SBATCH --partition=base_suma_rtx3090
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/%x_%j.out   # SLURM 로그
#SBATCH --chdir=/home/suyoung425/ASMB_KD   # ★ 프로젝트 루트 고정
# ------------------------------------------------------------

# scripts/run_sweep.py
import argparse, itertools, os, subprocess, yaml, time
import pathlib
# DEPRECATED: 남겨두되, 실행 시 경고만 출력
print("[WARN] run_sweep.py is deprecated – 통합 런처(launcher.py)를 이용하세요.", flush=True)


ROOT_DIR   = os.getcwd()   # ( --chdir 로 고정됨 )

def main():
    # ────────────────────────────────────────────────
    # 기본 정보 – SLURM 로그에 바로 찍힘
    # ────────────────────────────────────────────────
    print(f"[DBG] run_sweep.py started  cwd={os.getcwd()}", flush=True)

    p = argparse.ArgumentParser()
    # __file__ becomes a temporary slurm_script when submitted via sbatch.
    # Recover the original repository path using SLURM_SUBMIT_DIR.
    if "SLURM_SUBMIT_DIR" in os.environ:
        repo_root = pathlib.Path(os.environ["SLURM_SUBMIT_DIR"]).resolve()
    else:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
    p.add_argument("--base",  default=repo_root / "configs/base.yaml",
                   help="(기본) configs/base.yaml")
    # NEW ─ control.yaml 경로(방법·시나리오 고정용)
    p.add_argument("--control", default=repo_root / "configs/control.yaml",
                   help="(기본) configs/control.yaml")
    p.add_argument("--sweep", default=repo_root / "configs/ablation/kd_sweep.yaml",
                   help="(기본) configs/ablation/kd_sweep.yaml")
    # GPU·병렬 수 고정  ─ 변경 금지(실수 방지용)
    p.add_argument("--gpus",  default="0",    # GPU 1장 강제
                   help="쉼표로 구분된 GPU ID 리스트")
    p.add_argument("--max_parallel", type=int, default=1,  # 동시 1개
                   help="동시 실행 최대 프로세스 수")
    # ───────── Sweep 전략 선택 ─────────
    #   full   : 모든 list 변수의 Cartesian product (기존 동작)
    #   single : list 변수를 하나씩만 바꿔서 ‘기본+α’ 실험
    p.add_argument("--mode", choices=["full", "single"], default="single",
                   help="'full': 전체 조합,  'single': 변수별 개별 실험")
    #   필요한 변수만 골라서 sweep 하고 싶을 때
    #   ─ 기본값을 '' 로 두어 모든 list 항목을 sweep 대상으로 삼는다.
    p.add_argument("--keys", default="",
                   help="콤마로 구분된 sweep 대상 key "
                        "(비워두면 list‑항목 전부 sweep)")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="main.py 에 그대로 넘길 추가 CLI 인수")
    args = p.parse_args()

    print(f"[DBG] base  YAML = {args.base}",  flush=True)
    print(f"[DBG] sweep YAML = {args.sweep}", flush=True)

    # --- config 경로를 절대경로로 변환 ----------------------------------
    if not os.path.isabs(args.base):
        args.base = os.path.join(ROOT_DIR, args.base)
    if not os.path.isabs(args.sweep):
        args.sweep = os.path.join(ROOT_DIR, args.sweep)

    # Output directories under the project root
    out_root = repo_root / "outputs"
    os.makedirs(out_root, exist_ok=True)

    # ─────── 고정 값 검증 ───────
    gpu_ids = [g.strip() for g in args.gpus.split(',') if g.strip()]
    if len(gpu_ids) != 1:
        raise ValueError("이 스크립트는 GPU 1장만 사용하도록 고정되었습니다.")
    if args.max_parallel != 1:
        raise ValueError("max_parallel 은 1 로 고정하시길 바랍니다.")

    with open(args.sweep) as f:
        sweep_cfg = yaml.safe_load(f)
    print(f"[DBG] sweep keys = {list(sweep_cfg.keys())}", flush=True)

    # list 가 아닌 값은 무시
    sweep_vars = {k: v for k, v in sweep_cfg.items() if isinstance(v, list)}

    # --keys 필터링 (빈 문자열이면 생략)
    if args.keys.strip():
        wanted = [k.strip() for k in args.keys.split(',') if k.strip()]
        sweep_vars = {k: v for k, v in sweep_vars.items() if k in wanted}

    # ─ Sweep 전략 ─────────────────────────────────────────────
    param_sets: list[dict[str, str]] = []
    if args.mode == "full":
        keys, vals = zip(*sweep_vars.items())
        for tup in itertools.product(*vals):
            param_sets.append(dict(zip(keys, tup)))
    else:   # single mode
        for k, v_list in sweep_vars.items():
            for v in v_list:
                param_sets.append({k: v})

    print(f"[DBG] generated {len(param_sets)} experiment(s) "
          f"(mode={args.mode})", flush=True)

    # 항상 GPU 1, 동시 1
    max_jobs = args.max_parallel
    procs    = []

    for idx, params in enumerate(param_sets):
        exp_name = "_".join(f"{k}{v}" for k, v in params.items())
        out_dir  = os.path.join(ROOT_DIR, "outputs", "results", "sweep", exp_name)
        log_dir  = os.path.join(ROOT_DIR, "outputs", "sweep_logs")
        log_file = os.path.join(log_dir, f"{exp_name}.log")
        os.makedirs(out_dir, exist_ok=True); os.makedirs(log_dir, exist_ok=True)

        # ─ override.yaml 작성 (list 값 1개만 담김)
        override_path = os.path.join(out_dir, "override.yaml")
        with open(override_path, "w") as f_yaml:
            yaml.safe_dump(params, f_yaml)
        print(f"[DBG]   └─ override → {override_path}", flush=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[idx % len(gpu_ids)]

        # main.py 에는 cfg 3개만 넘김: base, control, override
        cfg_chain = f"{args.base},{args.control},{override_path}"
        cmd = ["python", os.path.join(ROOT_DIR, "main.py"),
               "--cfg", cfg_chain,
               "--results_dir", out_dir] + (args.extra or [])

        procs.append(subprocess.Popen(cmd, env=env,
                         stdout=open(log_file, "w"),
                         stderr=subprocess.STDOUT))

        print(f"[DBG] ▶ launch {idx+1}/{len(param_sets)}  "
              f"{params}  → PID={procs[-1].pid}  log={log_file}",
              flush=True)

        # 동시 실행 제한
        while len([p for p in procs if p.poll() is None]) >= max_jobs:
            time.sleep(10)

    # 모든 subprocess 종료 대기
    for p in procs:
        p.wait()

    print("[DBG] all subprocesses finished ✅", flush=True)

if __name__ == "__main__":
    main()
