# scripts/run_sweep.py
import argparse, itertools, os, subprocess, yaml, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",  required=True,
                   help="고정 YAML (예: configs/base.yaml)")
    p.add_argument("--sweep", required=True,
                   help="값이 list 인 항목만 들어있는 YAML")
    # GPU·병렬 수 고정  ─ 변경 금지(실수 방지용)
    p.add_argument("--gpus",  default="0",
                   help="(고정) GPU id = 0  ‑ 다른 값 넣으면 오류")
    p.add_argument("--max_parallel", type=int, default=1,
                   help="(고정) 동시 실행 = 1  ‑ 다른 값 넣으면 오류")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="main.py 에 그대로 넘길 추가 CLI 인수")
    args = p.parse_args()

    # ─────── 고정 값 검증 ───────
    if args.gpus.strip() not in {"0", "0,"}:
        raise ValueError("이 스크립트는 GPU 0 한​장만 사용하도록 고정되었습니다.")
    if args.max_parallel != 1:
        raise ValueError("max_parallel 은 1 로 고정하시길 바랍니다.")

    with open(args.sweep) as f:
        sweep_cfg = yaml.safe_load(f)

    # list 가 아닌 값은 무시
    sweep_vars = {k: v for k, v in sweep_cfg.items() if isinstance(v, list)}
    keys, vals = zip(*sweep_vars.items())
    combos = list(itertools.product(*vals))

    # 항상 GPU 0, 동시 1
    gpu_id   = "0"
    max_jobs = 1
    procs    = []

    for idx, tup in enumerate(combos):
        cli = []
        for k, v in zip(keys, tup):
            cli += [f"--{k}", str(v)]
        exp_name = "_".join(f"{k}{v}" for k, v in zip(keys, tup))
        out_dir  = f"results/sweep/{exp_name}"
        log_file = f"logs/{exp_name}.log"
        os.makedirs(out_dir,  exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id   # GPU 0 한​장만 노출

        cmd = ["python", "main.py",
               "--cfg", args.base,
               "--results_dir", out_dir] + (args.extra or []) + cli

        procs.append(subprocess.Popen(cmd, env=env,
                         stdout=open(log_file, "w"),
                         stderr=subprocess.STDOUT))

        # 동시 실행 제한
        while len([p for p in procs if p.poll() is None]) >= max_jobs:
            time.sleep(10)

    # 모든 subprocess 종료 대기
    for p in procs:
        p.wait()

if __name__ == "__main__":
    main()
