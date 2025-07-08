# scripts/run_sweep.py
import argparse, itertools, os, subprocess, yaml, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base",  required=True,
                   help="고정 YAML (예: configs/base.yaml)")
    p.add_argument("--sweep", required=True,
                   help="값이 list 인 항목만 들어있는 YAML")
    p.add_argument("--gpus",  default="0",
                   help="쉼표로 구분된 GPU ID 리스트 (예: 0,1,2,3)")
    p.add_argument("--max_parallel", type=int, default=4,
                   help="동시 실행 최대 프로세스 수")
    p.add_argument("--extra", nargs=argparse.REMAINDER,
                   help="main.py 에 그대로 넘길 추가 CLI 인수")
    args = p.parse_args()

    with open(args.sweep) as f:
        sweep_cfg = yaml.safe_load(f)

    # list 가 아닌 값은 무시
    sweep_vars = {k: v for k, v in sweep_cfg.items() if isinstance(v, list)}
    keys, vals = zip(*sweep_vars.items())
    combos = list(itertools.product(*vals))

    gpu_ids = [g.strip() for g in args.gpus.split(",")]
    max_jobs = args.max_parallel
    procs = []

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
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[idx % len(gpu_ids)]

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
