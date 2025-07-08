# scripts/run_sweep.py
import argparse
import itertools
import os
import subprocess
import yaml
import time

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
    # ───────── Sweep 전략 선택 ─────────
    #   full   : 모든 list 변수의 Cartesian product (기존 동작)
    #   single : list 변수를 하나씩만 바꿔서 ‘기본+α’ 실험
    p.add_argument("--mode", choices=["full", "single"], default="full",
                   help="'full': 전체 조합,  'single': 변수별 개별 실험")
    #   필요한 변수만 골라서 sweep 하고 싶을 때
    p.add_argument("--keys", default="",
                   help="콤마로 구분된 sweep 대상 key (비워두면 전부)")
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

    # --keys 필터링
    if args.keys:
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

    # 항상 GPU 0, 동시 1
    gpu_id   = "0"
    max_jobs = 1
    procs    = []

    for idx, params in enumerate(param_sets):
        cli = sum(([f"--{k}", str(v)] for k, v in params.items()), [])
        exp_name = "_".join(f"{k}{v}" for k, v in params.items())
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
