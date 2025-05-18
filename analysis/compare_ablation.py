#!/usr/bin/env python3

"""
analysis/compare_ablation.py

Example script to compare ablation results from multiple CSV files,
and produce a summary table or stats.
"""

import os
import pandas as pd

def main():
    # 예: results/summary.csv 에는 여러 파라미터/성과가 정리돼 있다고 가정
    summary_path = "results/summary.csv"
    if not os.path.exists(summary_path):
        print(f"[Error] summary.csv not found at {summary_path}")
        return

    df = pd.read_csv(summary_path)  
    # 예: 컬럼 ["method", "teacher_lr", "synergy_ce_alpha", "test_acc", "seed", "epoch"]

    # 1) group by key hyperparams => compute mean & std of test_acc
    group_cols = ["method", "teacher_lr", "synergy_ce_alpha"]
    agg_result = df.groupby(group_cols)["test_acc"].agg(["mean", "std"]).reset_index()
    # "mean" -> test_acc_mean, "std" -> test_acc_std

    # 2) print or save
    print("== Ablation Summary: By Method & LR & synergy_ce_alpha ==\n")
    print(agg_result)

    out_path = "results/ablation_summary.csv"
    agg_result.to_csv(out_path, index=False)
    print(f"[Info] ablation summary saved to {out_path}")

if __name__ == "__main__":
    main()
