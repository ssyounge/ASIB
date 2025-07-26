#!/usr/bin/env python3

"""
analysis/compare_ablation.py

Example script to compare ablation results from a CSV file and
produce a summary table or stats. The input path defaults to
``results/summary.csv`` and the output is written to
``results/ablation_summary.csv``.

Usage::

    python analysis/compare_ablation.py \
        --summary_csv results/summary.csv \
        --out_path results/ablation_summary.csv
"""

import os
import argparse
import pandas as pd
import logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate ablation results from a CSV file"
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="results/summary.csv",
        help="Path to input summary CSV",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="results/ablation_summary.csv",
        help="Where to save the aggregated summary",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    # 예: results/summary.csv 에는 여러 파라미터/성과가 정리돼 있다고 가정
    summary_path = args.summary_csv
    if not os.path.exists(summary_path):
        logging.error("summary.csv not found at %s", summary_path)
        return

    df = pd.read_csv(summary_path)  
    # 예: 컬럼 ["method", "teacher_lr", "synergy_ce_alpha", "test_acc", "seed", "epoch"]

    # 1) group by key hyperparams => compute mean & std of test_acc
    group_cols = ["method", "teacher_lr", "synergy_ce_alpha"]
    agg_result = df.groupby(group_cols)["test_acc"].agg(["mean", "std"]).reset_index()
    # "mean" -> test_acc_mean, "std" -> test_acc_std

    # 2) print or save
    logging.info("== Ablation Summary: By Method & LR & synergy_ce_alpha ==\n")
    logging.info("%s", agg_result)

    out_path = args.out_path
    agg_result.to_csv(out_path, index=False)
    logging.info("ablation summary saved to %s", out_path)

if __name__ == "__main__":
    main()
