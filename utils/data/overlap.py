#!/usr/bin/env python
# utils/make_overlap_subsets.py
"""Generate class–overlap JSON files for two teachers."""

import argparse, json, random, pathlib

def split_classes(n_cls=100, seed=42):
    random.seed(seed)
    cls = list(range(n_cls))
    random.shuffle(cls)
    return cls

def make_pairs(overlap_pct: int, n_cls=100, seed=42):
    base = split_classes(n_cls, seed)
    m = int(n_cls * overlap_pct / 100)
    common = base[:m]
    rest   = base[m:]
    half   = (n_cls - m) // 2
    t1 = sorted(common + rest[:half])
    t2 = sorted(common + rest[half:half*2])
    return {"overlap": overlap_pct, "T1": t1, "T2": t2}

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--overlap", type=int, required=True, help="0-100 (%)")
    pa.add_argument("--out", type=pathlib.Path, required=True)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    pairs = make_pairs(args.overlap, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(pairs, f)
    print(f"Saved {args.out} (overlap {args.overlap} %)")

if __name__ == "__main__":
    main()
