import argparse
import csv
import os
import re
from typing import List, Tuple

# Use non-interactive backend for PNG saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_PATTERNS = [
    re.compile(r"\[CFG\]\s*method=([^\s|]+)", re.IGNORECASE),
    re.compile(r'"method_name"\s*:\s*"([^"]+)"'),
    re.compile(r"KD\s*:\s*([^\s|]+)\s*\|"),
]

# For local epoch + acc lines
EPOCH_ACC_PATTERN = re.compile(r"\[StudentDistill ep=(\d+)\].*?testAcc=([0-9.]+)")

# For exp_id (rung label like L0_baseline)
EXP_ID_PATTERNS = [
    re.compile(r'"exp_id"\s*:\s*"([^"]+)"'),
    re.compile(r"ExpID\s*:\s*([A-Za-z0-9_\-]+)"),
    re.compile(r"Starting ASIB ablation experiment\s*:\s*([A-Za-z0-9_\-]+)")
]


def extract_method_or_rung(text: str) -> str:
    # Prefer rung label from exp_id (e.g., L0_baseline -> L0)
    exp_id = None
    for pat in EXP_ID_PATTERNS:
        m = pat.search(text)
        if m:
            exp_id = m.group(1).strip()
            break
    if exp_id:
        rung = exp_id.split("_")[0]
        if re.match(r"^L\d+", rung):
            return rung
    # Fallback to method name
    for pat in METHOD_PATTERNS:
        m = pat.search(text)
        if m:
            return m.group(1).strip()
    return exp_id or "unknown"


def extract_epoch_acc_global(lines: List[str]) -> List[Tuple[int, int, float]]:
    """
    Return list of (local_epoch, global_epoch, acc).
    Global epoch is computed by counting StudentDistill lines sequentially
    across stages (so it continues from 1 to total over all stages).
    """
    out: List[Tuple[int, int, float]] = []
    global_ep = 0
    for line in lines:
        m = EPOCH_ACC_PATTERN.search(line)
        if m:
            try:
                local_ep = int(m.group(1))
                acc = float(m.group(2))
                global_ep += 1
                out.append((local_ep, global_ep, acc))
            except Exception:
                continue
    return out


def parse_file(path: str) -> Tuple[str, List[Tuple[int, int, float]]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()
    text = "\n".join(lines)
    method = extract_method_or_rung(text)
    pairs = extract_epoch_acc_global(lines)
    return method, pairs


def write_csv(rows: List[Tuple[str, int, int, float]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "global_epoch", "local_epoch", "acc"])
        writer.writeheader()
        for method, global_epoch, local_epoch, acc in rows:
            writer.writerow({
                "method": method,
                "global_epoch": int(global_epoch),
                "local_epoch": int(local_epoch),
                "acc": f"{float(acc):.2f}",
            })


def plot_csv(rows: List[Tuple[str, int, int, float]], out_png: str, title: str = None) -> None:
    # Group by method
    by_method = {}
    for method, g_ep, l_ep, acc in rows:
        by_method.setdefault(method, []).append((int(g_ep), float(acc)))
    plt.figure(figsize=(9, 6))
    for method, pairs in by_method.items():
        pairs.sort(key=lambda t: t[0])
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        plt.plot(xs, ys, marker='o', linewidth=1.8, markersize=3.5, label=method)
    plt.xlabel("Global Epoch")
    plt.ylabel("Accuracy (%)")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Parse ASIB logs to CSV and plot (method, epoch, acc)")
    ap.add_argument("logs", nargs="+", help="Paths to .log files")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--out-png", required=False, help="Output PNG path for plot")
    ap.add_argument("--title", default=None, help="Plot title")
    args = ap.parse_args()

    all_rows: List[Tuple[str, int, int, float]] = []
    for p in args.logs:
        if not os.path.exists(p):
            continue
        method, pairs = parse_file(p)
        for local_ep, global_ep, acc in pairs:
            all_rows.append((method, global_ep, local_ep, acc))

    # Sort for readability: method then epoch
    all_rows.sort(key=lambda r: (r[0], int(r[1])))
    write_csv(all_rows, args.out_csv)

    if args.out_png:
        plot_csv(all_rows, args.out_png, title=args.title)


if __name__ == "__main__":
    main()


