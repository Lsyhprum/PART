#!/usr/bin/env python3
# ============================================================
# eval_tradeoff.py -- Aggregate the [RESULT] logs of equality inline experiments
# across (dataset x attribute config x quantization bits), and produce
# qps-recall tradeoff curves and a summary CSV.
#
# Log naming convention (produced by run.sh):
#   bit=0 : {LOG_DIR}/symqg_{DATASET}_A{ATTR_DIM}_C{ATTR_CARD}_inline.log
#   bit>0 : {LOG_DIR}/symqg_{DATASET}_A{ATTR_DIM}_C{ATTR_CARD}_inline_b{BITS}.log
# Example line:
#   [RESULT] algo=symqg param=100 k=100 qps=17998.1 recall=0.156731
# ============================================================
import os
import re
import sys
import glob
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_DIR = os.environ.get("SYMQG_LOG_DIR", "./_log")
FIG_DIR = os.environ.get("SYMQG_FIG_DIR", "./_fig/inline_tradeoff")

RESULT_PAT = re.compile(
    r"\[RESULT\]\s+algo=(\S+)\s+param=(\d+)\s+k=(\d+)\s+qps=([\d.]+)\s+recall=([\d.]+)"
)
# trailing optional _b{N} denotes quantization bits (default 0)
FILE_PAT = re.compile(r"symqg_(.+)_A(\d+)_C(\d+)_inline(?:_b(\d+))?\.log$")

# fixed display order
DS_ORDER = ["openai1536", "msmarco", "sift"]
AC_ORDER = ["A1_C10", "A1_C2", "A2_C10", "A3_C10"]

DS_COLOR = {"openai1536": "tab:blue", "msmarco": "tab:green", "sift": "tab:red"}
AC_STYLE = {"A1_C10": "-", "A1_C2": "--", "A2_C10": "-.", "A3_C10": ":"}
BIT_MARKER = {0: "o", 1: "^", 4: "s"}
BIT_LS = {0: "-", 1: "--", 4: "-."}


def parse_logs():
    """Return {dataset: {ac: {bits: [(ef, qps, recall), ...]}}}, each sorted by recall ascending."""
    data = {}
    for path in sorted(glob.glob(os.path.join(LOG_DIR, "symqg_*_inline*.log"))):
        fn = os.path.basename(path)
        m = FILE_PAT.search(fn)
        if not m:
            print(f"[warn] skipping unrecognized log: {fn}", file=sys.stderr)
            continue
        ds, A, C = m.group(1), m.group(2), m.group(3)
        ac = f"A{A}_C{C}"
        bits = int(m.group(4)) if m.group(4) else 0
        rows = []
        with open(path) as f:
            for line in f:
                r = RESULT_PAT.search(line)
                if r:
                    ef = int(r.group(2))
                    qps = float(r.group(4))
                    rec = float(r.group(5))
                    rows.append((ef, qps, rec))
        if not rows:
            print(f"[warn] no [RESULT] line in: {fn}", file=sys.stderr)
            continue
        rows.sort(key=lambda x: x[2])
        data.setdefault(ds, {}).setdefault(ac, {})[bits] = rows
        print(f"[info] {ds} {ac} b{bits}: {len(rows)} (ef,qps,recall) points")
    return data


def write_csv(data, csv_path):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "attr_config", "bits", "ef", "qps", "recall"])
        for ds in DS_ORDER:
            if ds not in data:
                continue
            for ac in AC_ORDER:
                if ac not in data[ds]:
                    continue
                for bits in sorted(data[ds][ac]):
                    for ef, qps, rec in data[ds][ac][bits]:
                        w.writerow([ds, ac, bits, ef, qps, rec])
    print(f"[csv] writing {csv_path}")


def _plot_curve(ax, rows, color, style, marker, label):
    xs = [r[2] for r in rows]
    ys = [r[1] for r in rows]
    ax.plot(xs, ys, marker=marker, ms=5, color=color, linestyle=style, label=label)


def plot_per_dataset(data):
    for ds in DS_ORDER:
        if ds not in data:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for ac in AC_ORDER:
            if ac not in data[ds]:
                continue
            for bits in sorted(data[ds][ac]):
                _plot_curve(ax, data[ds][ac][bits], DS_COLOR.get(ds, "gray"),
                            BIT_LS.get(bits, "-"), BIT_MARKER.get(bits, "o"),
                            f"{ac} b{bits}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("QPS")
        ax.set_title(f"Inline equality-filter tradeoff - {ds}")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Attr config / bits", fontsize=8)
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"tradeoff_{ds}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[fig] {out}")


def plot_bit_compare_per_ac(data):
    """One figure per (dataset, attribute config), overlaying the bit=0 and bit=1 curves to directly compare quantization gains."""
    for ds in DS_ORDER:
        if ds not in data:
            continue
        for ac in AC_ORDER:
            if ac not in data[ds]:
                continue
            if len(data[ds][ac]) < 2:
                continue  # skip comparison figure when only a single bit level
            fig, ax = plt.subplots(figsize=(7, 5))
            for bits in sorted(data[ds][ac]):
                _plot_curve(ax, data[ds][ac][bits], DS_COLOR.get(ds, "gray"),
                            BIT_LS.get(bits, "-"), BIT_MARKER.get(bits, "o"),
                            f"bit={bits}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("QPS")
            ax.set_title(f"quant bits comparison - {ds} {ac}")
            ax.grid(True, alpha=0.3)
            ax.legend(title="quantization bits")
            fig.tight_layout()
            out = os.path.join(FIG_DIR, f"tradeoff_{ds}_{ac}_bits.png")
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"[fig] {out}")


def plot_combined(data):
    fig, ax = plt.subplots(figsize=(9, 6))
    for ds in DS_ORDER:
        if ds not in data:
            continue
        for ac in AC_ORDER:
            if ac not in data[ds]:
                continue
            for bits in sorted(data[ds][ac]):
                _plot_curve(ax, data[ds][ac][bits], DS_COLOR.get(ds, "gray"),
                            BIT_LS.get(bits, "-"), BIT_MARKER.get(bits, "o"),
                            f"{ds} {ac} b{bits}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("QPS")
    ax.set_title("Inline equality-filter tradeoff - all datasets x attr configs x bits")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "tradeoff_all.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = parse_logs()
    if not data:
        print("[error] no valid inline experiment logs found.", file=sys.stderr)
        sys.exit(1)
    write_csv(data, os.path.join(FIG_DIR, "summary.csv"))
    plot_per_dataset(data)
    plot_bit_compare_per_ac(data)
    plot_combined(data)
    print("[done] summary complete.")


if __name__ == "__main__":
    main()
