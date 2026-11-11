#!/usr/bin/env python3
"""Parse the run logs of each C-sweep and A-sweep configuration
([RESULT] / Indexing time / Predicate violations), interpolate to obtain
QPS@0.9 recall, print a summary table and write a JSON file for the plotting
scripts to consume. This script only reads logs and never modifies any library
code."""
import re, json, os

N = 999000  # base dataset size (openai1536)
TARGET_RECALL = 0.9
LOG_DIR = "./_log"

# (name, A, C, logfile)
CONFIGS = [
    ("A1_C100",   1, 100,   "ac_sweep_A1_C100.log"),
    ("A1_C500",   1, 500,   "ac_sweep_A1_C500.log"),
    ("A1_C1000",  1, 1000,  "ac_sweep_A1_C1000_run.log"),
    ("A1_C5000",  1, 5000,  "ac_sweep_A1_C5000.log"),
    ("A1_C10000", 1, 10000, "ac_sweep_A1_C10000.log"),
    ("A1_C10",    1, 10,    "ac_sweep_A1_C10.log"),
    ("A2_C10",    2, 10,    "ac_sweep_A2_C10.log"),
    ("A3_C10",    3, 10,    "ac_sweep_A3_C10.log"),
    ("A4_C10",    4, 10,    "ac_sweep_A4_C10.log"),
]

def parse(path):
    idx_t = None
    pv = []
    rows = []  # (ef, qps, recall)
    with open(path) as f:
        for line in f:
            m = re.search(r"Indexing time\s+([\d.]+)\s+secs", line)
            if m: idx_t = float(m.group(1))
            m = re.search(r"Predicate violations:\s+(\d+)", line)
            if m: pv.append(int(m.group(1)))
            m = re.search(r"\[RESULT\] algo=(\S+) param=(\d+) k=(\d+) qps=([\d.]+) recall=([\d.]+)", line)
            if m: rows.append((int(m.group(2)), float(m.group(4)), float(m.group(5))))
    rows.sort()
    return idx_t, pv, rows

def qps_at_recall(rows, r):
    max_r = max((rc for _, _, rc in rows), default=0.0)
    if max_r < r:
        return None, max_r
    for i in range(1, len(rows)):
        e0, q0, r0 = rows[i-1]
        e1, q1, r1 = rows[i]
        if r0 <= r <= r1 or r1 <= r <= r0:
            if r1 == r0:
                return q0, max_r
            t = (r - r0) / (r1 - r0)
            return q0 + t * (q1 - q0), max_r
    return None, max_r

def main():
    results = []
    for name, A, C, lf in CONFIGS:
        path = os.path.join(LOG_DIR, lf)
        if not os.path.exists(path):
            print(f"[skip] {name}: missing log {path}")
            continue
        idx_t, pv, rows = parse(path)
        if not rows:
            print(f"[skip] {name}: no [RESULT] in {path}")
            continue
        qps90, max_r = qps_at_recall(rows, TARGET_RECALL)
        sel = (1.0 / C) ** A
        gt = int(round(sel * N))
        viol = max(pv) if pv else -1
        results.append({
            "name": name, "A": A, "C": C, "sel": sel, "gt": gt,
            "index_s": idx_t, "qps90": qps90, "maxrecall": max_r, "viol": viol,
        })
    # Print the two summary tables
    print(f"{'config':10} {'A':>3} {'C':>6} {'sel':>9} {'GT~':>8} {'Index(s)':>9} {'QPS@0.9':>9} {'maxRecall':>9} {'Viol':>5}")
    print("-" * 70)
    for sweep, label in [([r for r in results if r["A"] == 1 and r["C"] != 10], "C-sweep"),
                         ([r for r in results if r["C"] == 10], "A-sweep")]:
        print(f"--- {label} ---")
        for r in sweep:
            qps_str = f"{r['qps90']:8.0f}" if r["qps90"] is not None else "  N/A "
            print(f"{r['name']:10} {r['A']:3d} {r['C']:6d} {r['sel']:9.4f} {r['gt']:8d} "
                  f"{r['index_s']:9.1f} {qps_str:>9} {r['maxrecall']:9.4f} {r['viol']:5d}")
    with open(os.path.join(LOG_DIR, "ac_sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", os.path.join(LOG_DIR, "ac_sweep_results.json"))

if __name__ == "__main__":
    main()
