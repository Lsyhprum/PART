#!/usr/bin/env python3
"""
Generate "range attribute + filtered ground-truth + queries_range.json" at a target
selectivity, then self-check.

Same structure as gen_eq_attr_gt.py (equality attribute), except the query predicate
becomes a continuous category interval [lo, hi]:
  - base attribute: independently random categorical labels (N, A), cardinality C (C=100 for range).
  - query attribute: one [lo, hi] range predicate per attribute, (QN, 2A) interleaved
    [lo_a, hi_a], int32.
  - filtered gt: within the match set where every base attribute falls inside its query
    interval (multi-attribute AND), take top-K by metric.
  - expected sel = (w / C)^A, controlled by --w (w = number of continuous categories covered
    per attribute; w>1 gives a real interval).

Acceleration:
  - Heavy work offloaded to multi-threaded C++ (OpenMP): compute_range_attr_gt applies a
    [lo,hi] interval mask per query over the base, collects matching indices, and takes
    top-(K+M) candidates by float32 distance within the match set (M = margin slack), emitting
    candidate files + per-query match counts.
  - Python finalize: candidates have heterogeneous base attributes, so the equality mask does
    not apply; instead sort by float32 distance directly within the candidate subset (matching
    the verifier's float32 top-K recomputed over the full match set), ensuring the generated
    ground-truth passes verification.
  - queries_range.json format aligns with the range dataset builder / verifier
    (type='range', one [lo,hi] interval predicate per query).

Usage:
  python3 gen_range_attr_gt.py --ds-dir ../sift --A 1 --C 100 --w 50 --metric l2 \
      --base-name sift_base.fvecs --query-name sift_query.fvecs \
      --cpp-bin ./compute_range_attr_gt --margin 512 --threads 0
"""
import os
import sys
import json
import argparse
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _dataset_utils as U
from _dataset_utils import (attr_bin_write, fvecs_read, ivecs_read, ivecs_write,
                            topk_legal_set)

def run_compute_range_attr_gt(cpp_bin, base_path, query_path, base_attr_path,
                              query_attr_path, metric, K, margin, threads,
                              cand_path, count_path):
    """Invoke the multi-threaded range matcher to emit candidate files and per-query match counts."""
    cmd = [cpp_bin,
           '--base', base_path, '--query', query_path,
           '--base-attr', base_attr_path, '--query-attr', query_attr_path,
           '--metric', metric, '--K', str(K), '--margin', str(margin),
           '--threads', str(threads),
           '--cand-out', cand_path, '--count-out', count_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('compute_range_attr_gt failed:\n' + (r.stderr or r.stdout))
    if r.stderr.strip():
        print('   ', r.stderr.strip())


def finalize_via_recompute(cand_path, count_path, X, Q, base_attr, metric, K,
                           totalcap, N):
    """Read C++ candidates and compute the exact top-K by float32 distance on the candidate
    subset, consistent with the verifier.

    Key: under the range pipeline candidates have heterogeneous base attributes, so the
    equality recompute_gt mask (which requires all candidates to share a signature) cannot be
    reused. Here we simply sort by distance within the candidate subset and take the top K;
    because the candidates already contain the true top-K (computed exactly in double by C++),
    this top-K matches the verifier's float32 top-K over the full match set (same float32 distance).
    Returns (G int32[QN,K], sel float64[QN]).
    """
    QN = Q.shape[0]
    cand = np.fromfile(cand_path, dtype=np.int32)
    cand = cand.reshape(QN, totalcap)
    count = np.fromfile(count_path, dtype=np.int64)
    assert count.shape[0] == QN

    G = np.full((QN, K), -1, dtype=np.int32)
    sel = count.astype(np.float64) / N
    for i in range(QN):
        c = cand[i]
        c = c[c >= 0]
        if c.size == 0:
            continue
        Xc = np.ascontiguousarray(X[c], dtype=np.float32)
        if metric == 'ip':
            score = Q[i] @ Xc.T
            order = np.argsort(-score)[:min(K, c.size)]
        else:
            diff = Xc - Q[i]
            score = np.einsum('ij,ij->i', diff, diff)
            order = np.argsort(score)[:min(K, c.size)]
        G[i, :len(order)] = c[order]
    return G, sel


def self_check_range(G, X, Q, base_attr, query_ranges, metric, K, QN, n=256):
    """Independently recompute with the range mask on a sample and compare (subset semantics,
    tie-tolerant), consistent with the verifier."""
    A = base_attr.shape[1]
    rv = np.random.default_rng(20241010)
    sub = rv.choice(QN, size=min(n, QN), replace=False)
    for i in sub:
        stored = G[i]
        rng_i = query_ranges[i].reshape(A, 2)
        mask = np.all((base_attr >= rng_i[:, 0]) & (base_attr <= rng_i[:, 1]), axis=1)
        m_idx = np.where(mask)[0]
        expect_n = min(K, m_idx.size)
        valid = stored[stored >= 0]
        if valid.size != expect_n:
            return False
        if m_idx.size > 0 and valid.size > 0:
            legal = topk_legal_set(X, Q[i], m_idx, metric, expect_n)
            if not set(valid.tolist()).issubset(legal):
                return False
    return True


def generate_one(ds_dir, A, C, w, metric, K, seed, base_name, query_name,
                 cpp_bin, margin, threads, no_verify=False):
    """Generate all artifacts for a single (A, C, w) range config of one dataset and self-check.

    Artifacts are written to <ds_dir>/A{A}_C{C}_w{w}/ :
        base_attr.bin  query_attr_range.bin  groundtruth_range.ivecs  queries_range.json
        (plus symlinks to the root base/query vectors for convenient re-verification)
    Returns a stats dict.
    """
    assert metric in ('ip', 'l2')
    assert 1 <= w <= C, 'range width w must be within [1, C]'
    out_dir = os.path.join(ds_dir, f"A{A}_C{C}_w{w}")
    os.makedirs(out_dir, exist_ok=True)

    base_path = os.path.join(ds_dir, base_name)
    query_path = os.path.join(ds_dir, query_name)
    print(f'[load] base={base_name} query={query_name} (A={A}, C={C}, w={w}, metric={metric}, K={K})')
    X = fvecs_read(base_path).astype(np.float32)
    Q = fvecs_read(query_path).astype(np.float32)
    N, D = X.shape
    QN = Q.shape[0]
    print(f'      X={X.shape} Q={Q.shape}')

    # -------- base attribute: independent random stream, reproducible (C=100 categorical labels) --------
    rng = np.random.default_rng(seed + A * 10007 + C * 131 + w)
    base_attr = rng.integers(0, C, size=(N, A), dtype=np.int64)
    base_attr_bin = os.path.join(out_dir, 'base_attr.bin')
    attr_bin_write(base_attr_bin, base_attr, np.uint32)
    print(f'      base_attr={base_attr.shape} (C={C})')

    # -------- query range attribute: independent random stream, [lo, hi] continuous interval (w categories) --------
    rng_q = np.random.default_rng(seed + A * 10007 + C * 131 + w + 1234567)
    lo = rng_q.integers(0, C - w + 1, size=(QN, A), dtype=np.int64)
    hi = lo + (w - 1)                                  # ensure [lo, hi] subset of [0, C-1]
    # (QN, 2A) interleaved layout: columns [lo_0, hi_0, lo_1, hi_1, ...]; attr_bin_write records header A as 2A.
    qrange = np.ascontiguousarray(
        np.stack([lo, hi], axis=2).reshape(QN, 2 * A), dtype=np.int64)
    query_attr_bin = os.path.join(out_dir, 'query_attr_range.bin')
    attr_bin_write(query_attr_bin, qrange, np.int32)
    print(f'      query ranges={qrange.shape} (C={C}, w={w})')

    # -------- C++ multi-thread: top-(K+M) candidates in the match set + match counts --------
    totalcap = K + margin
    cand_path = os.path.join(out_dir, '.cand.int32')
    count_path = os.path.join(out_dir, '.count.int64')
    run_compute_range_attr_gt(cpp_bin, base_path, query_path, base_attr_bin,
                              query_attr_bin, metric, K, margin, threads,
                              cand_path, count_path)

    # -------- Python final: exact top-K on candidates --------
    G, sel = finalize_via_recompute(cand_path, count_path, X, Q, base_attr,
                                    metric, K, totalcap, N)
    os.remove(cand_path)
    os.remove(count_path)
    avg_sel = float(sel.mean())

    # -------- Write ivecs / json --------
    ivecs_write(os.path.join(out_dir, 'groundtruth_range.ivecs'), G)
    meta = [{'type': 'range', 'attr': list(range(A)),
             'value': [[int(qrange[i, 2 * a]), int(qrange[i, 2 * a + 1])]
                       for a in range(A)],
             'selectivity': float(sel[i]), 'metric': metric}
            for i in range(QN)]
    with open(os.path.join(out_dir, 'queries_range.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'      avg selectivity = {avg_sel:.4f} (min {sel.min():.4f}, max {sel.max():.4f})')

    # -------- Symlink root vectors for re-verification --------
    for nm in (base_name, query_name):
        link = os.path.join(out_dir, nm)
        target = os.path.abspath(os.path.join(ds_dir, nm))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)

    # -------- Self-check (optional): attribute IO round-trip + range-mask sample comparison --------
    if not no_verify:
        ba = U.attr_bin_read(base_attr_bin, np.uint32).astype(np.int64)
        qa = U.attr_bin_read(query_attr_bin, np.int32).astype(np.int64)
        G_read = ivecs_read(os.path.join(out_dir, 'groundtruth_range.ivecs')).astype(np.int64)
        assert ba.shape == base_attr.shape and qa.shape == qrange.shape, 'attribute shape mismatch on read-back'
        assert np.array_equal(ba, base_attr) and np.array_equal(qa, qrange), 'attribute content mismatch on read-back'
        assert G_read.shape == G.shape, 'gt shape mismatch on read-back'
        ok = self_check_range(G, X, Q, base_attr, qrange, metric, K, QN) and \
             self_check_range(G_read, X, Q, base_attr, qrange, metric, K, QN)
        assert ok, 'self-check failed: sample differs from range-mask recompute (verifier semantics)'
        print(f'      [self-check] PASS  (attribute IO round-trip + 256-sample subset of verifier legal set)')

    return {'ds_dir': ds_dir, 'A': A, 'C': C, 'w': w, 'metric': metric, 'K': K,
            'N': N, 'QN': QN, 'D': D,
            'avg_sel': avg_sel, 'min_sel': float(sel.min()),
            'max_sel': float(sel.max()), 'out_dir': out_dir,
            'self_check': 'SKIP' if no_verify else 'PASS'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds-dir', required=True, help='dataset root (contains <ds>_base.fvecs / <ds>_query.fvecs)')
    ap.add_argument('--base-name', required=True)
    ap.add_argument('--query-name', required=True)
    ap.add_argument('--A', type=int, required=True, help='number of attributes')
    ap.add_argument('--C', type=int, default=100, help='attribute cardinality -> selectivity (w/C)^A')
    ap.add_argument('--w', type=int, required=True, help='continuous categories covered per attribute (range width, clamped to [1,C])')
    ap.add_argument('--metric', choices=['ip', 'l2'], required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--cpp-bin', default=os.path.join(HERE, 'compute_range_attr_gt'),
                    help='path to the compute_range_attr_gt executable')
    ap.add_argument('--margin', type=int, default=512,
                    help='candidate margin M: candidate count = K+M, covering float-boundary ties to ensure the true top-K is included')
    ap.add_argument('--threads', type=int, default=0,
                    help='OpenMP threads, 0 = use all cores')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip self-check (attribute IO round-trip + range-mask sample); only generate artifacts, faster')
    args = ap.parse_args()

    r = generate_one(args.ds_dir, args.A, args.C, args.w, args.metric, args.K,
                     args.seed, args.base_name, args.query_name,
                     args.cpp_bin, args.margin, args.threads,
                     no_verify=args.no_verify)
    print('[ok]', json.dumps({k: v for k, v in r.items() if k != 'ds_dir'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
