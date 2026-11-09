#!/usr/bin/env python3
"""
Generate "equality attribute + filtered ground-truth + queries.json" at a target
selectivity, then self-check.

Acceleration strategy:
  - Heavy work is offloaded to a multi-threaded C++ program (OpenMP):
    compute_eq_attr_gt groups by the attribute signature (counting-sort CSR),
    and for each query selects top-(K+M) candidates (M = margin slack) within the
    attribute-equal match set, emitting candidate files + per-query match counts.
    See compute_eq_attr_gt.cpp (depends only on libgomp, no GPU).
  - On the Python side, finalize_via_recompute uses _dataset_utils.recompute_gt on
    the candidate set to compute the exact top-K (same logic as the verifier), so the
    generated ground-truth always passes verification.
  - Attribute / vector I/O reuses _dataset_utils attr_bin_write / fvecs_read / ivecs_write.
  - queries.json format aligns with the dataset builder (one predicate recorded per query).

Selectivity config (sel = (1/C)^A; fix C=10, vary A):
  A1_C2  -> 50%   A1_C10 -> 10%   A2_C10 -> 1%   A3_C10 -> 0.1%

Usage:
  python3 gen_eq_attr_gt.py --ds-dir ../sift --A 1 --C 2 --metric l2 \
      --base-name sift_base.fvecs --query-name sift_query.fvecs \
      --cpp-bin ./compute_eq_attr_gt --margin 256 --threads 0

Can also be imported as generate_one(...) for reuse in other pipelines.
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
                            recompute_gt, load_attr, topk_legal_set)


def run_compute_eq_attr_gt(cpp_bin, base_path, query_path, base_attr_path,
                           query_attr_path, metric, K, C, margin, threads,
                           cand_path, count_path):
    """Invoke the multi-threaded C++ program to emit candidate files and per-query match counts.
    C is the attribute cardinality (signature radix); it must match the cardinality used when
    the attributes were generated, otherwise A>=2 causes signature collisions."""
    cmd = [cpp_bin,
           '--base', base_path, '--query', query_path,
           '--base-attr', base_attr_path, '--query-attr', query_attr_path,
           '--metric', metric, '--K', str(K), '--C', str(C),
           '--margin', str(margin),
           '--threads', str(threads),
           '--cand-out', cand_path, '--count-out', count_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('compute_eq_attr_gt failed:\n' + (r.stderr or r.stdout))
    # C++ progress summary is printed to stderr; forward it
    if r.stderr.strip():
        print('   ', r.stderr.strip())


def finalize_via_recompute(cand_path, count_path, X, Q, query_attr, metric, K,
                           totalcap, N):
    """Read C++ candidates and compute the exact top-K on the candidate subset using
    recompute_gt (same logic as the verifier).

    Returns (G int32[QN,K], sel float64[QN]). The candidate margin guarantees the true
    top-K is within the candidates, so the final result is the exact top-K within the
    match set and always passes the independent verifier.
    """
    QN = Q.shape[0]
    cand = np.fromfile(cand_path, dtype=np.int32)
    cand = cand.reshape(QN, totalcap)
    count = np.fromfile(count_path, dtype=np.int64)
    assert count.shape[0] == QN

    G = np.full((QN, K), -1, dtype=np.int32)
    sel = count.astype(np.float64) / N
    A = query_attr.shape[1]
    for i in range(QN):
        c = cand[i]
        c = c[c >= 0]
        if c.size == 0:
            continue
        Xc = np.ascontiguousarray(X[c], dtype=np.float32)
        # Build an "all-matched" subset: every row of base_attr equals query_attr[i],
        # so the mask is True for all candidates.
        bsub = np.broadcast_to(query_attr[i:i + 1], (c.size, A)).copy().astype(np.int64)
        Gc, _ = recompute_gt(Xc, Q[i:i + 1], bsub, bsub, metric, K)
        g0 = Gc[0][Gc[0] >= 0]          # drop -1 padding slots; otherwise c[-1] negative index would repeat the last candidate
        G[i, :g0.size] = c[g0]
    return G, sel


def self_check(G, X, Q, base_attr, query_attr, metric, K, QN, n=256):
    """Verification identical to the independent verifier: 'subset + 1e-6 boundary' check,
    tie-tolerant."""
    rv = np.random.default_rng(20241010)
    sub = rv.choice(QN, size=min(n, QN), replace=False)
    Gref, _ = recompute_gt(X, Q[sub], base_attr, query_attr[sub], metric, K)
    for j, i in enumerate(sub):
        stored = G[i]
        mask = np.all(base_attr == query_attr[i], axis=1)
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


def generate_one(ds_dir, A, C, metric, K, seed, base_name, query_name,
                 cpp_bin, margin, threads, no_verify=False):
    """Generate all artifacts for a single (A, C) config of one dataset and self-check.

    Artifacts are written to <ds_dir>/A{A}_C{C}/ :
        base_attr.bin  query_attr.bin  groundtruth.ivecs  queries.json
        (plus symlinks to the root base/query vectors for convenient re-verification)
    Returns a stats dict.
    """
    assert metric in ('ip', 'l2')
    out_dir = os.path.join(ds_dir, f"A{A}_C{C}")
    os.makedirs(out_dir, exist_ok=True)

    base_path = os.path.join(ds_dir, base_name)
    query_path = os.path.join(ds_dir, query_name)
    print(f'[load] base={base_name} query={query_name} (A={A}, C={C}, metric={metric}, K={K})')
    X = fvecs_read(base_path).astype(np.float32)
    Q = fvecs_read(query_path).astype(np.float32)
    N, D = X.shape
    QN = Q.shape[0]
    print(f'      X={X.shape} Q={Q.shape}')

    # -------- Attributes: independent random stream, reproducible --------
    rng = np.random.default_rng(seed + A * 100 + C)
    base_attr = rng.integers(0, C, size=(N, A), dtype=np.int64)
    query_attr = rng.integers(0, C, size=(QN, A), dtype=np.int64)
    base_attr_bin = os.path.join(out_dir, 'base_attr.bin')
    query_attr_bin = os.path.join(out_dir, 'query_attr.bin')
    attr_bin_write(base_attr_bin, base_attr, np.uint32)
    attr_bin_write(query_attr_bin, query_attr, np.int32)
    print(f'      base_attr={base_attr.shape} query_attr={query_attr.shape} (C={C})')

    # -------- C++ multi-thread: top-(K+M) candidates in the match set + match counts --------
    totalcap = K + margin
    cand_path = os.path.join(out_dir, '.cand.int32')
    count_path = os.path.join(out_dir, '.count.int64')
    run_compute_eq_attr_gt(cpp_bin, base_path, query_path, base_attr_bin,
                           query_attr_bin, metric, K, C, margin, threads,
                           cand_path, count_path)

    # -------- Python final: exact top-K on candidates via recompute_gt (same as verifier) --------
    G, sel = finalize_via_recompute(cand_path, count_path, X, Q, query_attr,
                                    metric, K, totalcap, N)
    os.remove(cand_path)
    os.remove(count_path)
    avg_sel = float(sel.mean())

    # -------- Write ivecs / json --------
    ivecs_write(os.path.join(out_dir, 'groundtruth.ivecs'), G)
    meta = [{'type': 'eq', 'attr': list(range(A)),
             'value': [int(v) for v in query_attr[i]],
             'selectivity': float(sel[i]), 'metric': metric}
            for i in range(QN)]
    with open(os.path.join(out_dir, 'queries.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'      avg selectivity = {avg_sel:.4f} (min {sel.min():.4f}, max {sel.max():.4f})')

    # -------- Symlink root vectors for re-verification --------
    for nm in (base_name, query_name):
        link = os.path.join(out_dir, nm)
        target = os.path.abspath(os.path.join(ds_dir, nm))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)

    # -------- Self-check (optional): read attributes back, compare against recompute_gt sample --------
    if not no_verify:
        ba = load_attr(base_attr_bin, np.uint32, A_default=A).astype(np.int64)
        qa = load_attr(query_attr_bin, np.int32, A_default=A).astype(np.int64)
        G_read = ivecs_read(os.path.join(out_dir, 'groundtruth.ivecs')).astype(np.int64)
        assert ba.shape == base_attr.shape and qa.shape == query_attr.shape, 'attribute shape mismatch on read-back'
        assert np.array_equal(ba, base_attr) and np.array_equal(qa, query_attr), 'attribute content mismatch on read-back'
        assert G_read.shape == G.shape, 'gt shape mismatch on read-back'
        ok = self_check(G, X, Q, base_attr, query_attr, metric, K, QN) and \
             self_check(G_read, X, Q, base_attr, query_attr, metric, K, QN)
        assert ok, 'self-check failed: sample differs from recompute_gt (verifier semantics)'
        print(f'      [self-check] PASS  (attribute IO round-trip + 256-sample subset of verifier legal set)')

    return {'ds_dir': ds_dir, 'A': A, 'C': C, 'metric': metric, 'K': K,
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
    ap.add_argument('--C', type=int, required=True, help='attribute cardinality -> selectivity 1/C^A')
    ap.add_argument('--metric', choices=['ip', 'l2'], required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--cpp-bin', default=os.path.join(HERE, 'compute_eq_attr_gt'),
                    help='path to the compute_eq_attr_gt executable')
    ap.add_argument('--margin', type=int, default=512,
                    help='candidate margin M: candidate count = K+M, covering float-boundary ties to ensure the true top-K is included')
    ap.add_argument('--threads', type=int, default=0,
                    help='OpenMP threads, 0 = use all cores')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip self-check (attribute IO round-trip + recompute_gt sample); only generate artifacts, faster')
    args = ap.parse_args()

    r = generate_one(args.ds_dir, args.A, args.C, args.metric, args.K,
                     args.seed, args.base_name, args.query_name,
                     args.cpp_bin, args.margin, args.threads,
                     no_verify=args.no_verify)
    print('[ok]', json.dumps({k: v for k, v in r.items() if k != 'ds_dir'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
