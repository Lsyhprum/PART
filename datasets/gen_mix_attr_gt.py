#!/usr/bin/env python3
"""
Generate "mixed attribute + filtered ground-truth + queries_mix.json" at a target
selectivity, then self-check.

Mixed attribute = each vector carries **one equality attribute + one range attribute**
(fixed 1+1, no general A extension):
  - base attribute: two independent random streams, column 0 = equality category in [0, C_eq),
    column 1 = range category in [0, C_range); written together to base_attr.bin ((N,2) uint32).
  - query attribute: eq_val in [0, C_eq) + continuous interval [lo, hi] (width w, hi = lo + w - 1);
    written together to query_attr_mix.bin ((QN,3) int32, columns = [eq_val, lo, hi]).
  - filtered gt: within the match set where base equality attribute == eq_val **AND** base range
    attribute falls in [lo, hi], take top-K by metric.
  - selectivity: sel = (1/C_eq) * (w / C_range). Default recipe C_eq=10 x (C_range=100, w=10) = 1%.

Acceleration (same structure as gen_range_attr_gt.py):
  - Heavy work offloaded to multi-threaded C++ (OpenMP): compute_mix_attr_gt applies a
    "equality match AND interval hit" two-condition mask per query, collects matching indices,
    and takes top-(K+M) candidates by double distance within the match set (M = margin slack),
    emitting candidate files + per-query match counts.
  - Python finalize: candidates have heterogeneous base attributes, so the equality mask does
    not apply; instead sort by float32 distance within the candidate subset (matching the
    self-check's float32 top-K recomputed over the full match set, same float32 distance),
    ensuring the generated ground-truth passes the self-check.
  - self_check_mix: 256-sample, independently recomputed with the mixed mask, count check +
    reuse of _dataset_utils.topk_legal_set subset check (1e-6 tolerance).

Seeds are offset (independent of the equality/range attribute streams):
  base  stream = default_rng(seed + 5550000 + C_eq*1009 + C_range*13 + w)
  query stream = default_rng(same + 7654321)

Usage:
  python3 gen_mix_attr_gt.py --ds-dir ../sift --base-name sift_base.fvecs \
      --query-name sift_query.fvecs --C-eq 10 --C-range 100 --w 10 --metric l2 \
      --K 100 --seed 1000 --cpp-bin ./compute_mix_attr_gt --margin 512 --threads 0
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


def run_compute_mix_attr_gt(cpp_bin, base_path, query_path, base_attr_path,
                            query_attr_path, metric, K, margin, threads,
                            cand_path, count_path):
    """Invoke the multi-threaded C++ program to emit candidate files and per-query match counts."""
    cmd = [cpp_bin,
           '--base', base_path, '--query', query_path,
           '--base-attr', base_attr_path, '--query-attr', query_attr_path,
           '--metric', metric, '--K', str(K), '--margin', str(margin),
           '--threads', str(threads),
           '--cand-out', cand_path, '--count-out', count_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('compute_mix_attr_gt failed:\n' + (r.stderr or r.stdout))
    # C++ progress summary is printed to stderr; forward it
    if r.stderr.strip():
        print('   ', r.stderr.strip())


def finalize_via_recompute(cand_path, count_path, X, Q, metric, K, totalcap, N):
    """Read C++ candidates and compute the exact top-K by float32 distance on the candidate subset.

    Key: under the mixed pipeline candidates have heterogeneous base attributes (equality/range
    vary), so the equality recompute_gt mask does not apply; here we sort by distance within the
    candidate subset and take the top K. Because the candidates already contain the true top-K
    (computed in double by C++), this top-K matches the self-check's float32 top-K over the full
    match set (same float32 distance).
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


def self_check_mix(G, X, Q, base_attr, query_attr_mix, metric, K, QN, n=256):
    """Independently recompute with the mixed mask on a sample and compare (subset semantics,
    tie-tolerant), consistent with the verifier.

    mask = (base equality column == query eq_val) AND (base range column in [lo, hi]).
    The boundary legal set reuses _dataset_utils.topk_legal_set (same source as the verifiers).
    """
    rv = np.random.default_rng(20241010)
    sub = rv.choice(QN, size=min(n, QN), replace=False)
    be = base_attr[:, 0]
    br = base_attr[:, 1]
    for i in sub:
        stored = G[i]
        eqv = int(query_attr_mix[i, 0])
        lo = int(query_attr_mix[i, 1])
        hi = int(query_attr_mix[i, 2])
        mask = (be == eqv) & (br >= lo) & (br <= hi)
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


def generate_one(ds_dir, c_eq, c_range, w, metric, K, seed, base_name, query_name,
                 cpp_bin, margin, threads, no_verify=False):
    """Generate all artifacts for a single mixed config of one dataset and self-check.

    Artifacts are written to <ds_dir>/MIX_EQ_C{c_eq}_R{c_range}_w{w}/ :
        base_attr.bin ((N,2) uint32 [eq_cat, range_cat])
        query_attr_mix.bin ((QN,3) int32 [eq_val, lo, hi])
        groundtruth_mix.ivecs  queries_mix.json
        (plus symlinks to the root base/query vectors for convenient re-verification)
    Returns a stats dict.
    """
    assert metric in ('ip', 'l2')
    assert c_eq >= 2, 'C_eq must be >= 2 (equality cardinality)'
    assert c_range >= 2, 'C_range must be >= 2 (range cardinality)'
    assert 1 <= w <= c_range, 'range width w must be within [1, C_range]'
    out_dir = os.path.join(ds_dir, f"MIX_EQ_C{c_eq}_R{c_range}_w{w}")
    os.makedirs(out_dir, exist_ok=True)

    base_path = os.path.join(ds_dir, base_name)
    query_path = os.path.join(ds_dir, query_name)
    print(f'[load] base={base_name} query={query_name} '
          f'(mixed: 1 equality C={c_eq} + 1 range C={c_range} w={w}, metric={metric}, K={K})')
    X = fvecs_read(base_path).astype(np.float32)
    Q = fvecs_read(query_path).astype(np.float32)
    N, D = X.shape
    QN = Q.shape[0]
    print(f'      X={X.shape} Q={Q.shape}')

    # -------- Attributes: two independent random streams (base / query), seeds offset from eq/range --------
    s = seed + 5550000 + c_eq * 1009 + c_range * 13 + w
    rng_b = np.random.default_rng(s)
    eq_cat = rng_b.integers(0, c_eq, size=(N,), dtype=np.int64)          # base equality category
    rng_cat = rng_b.integers(0, c_range, size=(N,), dtype=np.int64)      # base range category
    base_attr = np.ascontiguousarray(np.stack([eq_cat, rng_cat], axis=1),
                                     dtype=np.int64)                     # (N,2)
    base_attr_bin = os.path.join(out_dir, 'base_attr.bin')
    attr_bin_write(base_attr_bin, base_attr, np.uint32)
    print(f'      base_attr={base_attr.shape} (eq in [0,{c_eq}), range in [0,{c_range}))')

    rng_q = np.random.default_rng(s + 7654321)
    eq_val = rng_q.integers(0, c_eq, size=(QN,), dtype=np.int64)         # query equality target
    lo = rng_q.integers(0, c_range - w + 1, size=(QN,), dtype=np.int64)  # interval left end
    hi = lo + (w - 1)                                                    # ensure [lo,hi] subset of [0,C_range-1]
    query_attr_mix = np.ascontiguousarray(np.stack([eq_val, lo, hi], axis=1),
                                          dtype=np.int64)                # (QN,3) [eq,lo,hi]
    query_attr_bin = os.path.join(out_dir, 'query_attr_mix.bin')
    attr_bin_write(query_attr_bin, query_attr_mix, np.int32)
    print(f'      query_attr_mix={query_attr_mix.shape} (C_eq={c_eq}, C_range={c_range}, w={w})')

    # -------- C++ multi-thread: top-(K+M) candidates in the two-condition match set + counts --------
    totalcap = K + margin
    cand_path = os.path.join(out_dir, '.cand.int32')
    count_path = os.path.join(out_dir, '.count.int64')
    run_compute_mix_attr_gt(cpp_bin, base_path, query_path, base_attr_bin,
                            query_attr_bin, metric, K, margin, threads,
                            cand_path, count_path)

    # -------- Python final: exact top-K by float32 distance on candidates --------
    G, sel = finalize_via_recompute(cand_path, count_path, X, Q,
                                    metric, K, totalcap, N)
    os.remove(cand_path)
    os.remove(count_path)
    avg_sel = float(sel.mean())

    # -------- Write ivecs / json --------
    ivecs_write(os.path.join(out_dir, 'groundtruth_mix.ivecs'), G)
    meta = [{'type': 'mix', 'attr': ['eq', 'range'],
             'eq': int(query_attr_mix[i, 0]),
             'range': [int(query_attr_mix[i, 1]), int(query_attr_mix[i, 2])],
             'selectivity': float(sel[i]), 'metric': metric}
            for i in range(QN)]
    with open(os.path.join(out_dir, 'queries_mix.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'      avg selectivity = {avg_sel:.4f} (min {sel.min():.4f}, max {sel.max():.4f}'
          f'  [expected sel = (1/{c_eq})*({w}/{c_range}) = {1.0/c_eq*w/c_range:.4f}]')

    # -------- Symlink root vectors for re-verification --------
    for nm in (base_name, query_name):
        link = os.path.join(out_dir, nm)
        target = os.path.abspath(os.path.join(ds_dir, nm))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)

    # -------- Self-check (optional): independently recompute with mixed mask on a sample --------
    if not no_verify:
        ba = U.attr_bin_read(base_attr_bin, np.uint32).astype(np.int64)
        qa = U.attr_bin_read(query_attr_bin, np.int32).astype(np.int64)
        G_read = ivecs_read(os.path.join(out_dir, 'groundtruth_mix.ivecs')).astype(np.int64)
        assert ba.shape == base_attr.shape and qa.shape == query_attr_mix.shape, 'attribute shape mismatch on read-back'
        assert np.array_equal(ba, base_attr) and np.array_equal(qa, query_attr_mix), 'attribute content mismatch on read-back'
        assert G_read.shape == G.shape, 'gt shape mismatch on read-back'
        ok = self_check_mix(G, X, Q, base_attr, query_attr_mix, metric, K, QN) and \
             self_check_mix(G_read, X, Q, base_attr, query_attr_mix, metric, K, QN)
        assert ok, 'self-check failed: sample differs from mixed-mask recompute (verifier semantics)'
        print(f'      [self-check] PASS  (attribute IO round-trip + 256-sample subset of verifier legal set)')

    return {'ds_dir': ds_dir, 'c_eq': c_eq, 'c_range': c_range, 'w': w,
            'metric': metric, 'K': K, 'N': N, 'QN': QN, 'D': D,
            'avg_sel': avg_sel, 'min_sel': float(sel.min()),
            'max_sel': float(sel.max()), 'out_dir': out_dir,
            'self_check': 'SKIP' if no_verify else 'PASS'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds-dir', required=True, help='dataset root (contains <ds>_base.fvecs / <ds>_query.fvecs)')
    ap.add_argument('--base-name', required=True)
    ap.add_argument('--query-name', required=True)
    ap.add_argument('--C-eq', type=int, default=10, dest='c_eq',
                    help='equality attribute cardinality -> 1/C_eq (default 10)')
    ap.add_argument('--C-range', type=int, default=100, dest='c_range',
                    help='range attribute cardinality -> combined with w gives w/C_range (default 100)')
    ap.add_argument('--w', type=int, default=10,
                    help='continuous categories w covered by the range predicate (clamped to [1, C_range]; sel = (1/C_eq)*(w/C_range), default 10 -> 1%% with C_eq=10)')
    ap.add_argument('--metric', choices=['ip', 'l2'], required=True)
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--cpp-bin', default=os.path.join(HERE, 'compute_mix_attr_gt'),
                    help='path to the compute_mix_attr_gt executable')
    ap.add_argument('--margin', type=int, default=512,
                    help='candidate margin M: candidate count = K+M, covering float-boundary ties to ensure the true top-K is included')
    ap.add_argument('--threads', type=int, default=0,
                    help='OpenMP threads, 0 = use all cores')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip self-check (attribute IO round-trip + mixed-mask sample); only generate artifacts, faster')
    args = ap.parse_args()
    r = generate_one(args.ds_dir, args.c_eq, args.c_range, args.w, args.metric, args.K,
                     args.seed, args.base_name, args.query_name,
                     args.cpp_bin, args.margin, args.threads,
                     no_verify=args.no_verify)
    print('[ok]', json.dumps({k: v for k, v in r.items() if k != 'ds_dir'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
