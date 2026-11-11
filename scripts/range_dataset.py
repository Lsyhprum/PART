#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
range_dataset.py -- range-attribute predicate bin + range-filter groundtruth generator.

Reusing the existing base_attr (provided by base_attr_A{ATTR_DIM}_C{ATTR_CARD}.bin)
as the database attribute groundtruth, generate:
  - query_attr_A*_C*_range.bin : (Nq, 2A) interleaved [lo_a, hi_a] int32 range
    predicates (header A denotes 2A; lo=-1 means this dimension is unconstrained)
  - {ORIG_DATASET}_groundtruth_range_A*_C*.ivecs : range-filter groundtruth
    (the set of base points whose attribute falls within each query interval
    [lo,hi], sorted by L2 distance and truncated to top-K)

Difference from reference library 1.range_dataset.py: base attributes are NOT
regenerated randomly; the existing base_attr is reused directly, ensuring the
same groundtruth is used by the graph-kernel exact gate and the rotation encoding
(critical -- otherwise GT and predicate/encoding would be inconsistent).
"""
import os
import argparse
import numpy as np

from _dataset_utils import fvecs_read, fvecs_write, ivecs_write, attr_bin_read, attr_bin_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-base', required=True, help='base fvecs path (rerank/GT distance)')
    ap.add_argument('--src-query', required=True, help='query fvecs path')
    ap.add_argument('--out-dir', required=True, help='output directory')
    ap.add_argument('--base-attr-name', required=True, help='existing base attribute bin (with 8-byte header)')
    ap.add_argument('--query-attr-name', required=True, help='output range predicate bin name (Nq x 2A)')
    ap.add_argument('--groundtruth-name', required=True, help='output range GT name')
    ap.add_argument('--A', type=int, default=1, help='number of attributes')
    ap.add_argument('--card', type=int, default=50, help='per-attribute cardinality C -> selectivity 1/card')
    ap.add_argument('--range-w', type=int, default=5,
                    help='range coverage of consecutive categories w (clamped to [1, card]; w=1 degrades to equality)')
    ap.add_argument('--K', type=int, default=100, help='filter groundtruth / Recall@K')
    ap.add_argument('--n', type=int, default=100000, help='take first N base rows')
    ap.add_argument('--nq', type=int, default=1000, help='number of queries')
    ap.add_argument('--d', type=int, default=1536, help='vector dimension')
    ap.add_argument('--metric', choices=['ip', 'l2'], default='l2', help='distance metric')
    args = ap.parse_args()

    OUT_DIR = args.out_dir
    os.makedirs(OUT_DIR, exist_ok=True)

    print('[1/3] loading base/query from', args.src_base, '/', args.src_query)
    X = fvecs_read(args.src_base).astype(np.float32)[:args.n]
    Q = fvecs_read(args.src_query).astype(np.float32)[:args.nq]
    N, Nq = X.shape[0], Q.shape[0]
    assert X.shape[1] == args.d and Q.shape[1] == args.d
    print('      base', X.shape, 'query', Q.shape)

    print('[2/3] loading base attributes (reuse existing):', args.base_attr_name)
    # base_attr may be in headerless raw format: adaptively detect the 8-byte
    # header using the known N/A.
    attrs = attr_bin_read(os.path.join(OUT_DIR, args.base_attr_name), np.uint32, n=N, a=args.A)  # (N, A)
    assert attrs.shape[0] == N and attrs.shape[1] == args.A, \
        f'base_attr shape {attrs.shape} incompatible with N={N}, A={args.A}'
    print('      base attrs', attrs.shape, 'card', args.card)

    # ---------------- query range predicates [lo, hi] ----------------
    rng_q = np.random.default_rng(2000)
    w = min(max(args.range_w, 1), args.card)
    lo = rng_q.integers(0, args.card - w + 1, size=(Nq, args.A), dtype=np.int64)
    hi = lo + (w - 1)
    qrange = np.ascontiguousarray(
        np.stack([lo, hi], axis=2).reshape(Nq, 2 * args.A), dtype=np.int64)
    attr_bin_write(os.path.join(OUT_DIR, args.query_attr_name), qrange, np.int32)
    print('      query ranges', qrange.shape, 'card', args.card, 'range-w', w)

    # ---------------- range-filter groundtruth ----------------
    print('[3/3] building range filtered groundtruth (metric=%s) ...' % args.metric)
    G = np.full((Nq, args.K), -1, dtype=np.int32)
    sel = np.zeros(Nq, dtype=np.float64)
    for i in range(Nq):
        rng_i = qrange[i].reshape(args.A, 2)
        mask = np.all((attrs >= rng_i[:, 0]) & (attrs <= rng_i[:, 1]), axis=1)
        m_idx = np.where(mask)[0]
        sel[i] = m_idx.size / N
        if m_idx.size == 0:
            continue
        if args.metric == 'ip':
            score = Q[i] @ X[m_idx].T
            order = np.argsort(-score)[:min(args.K, m_idx.size)]
        else:
            diff = X[m_idx] - Q[i]
            score = np.einsum('ij,ij->i', diff, diff)   # L2^2
            order = np.argsort(score)[:min(args.K, m_idx.size)]
        G[i, :len(order)] = m_idx[order]
    ivecs_write(os.path.join(OUT_DIR, args.groundtruth_name), G)
    print('      avg selectivity = %.4f (min %.4f, max %.4f)'
          % (sel.mean(), sel.min(), sel.max()))
    print('[ok] range dataset ->', OUT_DIR)


if __name__ == '__main__':
    main()
