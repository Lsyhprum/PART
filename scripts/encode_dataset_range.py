#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encode_dataset_range.py -- range-attribute rotation encoding generator
(rotation recall enhancement).

Apply a numeric-range RoPE rotation encoding to the unified dataset:
  - base  : RoPE rotation by numeric attribute attrs_num (rope_encode_cpr)
            -> data point phase proportional to value/L; close values -> close phases
  - query : for each range predicate [lo,hi], split D3 into M sub-windows and apply
            the same-phase rotation to each sub-window center
            (rope_query_range_cpr) -> outputs Nq*M rotated query rows

Interface with the C++ search side (key):
  - This generator only produces "rotated base/query fvecs"; the Symqg graph is
    built on the rotated base, and search uses the rotated query rows (Nq*M rows,
    ordered i*M+s) to perform M graph searches and union candidates, then
    rerank with original-space L2^2 + attr_match_range exact gate (rerank stage).
  - lo == -1 means this dimension is unconstrained -> all sub-centers for that
    query are left unrotated (identity) but still occupy M rows as placeholders.

Design constraints:
  - Reuse _dataset_utils' RoPE primitives (single mathematical source).
  - v1 only supports single attribute A=1; input vectors must already be
    unit-norm (rotation preserves magnitude, encoded output must stay unit-norm).
"""
import os
import json
import argparse

import numpy as np

from _dataset_utils import (fvecs_read, fvecs_write, attr_bin_read,
                            rope_encode_cpr, rope_query_range_cpr,
                            cpr_block_split)


def split_window(lo, hi, M):
    """D3 equal split: cut integer window [lo,hi] into M sub-windows, return
    [(slo, shi), ...]. Adjacent sub-windows are end-to-end; for M>=1 the union
    is still [lo,hi]."""
    out = []
    for s in range(M):
        slo = lo + (hi - lo) * s // M
        shi = lo + (hi - lo) * (s + 1) // M
        out.append((slo, shi))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="range-attribute rotation encoding generator (base RoPE + query M subcenters)")
    ap.add_argument('--ds', required=True, help='unified dataset directory (contains base/query/attr)')
    ap.add_argument('--out', required=True, help='rotated artifact output directory (same dir as --ds is fine)')
    ap.add_argument('--d', type=int, default=1536, help='vector dimension')
    ap.add_argument('--A', type=int, default=1, help='number of attributes (v1 supports A=1 only)')
    ap.add_argument('--L', type=int, default=50, help='numeric attribute upper bound (phase denominator, = cardinality C)')
    ap.add_argument('--base-freq', type=float, default=1.0, help='RoPE frequency scale')
    ap.add_argument('--M', type=int, default=1, help='number of sub-window centers per query (must match C++ --range_m)')
    ap.add_argument('--base-name', required=True, help='input base vector file name')
    ap.add_argument('--query-name', required=True, help='input query vector file name')
    ap.add_argument('--base-attr-name', required=True, help='input base numeric attribute file name')
    ap.add_argument('--query-range-name', required=True, help='input query range predicate file name (Nq,2A)')
    ap.add_argument('--rotated-base-name', required=True, help='output rotated base file name')
    ap.add_argument('--rotated-query-name', required=True, help='output rotated query file name (Nq*M rows)')
    args = ap.parse_args()

    assert args.A == 1, "v1 range rotation enhancement supports single attribute A=1 only"
    assert args.M >= 1, "--M must be >= 1"
    D = args.d
    P = D // 2
    L = args.L
    M = args.M
    DS = args.ds
    OUT = args.out

    print('[1] loading raw dataset from', DS, '...')
    X = fvecs_read(os.path.join(DS, args.base_name)).astype(np.float32)
    Q = fvecs_read(os.path.join(DS, args.query_name)).astype(np.float32)
    N = X.shape[0]
    Nq = Q.shape[0]
    attrs_num = attr_bin_read(os.path.join(DS, args.base_attr_name), np.uint32, n=N, a=args.A)   # (N, A)
    qrange = attr_bin_read(os.path.join(DS, args.query_range_name), np.int32, n=Nq, a=2 * args.A)  # (Nq, 2A)
    assert attrs_num.shape == (N, args.A), f'base attribute shape {attrs_num.shape} should be (N={N}, A={args.A})'
    assert qrange.shape == (Nq, 2 * args.A), f'query_range shape {qrange.shape} should be (Nq={Nq}, 2A={2*args.A})'
    print('      base', X.shape, 'query', Q.shape,
          'base_attr', attrs_num.shape, 'query_range', qrange.shape)

    blocks = cpr_block_split(args.A, P)

    print('[2] range-rotating base (rope_encode_cpr, A=%d, L=%d) ...' % (args.A, L))
    E_base = rope_encode_cpr(X, attrs_num, L, blocks, base_freq=args.base_freq)

    print('[3] rotating queries: %d logical x M=%d subcenters = %d rows ...' % (Nq, M, Nq * M))
    E_q = np.empty((Nq * M, D), dtype=np.float32)
    n_unconstrained = 0
    for i in range(Nq):
        lo = int(qrange[i, 0])
        hi = int(qrange[i, 1])
        if lo == -1:                       # unconstrained dim -> identity, still occupy M rows
            n_unconstrained += 1
            for s in range(M):
                E_q[i * M + s] = Q[i]
        else:
            for s, (slo, shi) in enumerate(split_window(lo, hi, M)):
                E_q[i * M + s] = rope_query_range_cpr(
                    Q[i], slo, shi, L, blocks[0], base_freq=args.base_freq)
    print('      unconstrained queries (lo==-1):', n_unconstrained, '/', Nq)

    # Unit-norm sanity (orthogonal rotation preserves magnitude)
    nb = np.linalg.norm(E_base, axis=1)
    nq = np.linalg.norm(E_q, axis=1)
    print('      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e'
          % (nb.mean(), nb.std(), nq.mean(), nq.std()))
    assert abs(nb.mean() - 1) < 1e-3 and abs(nq.mean() - 1) < 1e-3, \
        'encoding must preserve unit norm (ensure input base/query are normalized)'

    print('[4] writing range-rotated dataset ...')
    fvecs_write(os.path.join(OUT, args.rotated_base_name), E_base)
    fvecs_write(os.path.join(OUT, args.rotated_query_name), E_q)

    meta = {
        'D': D, 'A': args.A, 'L': L, 'BASE_FREQ': args.base_freq, 'M': M,
        'n_base': int(N), 'n_query_logical': int(Nq), 'n_query_rows': int(Nq * M),
        'n_unconstrained': int(n_unconstrained),
        'rotated_base': args.rotated_base_name,
        'rotated_query': args.rotated_query_name,
        'note': 'range RoPE rotation: base by attr value, query by M equal-split '
                'sub-window centers (D3). C++ search uses --range_m=%d to scan M '
                'subcenters per logical query and union candidates before exact '
                'rerank.' % M,
    }
    with open(os.path.join(OUT, 'encode_range_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('[ok] range-rotated dataset ->', OUT, 'base', E_base.shape, 'query', E_q.shape)


if __name__ == '__main__':
    main()
