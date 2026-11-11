#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply a cpr (disjoint-block) single-attribute rotation encoding to the unified dataset
(local copy of symqg, aligned with IVF/RaBitQ-inline-rot).

The only difference from IVF's 3.encode_dataset.py: relax the assertion that
the encoding must preserve unit norm. Reason: symqg's sift / msmarco inputs are
not normalized, while encode_cpr only applies an orthogonal 2D rotation,
which preserves distance (rotated-space L2^2 == original-space L2^2) so the GT
stays valid; normalization is unnecessary (it would change the norm -> invalid GT).
"""

import os
import shutil
import argparse
import numpy as np

from _dataset_utils import (fvecs_read, fvecs_write, attr_bin_read,
                            encode_cpr, cpr_block_split)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=os.path.join(PKG, 'ds'),
                    help='unified dataset directory (default <pkg>/ds)')
    ap.add_argument('--out', default=os.path.join(PKG, 'ds_ro'),
                    help='encoded output directory (default <pkg>/ds_ro)')
    ap.add_argument('--d', type=int, default=1536)
    ap.add_argument('--A', type=int, default=1, help='number of attributes (must match 1.build_dataset.py)')
    ap.add_argument('--enc-seed', type=int, default=1234)
    ap.add_argument('--base-name', default='base_100k.fvecs',
                    help='input base vector file name')
    ap.add_argument('--query-name', default='query_1k.fvecs',
                    help='input query vector file name')
    ap.add_argument('--base-attr-name', default='base_attr_A1_C100.bin',
                    help='input base attribute file name (self-describing bin with header)')
    ap.add_argument('--query-attr-name', default='query_attr_A1_C100.bin',
                    help='input query attribute file name (self-describing bin with header)')
    ap.add_argument('--groundtruth-name', default='openai1536_100k_groundtruth_A1_C100.ivecs',
                    help='input groundtruth file name (with A/C suffix, consistent with 1.build_dataset.py)')
    ap.add_argument('--queries-name', default='queries.json',
                    help='input queries metadata file name')
    ap.add_argument('--rotated-base-name', default='openai1536_100k_rotated_base_A1_C100.fvecs',
                    help='output rotated-encoded base vector file name (with A/C suffix to distinguish attr count / cardinality)')
    ap.add_argument('--rotated-query-name', default='openai1536_100k_rotated_query_A1_C100.fvecs',
                    help='output rotated-encoded query vector file name (with A/C suffix to distinguish attr count / cardinality)')
    args = ap.parse_args()

    COPY_FILES = [args.groundtruth_name]

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    D = args.d
    P = D // 2
    A = args.A
    # Split the P dimension pairs evenly across A attributes with no overlap ->
    # independent multi-attribute encoding (zero cross-block crosstalk).
    blocks = cpr_block_split(A, P)
    seeds = [args.enc_seed + a for a in range(A)]

    print('[1] loading unified dataset from', args.ds, '...')
    X = fvecs_read(os.path.join(args.ds, args.base_name)).astype(np.float32)
    Q = fvecs_read(os.path.join(args.ds, args.query_name)).astype(np.float32)
    attrs = attr_bin_read(os.path.join(args.ds, args.base_attr_name), np.uint32)
    attrs_q = attr_bin_read(os.path.join(args.ds, args.query_attr_name), np.int32)
    targets = attrs_q  # (nq, A) full attribute matrix
    N, _ = X.shape
    assert attrs.shape[1] == A and attrs.shape[0] == N, \
        f'base attribute dim {attrs.shape[1]} != A={A}'
    assert attrs_q.shape[1] == A, f'query attribute dim {attrs_q.shape[1]} != A={A}'
    assert attrs_q.shape[0] == Q.shape[0], 'query attribute count != query count'
    print('      base', X.shape, 'query', Q.shape)

    print('[2] disjoint multi-attr rotation (A=%d) ...' % A)
    E_base = encode_cpr(X, attrs, blocks, seeds)
    E_q = encode_cpr(Q, targets, blocks, seeds)

    nb = np.linalg.norm(E_base, axis=1)
    nq = np.linalg.norm(E_q, axis=1)
    print('      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e'
          % (nb.mean(), nb.std(), nq.mean(), nq.std()))
    # Relaxed check: when inputs are unnormalized (sift/msmarco), the rotated
    # output is still non-unit-norm, but the rotation preserves distance so the GT
    # stays valid.
    if abs(nb.mean() - 1) >= 1e-3 or abs(nq.mean() - 1) >= 1e-3:
        print('[warn] encoded output is not unit-norm (inputs unnormalized); orthogonal rotation preserves distance, original-space GT stays valid.')

    print('[3] writing encoded dataset ...')
    fvecs_write(os.path.join(OUT, args.rotated_base_name), E_base)
    fvecs_write(os.path.join(OUT, args.rotated_query_name), E_q)
    for f in COPY_FILES:
        shutil.copy(os.path.join(args.ds, f), os.path.join(OUT, f))
    print('[ok] encoded dataset ->', OUT, 'base', E_base.shape, 'query', E_q.shape)


if __name__ == '__main__':
    main()
