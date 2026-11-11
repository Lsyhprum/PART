#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply an APR/CDMA phase-superposition encoding to the unified dataset
(local copy of symqg).

This script is a thin wrapper: it prefers the C++ encoder bin/encode_apr
(several orders of magnitude faster; arxiv D=4096/1M drops from minutes to
seconds~minutes). When the C++ binary is missing it compiles on the fly, and on
compile/run failure it falls back to the pure-Python implementation
(_dataset_utils.encode_apr), so the existing sift/msmarco/openai1536 pipeline
never regresses. The CLI is identical to the Python original, so run.sh calls
are unchanged.

The salt is generated precisely by numpy (seeds[a]=enc_seed+a, integers(0,2**63))
and passed to C++ via --salt-file, keeping continuity with existing APR results
and byte-identical reruns.
"""

import os
import sys
import argparse
import subprocess
import tempfile

import numpy as np

from _dataset_utils import (fvecs_read, fvecs_write, attr_bin_read, encode_apr)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(PKG, 'bin', 'encode_apr')
CPP_SRC = os.path.join(SCRIPT_DIR, 'encode_apr.cpp')


def build_binary():
    """Try to compile when the C++ binary is missing; return True on success."""
    if os.path.exists(BIN):
        return True
    print('[encode] C++ binary missing, trying to compile %s ...' % BIN, file=sys.stderr)
    try:
        subprocess.run(
            ['g++', '-std=c++17', '-o', BIN, CPP_SRC, '-I', SCRIPT_DIR,
             '-Ofast', '-march=core-avx2', '-pthread',
             '-fno-fast-math', '-ffp-contract=off'],
            check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print('[encode] compile failed, falling back to Python: %s' % e, file=sys.stderr)
        return False
    return os.path.exists(BIN)


def run_cpp(args, threads):
    P = args.d // 2
    seeds = [args.enc_seed + a for a in range(args.A)]
    salt = np.concatenate([
        np.random.default_rng(s).integers(0, 2 ** 63, size=P, dtype=np.int64)
        for s in seeds
    ]).astype('<i8')
    tmp = None
    try:
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.salt')
        tmp = tf.name
        salt.tofile(tf)
        tf.close()
        cmd = [
            BIN, '--ds', args.ds, '--out', args.out, '--d', str(args.d),
            '--A', str(args.A), '--enc-seed', str(args.enc_seed),
            '--base-name', args.base_name, '--query-name', args.query_name,
            '--base-attr-name', args.base_attr_name,
            '--query-attr-name', args.query_attr_name,
            '--groundtruth-name', args.groundtruth_name,
            '--apr-base-name', args.apr_base_name,
            '--apr-query-name', args.apr_query_name,
            '--salt-file', tmp, '--threads', str(threads),
        ]
        rc = subprocess.run(cmd).returncode
        return rc == 0
    except Exception as e:  # noqa: BLE001
        print('[encode] C++ run error, falling back to Python: %s' % e, file=sys.stderr)
        return False
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def python_fallback(args):
    """Original Python implementation (same behavior as old encode_attr_apr.py)."""
    COPY_FILES = [args.groundtruth_name]
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    D = args.d
    A = args.A
    seeds = [args.enc_seed + a for a in range(A)]

    print('[1] loading unified dataset from', args.ds, '...')
    X = fvecs_read(os.path.join(args.ds, args.base_name)).astype(np.float32)
    Q = fvecs_read(os.path.join(args.ds, args.query_name)).astype(np.float32)
    attrs = attr_bin_read(os.path.join(args.ds, args.base_attr_name), np.uint32)
    attrs_q = attr_bin_read(os.path.join(args.ds, args.query_attr_name), np.int32)
    targets = attrs_q  # (nq, A) full attribute matrix
    N, _ = X.shape
    assert attrs.shape[1] == A and attrs.shape[0] == N, \
        'base attribute dim %s != A=%d' % (attrs.shape[1], A)
    assert attrs_q.shape[1] == A, 'query attribute dim %s != A=%d' % (attrs_q.shape[1], A)
    assert attrs_q.shape[0] == Q.shape[0], 'query attribute count != query count'
    print('      base', X.shape, 'query', Q.shape)

    print('[2] apr multi-attr rotation (A=%d) ...' % A)
    E_base = encode_apr(X, attrs, seeds)
    E_q = encode_apr(Q, targets, seeds)

    nb = np.linalg.norm(E_base, axis=1)
    nq = np.linalg.norm(E_q, axis=1)
    print('      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e'
          % (nb.mean(), nb.std(), nq.mean(), nq.std()))
    if abs(nb.mean() - 1) >= 1e-3 or abs(nq.mean() - 1) >= 1e-3:
        print('[warn] encoded output is not unit-norm (inputs unnormalized); orthogonal rotation preserves distance, original-space GT stays valid.')

    print('[3] writing encoded dataset ...')
    fvecs_write(os.path.join(OUT, args.apr_base_name), E_base)
    fvecs_write(os.path.join(OUT, args.apr_query_name), E_q)
    for f in COPY_FILES:
        import shutil
        shutil.copy(os.path.join(args.ds, f), os.path.join(OUT, f))
    print('[ok] encoded dataset ->', OUT, 'base', E_base.shape, 'query', E_q.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=os.path.join(PKG, 'ds'))
    ap.add_argument('--out', default=os.path.join(PKG, 'ds_apr'))
    ap.add_argument('--d', type=int, default=1536)
    ap.add_argument('--A', type=int, default=1)
    ap.add_argument('--enc-seed', type=int, default=1234)
    ap.add_argument('--base-name', default='base_100k.fvecs')
    ap.add_argument('--query-name', default='query_1k.fvecs')
    ap.add_argument('--base-attr-name', default='base_attr_A1_C100.bin')
    ap.add_argument('--query-attr-name', default='query_attr_A1_C100.bin')
    ap.add_argument('--groundtruth-name',
                    default='openai1536_100k_groundtruth_A1_C100.ivecs')
    ap.add_argument('--queries-name', default='queries.json')
    ap.add_argument('--apr-base-name', default='openai1536_100k_apr_base_A1_C100.fvecs')
    ap.add_argument('--apr-query-name', default='openai1536_100k_apr_query_A1_C100.fvecs')
    args = ap.parse_args()

    threads = min(64, (os.cpu_count() or 8))

    if build_binary() and run_cpp(args, threads):
        return
    print('[encode] falling back to Python APR encoder', file=sys.stderr)
    python_fallback(args)


if __name__ == '__main__':
    main()
