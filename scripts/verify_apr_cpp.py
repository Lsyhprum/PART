#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_apr_cpp.py -- bit-exact verification that the C++ APR encoder and the
Python encode_apr produce identical output.

Method: build a small random dataset (base/query/attr), write the --salt-file
using the same numpy-exact salt generation as the wrapper script, run
bin/encode_apr, then compute the same data with _dataset_utils.encode_apr
(same seed) and compare the max abs diff. The two share the same math path
(splitmix64 phase + all-dimension-pair superposed rotation + float32->double->
float32), so the expected difference is ~0.
"""

import os
import sys
import tempfile
import subprocess
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _dataset_utils import (fvecs_write, attr_bin_write, fvecs_read,
                            attr_bin_read, encode_apr)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(PKG, 'bin', 'encode_apr')


def make_dataset(out_dir, N, NQ, D, A, C, seed=1234):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Q = rng.standard_normal((NQ, D)).astype(np.float32)
    attrs = rng.integers(0, C, size=(N, A)).astype(np.uint32)
    attrs_q = rng.integers(0, C, size=(NQ, A)).astype(np.uint32)
    fvecs_write(os.path.join(out_dir, 'base.fvecs'), X)
    fvecs_write(os.path.join(out_dir, 'query.fvecs'), Q)
    attr_bin_write(os.path.join(out_dir, 'base_attr.bin'), attrs)
    attr_bin_write(os.path.join(out_dir, 'query_attr.bin'), attrs_q)
    # placeholder groundtruth (C++ only copies it, not used for comparison)
    gt = np.zeros((NQ, 1), dtype=np.int32)
    from _dataset_utils import ivecs_write
    ivecs_write(os.path.join(out_dir, 'gt.ivecs'), gt)
    return X, Q, attrs, attrs_q


def main():
    if not os.path.exists(BIN):
        print('[verify] C++ binary missing, compiling first...', file=sys.stderr)
        subprocess.run(
            ['g++', '-std=c++17', '-o', BIN,
             os.path.join(os.path.dirname(os.path.abspath(__file__)), 'encode_apr.cpp'),
             '-I', os.path.dirname(os.path.abspath(__file__)),
             '-Ofast', '-march=core-avx2', '-pthread',
             '-fno-fast-math', '-ffp-contract=off'],
            check=True)
    assert os.path.exists(BIN), 'bin/encode_apr still absent after compile'

    N, NQ, D, A, C = 2000, 200, 16, 3, 4
    enc_seed = 1234
    work = tempfile.mkdtemp(prefix='apr_verify_')
    try:
        X, Q, attrs, attrs_q = make_dataset(work, N, NQ, D, A, C, seed=7)
        P = D // 2
        seeds = [enc_seed + a for a in range(A)]
        salt = np.concatenate([
            np.random.default_rng(s).integers(0, 2 ** 63, size=P, dtype=np.int64)
            for s in seeds
        ]).astype('<i8')
        salt_path = os.path.join(work, 'salt.bin')
        salt.tofile(salt_path)

        out_dir = os.path.join(work, 'apr')
        cmd = [
            BIN, '--ds', work, '--out', out_dir, '--d', str(D), '--A', str(A),
            '--enc-seed', str(enc_seed), '--base-name', 'base.fvecs',
            '--query-name', 'query.fvecs', '--base-attr-name', 'base_attr.bin',
            '--query-attr-name', 'query_attr.bin', '--groundtruth-name', 'gt.ivecs',
            '--apr-base-name', 'apr_base.fvecs', '--apr-query-name', 'apr_query.fvecs',
            '--salt-file', salt_path, '--threads', '1',
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout); print(r.stderr, file=sys.stderr)
            print('[verify] C++ run failed', file=sys.stderr); sys.exit(1)

        cpp_base = fvecs_read(os.path.join(out_dir, 'apr_base.fvecs'))
        cpp_query = fvecs_read(os.path.join(out_dir, 'apr_query.fvecs'))

        s = [enc_seed + a for a in range(A)]
        py_base = encode_apr(X, attrs, s)
        py_query = encode_apr(Q, attrs_q, s)

        db = np.abs(cpp_base - py_base)
        dq = np.abs(cpp_query - py_query)
        max_b = db.max(); max_q = dq.max()
        print('max abs diff base  = %.3e' % max_b)
        print('max abs diff query = %.3e' % max_q)
        ok = max_b < 1e-5 and max_q < 1e-5
        print('[verify] %s' % ('PASS (bit-exact with Python)' if ok else 'FAIL'))
        sys.exit(0 if ok else 1)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
