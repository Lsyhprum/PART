#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对统一数据集施加 FDM/CDMA 相位叠加编码（symqg 本地副本）。

本脚本为薄包装：优先调用 C++ 编码器 bin/encode_fdm（速度快几个数量级，arxiv
D=4096/1M 从分钟级降到秒～分钟级）；C++ 二进制缺失时自动编译，编译或运行失败
时回退到纯 Python 实现（_dataset_utils.encode_fdm），保证现有 sift/msmarco/
openai1536 流程不退化。CLI 与原 Python 版完全一致，run.sh 调用方式不变。

salt 由 numpy 精确生成（seeds[a]=enc_seed+a，integers(0,2**63)），经 --salt-file
传给 C++，保证与现有 FDM 结果连续、重跑字节一致。
"""

import os
import sys
import argparse
import subprocess
import tempfile

import numpy as np

from _dataset_utils import (fvecs_read, fvecs_write, attr_bin_read, encode_fdm)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(PKG, 'bin', 'encode_fdm')
CPP_SRC = os.path.join(SCRIPT_DIR, 'encode_fdm.cpp')


def build_binary():
    """C++ 二进制缺失时尝试编译；成功返回 True，否则 False。"""
    if os.path.exists(BIN):
        return True
    print('[encode] C++ 二进制缺失，尝试编译 %s ...' % BIN, file=sys.stderr)
    try:
        subprocess.run(
            ['g++', '-std=c++17', '-o', BIN, CPP_SRC, '-I', SCRIPT_DIR,
             '-Ofast', '-march=core-avx2', '-pthread',
             '-fno-fast-math', '-ffp-contract=off'],
            check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print('[encode] 编译失败，回退 Python 实现: %s' % e, file=sys.stderr)
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
            '--fdm-base-name', args.fdm_base_name,
            '--fdm-query-name', args.fdm_query_name,
            '--salt-file', tmp, '--threads', str(threads),
        ]
        rc = subprocess.run(cmd).returncode
        return rc == 0
    except Exception as e:  # noqa: BLE001
        print('[encode] C++ 运行异常，回退 Python 实现: %s' % e, file=sys.stderr)
        return False
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)


def python_fallback(args):
    """原 Python 实现（与旧版 encode_attr_fdm.py 行为一致）。"""
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
    targets = attrs_q  # (nq, A) 完整属性矩阵
    N, _ = X.shape
    assert attrs.shape[1] == A and attrs.shape[0] == N, \
        'base 属性维数 %s 与 A=%d 不一致' % (attrs.shape[1], A)
    assert attrs_q.shape[1] == A, 'query 属性维数 %s 与 A=%d 不一致' % (attrs_q.shape[1], A)
    assert attrs_q.shape[0] == Q.shape[0], 'query 属性数与查询数不一致'
    print('      base', X.shape, 'query', Q.shape)

    print('[2] fdm multi-attr rotation (A=%d) ...' % A)
    E_base = encode_fdm(X, attrs, seeds)
    E_q = encode_fdm(Q, targets, seeds)

    nb = np.linalg.norm(E_base, axis=1)
    nq = np.linalg.norm(E_q, axis=1)
    print('      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e'
          % (nb.mean(), nb.std(), nq.mean(), nq.std()))
    if abs(nb.mean() - 1) >= 1e-3 or abs(nq.mean() - 1) >= 1e-3:
        print('[warn] 编码输出非单位范数（输入未归一化）；正交旋转保距，原始空间 GT 仍有效。')

    print('[3] writing encoded dataset ...')
    fvecs_write(os.path.join(OUT, args.fdm_base_name), E_base)
    fvecs_write(os.path.join(OUT, args.fdm_query_name), E_q)
    for f in COPY_FILES:
        import shutil
        shutil.copy(os.path.join(args.ds, f), os.path.join(OUT, f))
    print('[ok] encoded dataset ->', OUT, 'base', E_base.shape, 'query', E_q.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=os.path.join(PKG, 'ds'))
    ap.add_argument('--out', default=os.path.join(PKG, 'ds_fdm'))
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
    ap.add_argument('--fdm-base-name', default='openai1536_100k_fdm_base_A1_C100.fvecs')
    ap.add_argument('--fdm-query-name', default='openai1536_100k_fdm_query_A1_C100.fvecs')
    args = ap.parse_args()

    threads = min(64, (os.cpu_count() or 8))

    if build_binary() and run_cpp(args, threads):
        return
    print('[encode] 回退到 Python FDM 编码器', file=sys.stderr)
    python_fallback(args)


if __name__ == '__main__':
    main()
