#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对统一数据集施加 disjoint 单属性旋转编码（symqg 本地副本，对齐 IVF/RaBitQ-inline-rot）。

与 IVF 的 3.encode_dataset.py 唯一差异：放宽「编码须保持单位范数」断言。
原因：symqg 的 sift / msmarco 输入未归一化，而 encode_disjoint 仅施加正交 2D 旋转，
保距（旋转空间 L2² == 原始空间 L2²），GT 仍有效；无需归一化（归一化会改变范数→GT 失效）。
"""

import os
import shutil
import argparse
import numpy as np

from _dataset_utils import (fvecs_read, fvecs_write, attr_bin_read,
                            encode_disjoint, block_split)

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=os.path.join(PKG, 'ds'),
                    help='统一数据集目录（默认 <包>/ds）')
    ap.add_argument('--out', default=os.path.join(PKG, 'ds_ro'),
                    help='编码输出目录（默认 <包>/ds_ro）')
    ap.add_argument('--d', type=int, default=1536)
    ap.add_argument('--A', type=int, default=1, help='属性数（须与 1.build_dataset.py 一致）')
    ap.add_argument('--enc-seed', type=int, default=1234)
    ap.add_argument('--base-name', default='base_100k.fvecs',
                    help='输入 base 向量文件名')
    ap.add_argument('--query-name', default='query_1k.fvecs',
                    help='输入 query 向量文件名')
    ap.add_argument('--base-attr-name', default='base_attr_A1_C100.bin',
                    help='输入 base 属性文件名（带头自描述 bin）')
    ap.add_argument('--query-attr-name', default='query_attr_A1_C100.bin',
                    help='输入 query 属性文件名（带头自描述 bin）')
    ap.add_argument('--groundtruth-name', default='openai1536_100k_groundtruth_A1_C100.ivecs',
                    help='输入 groundtruth 文件名（含 A/C 后缀，与 1.build_dataset.py 保持一致）')
    ap.add_argument('--queries-name', default='queries.json',
                    help='输入 queries 元数据文件名')
    ap.add_argument('--rotated-base-name', default='openai1536_100k_rotated_base_A1_C100.fvecs',
                    help='输出旋转编码 base 向量文件名（含 A/C 后缀以区分属性数/基数）')
    ap.add_argument('--rotated-query-name', default='openai1536_100k_rotated_query_A1_C100.fvecs',
                    help='输出旋转编码 query 向量文件名（含 A/C 后缀以区分属性数/基数）')
    args = ap.parse_args()

    COPY_FILES = [args.groundtruth_name]

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    D = args.d
    P = D // 2
    A = args.A
    # 把 P 个维度对均分给 A 个属性，互不重叠 -> 多属性独立编码（块间零串扰）
    blocks = block_split(A, P)
    seeds = [args.enc_seed + a for a in range(A)]

    print('[1] loading unified dataset from', args.ds, '...')
    X = fvecs_read(os.path.join(args.ds, args.base_name)).astype(np.float32)
    Q = fvecs_read(os.path.join(args.ds, args.query_name)).astype(np.float32)
    attrs = attr_bin_read(os.path.join(args.ds, args.base_attr_name), np.uint32)
    attrs_q = attr_bin_read(os.path.join(args.ds, args.query_attr_name), np.int32)
    targets = attrs_q  # (nq, A) 完整属性矩阵
    N, _ = X.shape
    assert attrs.shape[1] == A and attrs.shape[0] == N, \
        f'base 属性维数 {attrs.shape[1]} 与 A={A} 不一致'
    assert attrs_q.shape[1] == A, f'query 属性维数 {attrs_q.shape[1]} 与 A={A} 不一致'
    assert attrs_q.shape[0] == Q.shape[0], 'query 属性数与查询数不一致'
    print('      base', X.shape, 'query', Q.shape)

    print('[2] disjoint multi-attr rotation (A=%d) ...' % A)
    E_base = encode_disjoint(X, attrs, blocks, seeds)
    E_q = encode_disjoint(Q, targets, blocks, seeds)

    nb = np.linalg.norm(E_base, axis=1)
    nq = np.linalg.norm(E_q, axis=1)
    print('      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e'
          % (nb.mean(), nb.std(), nq.mean(), nq.std()))
    # 放宽：输入未归一化时（sift/msmarco）旋转后仍非单位范数，但旋转保距，GT 仍有效。
    if abs(nb.mean() - 1) >= 1e-3 or abs(nq.mean() - 1) >= 1e-3:
        print('[warn] 编码输出非单位范数（输入未归一化）；正交旋转保距，原始空间 GT 仍有效。')

    print('[3] writing encoded dataset ...')
    fvecs_write(os.path.join(OUT, args.rotated_base_name), E_base)
    fvecs_write(os.path.join(OUT, args.rotated_query_name), E_q)
    for f in COPY_FILES:
        shutil.copy(os.path.join(args.ds, f), os.path.join(OUT, f))
    print('[ok] encoded dataset ->', OUT, 'base', E_base.shape, 'query', E_q.shape)


if __name__ == '__main__':
    main()
