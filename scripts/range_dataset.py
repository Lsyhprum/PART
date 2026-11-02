#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
range_dataset.py — 范围属性谓词 bin + 范围过滤真值生成器。

复用已存在的 base_attr（由 base_attr_A{ATTR_DIM}_C{ATTR_CARD}.bin 提供）作为
数据库属性真值，生成：
  - query_attr_A*_C*_range.bin : (Nq, 2A) 交错 [lo_a, hi_a] 的 int32 范围谓词
    （头 A 记为 2A；lo=-1 表示该维不约束）
  - {ORIG_DATASET}_groundtruth_range_A*_C*.ivecs : 范围过滤真值
    （base 中属性落入各查询区间 [lo,hi] 内的匹配集，按 L2 距离排序取 top-K）

与参考库 1.range_dataset.py 的差异：base 属性【不重新随机生成】，直接复用现有
base_attr，确保与图内核精确闸门、旋转编码使用同一份真值（关键，否则 GT 与
谓词/编码口径不一致）。
"""
import os
import argparse
import numpy as np

from _dataset_utils import fvecs_read, fvecs_write, ivecs_write, attr_bin_read, attr_bin_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-base', required=True, help='base fvecs 路径（重排/GT 距离用）')
    ap.add_argument('--src-query', required=True, help='query fvecs 路径')
    ap.add_argument('--out-dir', required=True, help='输出目录')
    ap.add_argument('--base-attr-name', required=True, help='已有 base 属性 bin（带头 8 字节头）')
    ap.add_argument('--query-attr-name', required=True, help='输出 range 谓词 bin 名 (Nq x 2A)')
    ap.add_argument('--groundtruth-name', required=True, help='输出范围 GT 名')
    ap.add_argument('--A', type=int, default=1, help='属性数')
    ap.add_argument('--card', type=int, default=50, help='每属性基数 C -> 选择性 1/card')
    ap.add_argument('--range-w', type=int, default=5,
                    help='范围覆盖连续类别数 w（clamp 到 [1, card]；w=1 退化为等值）')
    ap.add_argument('--K', type=int, default=100, help='过滤真值 / Recall@K')
    ap.add_argument('--n', type=int, default=100000, help='取 base 前 N 行')
    ap.add_argument('--nq', type=int, default=1000, help='查询数')
    ap.add_argument('--d', type=int, default=1536, help='向量维度')
    ap.add_argument('--metric', choices=['ip', 'l2'], default='l2', help='距离度量')
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
    # base_attr 可能是无头纯数据格式：用已知 N/A 自适应判定有无 8 字节头
    attrs = attr_bin_read(os.path.join(OUT_DIR, args.base_attr_name), np.uint32, n=N, a=args.A)  # (N, A)
    assert attrs.shape[0] == N and attrs.shape[1] == args.A, \
        f'base_attr 形状 {attrs.shape} 与 N={N}, A={args.A} 不一致'
    print('      base attrs', attrs.shape, 'card', args.card)

    # ---------------- query 范围谓词 [lo, hi] ----------------
    rng_q = np.random.default_rng(2000)
    w = min(max(args.range_w, 1), args.card)
    lo = rng_q.integers(0, args.card - w + 1, size=(Nq, args.A), dtype=np.int64)
    hi = lo + (w - 1)
    qrange = np.ascontiguousarray(
        np.stack([lo, hi], axis=2).reshape(Nq, 2 * args.A), dtype=np.int64)
    attr_bin_write(os.path.join(OUT_DIR, args.query_attr_name), qrange, np.int32)
    print('      query ranges', qrange.shape, 'card', args.card, 'range-w', w)

    # ---------------- 范围过滤真值 ----------------
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
            score = np.einsum('ij,ij->i', diff, diff)   # L2²
            order = np.argsort(score)[:min(args.K, m_idx.size)]
        G[i, :len(order)] = m_idx[order]
    ivecs_write(os.path.join(OUT_DIR, args.groundtruth_name), G)
    print('      avg selectivity = %.4f (min %.4f, max %.4f)'
          % (sel.mean(), sel.min(), sel.max()))
    print('[ok] range dataset ->', OUT_DIR)


if __name__ == '__main__':
    main()
