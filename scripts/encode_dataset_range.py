#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encode_dataset_range.py — 范围属性旋转编码生成器（旋转召回增强）。

对统一数据集施加「数值范围 RoPE 旋转编码」：
  - base  ：按数值属性 attrs_num 做 RoPE 旋转（rope_encode_disjoint）
            -> 数据点相位 ∝ value/L，值相近者相位相近
  - query ：对每条范围谓词 [lo,hi]，按 D3 等分 M 个子窗口，对每个子窗口中心做
            同相旋转（rope_query_range_disjoint）-> 输出 Nq*M 行旋转查询

与 C++ 检索端对接（关键）：
  - 本生成器只产出「旋转后的 base/query fvecs」；Symqg 图建在旋转 base 上，
    search 用旋转 query 行（Nq*M 行，顺序 i*M+s）做 M 次图搜索并集候选，
    再用原始空间 L2² 重排 + attr_match_range 精确闸门（rerank 阶段）取 top-K。
  - lo == -1 表示该维不约束 -> 该查询全部子中心不做旋转（恒等），仍占 M 行占位。

设计约束：
  - 复用 _dataset_utils 的 RoPE 原语（单一数学来源）。
  - v1 仅支持单属性 A=1；输入向量须已单位范数（旋转保模，编码后仍须单位范数）。
"""
import os
import json
import argparse

import numpy as np

from _dataset_utils import (fvecs_read, fvecs_write, attr_bin_read,
                            rope_encode_disjoint, rope_query_range_disjoint,
                            block_split)


def split_window(lo, hi, M):
    """D3 等分：把整数窗口 [lo,hi] 切成 M 个子窗口，返回 [(slo, shi), ...]。
    相邻子窗口端点相接，M>=1 时并集仍为 [lo,hi]。"""
    out = []
    for s in range(M):
        slo = lo + (hi - lo) * s // M
        shi = lo + (hi - lo) * (s + 1) // M
        out.append((slo, shi))
    return out


def main():
    ap = argparse.ArgumentParser(
        description="范围属性旋转编码生成器（base RoPE + query M 子中心）")
    ap.add_argument('--ds', required=True, help='统一数据集目录（含 base/query/attr）')
    ap.add_argument('--out', required=True, help='旋转产物输出目录（与 --ds 同目录即可）')
    ap.add_argument('--d', type=int, default=1536, help='向量维度')
    ap.add_argument('--A', type=int, default=1, help='属性数（v1 仅支持 A=1）')
    ap.add_argument('--L', type=int, default=50, help='数值属性上界（相位分母，=基数 C）')
    ap.add_argument('--base-freq', type=float, default=1.0, help='RoPE 频率尺度')
    ap.add_argument('--M', type=int, default=1, help='每条查询子窗口中心数（须与 C++ --range_m 一致）')
    ap.add_argument('--base-name', required=True, help='输入 base 向量文件名')
    ap.add_argument('--query-name', required=True, help='输入 query 向量文件名')
    ap.add_argument('--base-attr-name', required=True, help='输入 base 数值属性文件名')
    ap.add_argument('--query-range-name', required=True, help='输入 query 范围谓词文件名 (Nq,2A)')
    ap.add_argument('--rotated-base-name', required=True, help='输出旋转 base 文件名')
    ap.add_argument('--rotated-query-name', required=True, help='输出旋转 query 文件名（Nq*M 行）')
    args = ap.parse_args()

    assert args.A == 1, "v1 范围旋转增强仅支持单属性 A=1"
    assert args.M >= 1, "--M 须 >= 1"
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
    assert attrs_num.shape == (N, args.A), f'base 属性形状 {attrs_num.shape} 应为 (N={N}, A={args.A})'
    assert qrange.shape == (Nq, 2 * args.A), f'query_range 形状 {qrange.shape} 应为 (Nq={Nq}, 2A={2*args.A})'
    print('      base', X.shape, 'query', Q.shape,
          'base_attr', attrs_num.shape, 'query_range', qrange.shape)

    blocks = block_split(args.A, P)

    print('[2] range-rotating base (rope_encode_disjoint, A=%d, L=%d) ...' % (args.A, L))
    E_base = rope_encode_disjoint(X, attrs_num, L, blocks, base_freq=args.base_freq)

    print('[3] rotating queries: %d logical x M=%d subcenters = %d rows ...' % (Nq, M, Nq * M))
    E_q = np.empty((Nq * M, D), dtype=np.float32)
    n_unconstrained = 0
    for i in range(Nq):
        lo = int(qrange[i, 0])
        hi = int(qrange[i, 1])
        if lo == -1:                       # 该维不约束 -> 恒等，仍占 M 行
            n_unconstrained += 1
            for s in range(M):
                E_q[i * M + s] = Q[i]
        else:
            for s, (slo, shi) in enumerate(split_window(lo, hi, M)):
                E_q[i * M + s] = rope_query_range_disjoint(
                    Q[i], slo, shi, L, blocks[0], base_freq=args.base_freq)
    print('      unconstrained queries (lo==-1):', n_unconstrained, '/', Nq)

    # 单位范数健全性（正交旋转保模）
    nb = np.linalg.norm(E_base, axis=1)
    nq = np.linalg.norm(E_q, axis=1)
    print('      base norm mean=%.4f std=%.2e | query norm mean=%.4f std=%.2e'
          % (nb.mean(), nb.std(), nq.mean(), nq.std()))
    assert abs(nb.mean() - 1) < 1e-3 and abs(nq.mean() - 1) < 1e-3, \
        '编码须保持单位范数（请确认输入 base/query 已归一化）'

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
