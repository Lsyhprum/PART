#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_dataset_utils.py — 数据 IO 与 RoPE 旋转原语（自包含，仅依赖 numpy）。

从参考库 RaBitQ-range-inline-rot/_utils/_dataset_utils.py vendor 而来，
去掉 faiss 依赖（get_nearestneighbors 未使用），保留 fvecs/ivecs/
属性 bin 的读写与 disjoint-block RoPE 数值范围旋转原语。
"""
import struct
import numpy as np


#######################################################
# Data file read/write
#######################################################

def bin_read(fname, vec_dtype=np.float32):
    raw = np.fromfile(fname, dtype=np.int32)
    n, d = raw[0], raw[1]
    return raw[2:].view(vec_dtype).reshape(n, d).copy()


def bin_write(filename, data, vec_dtype=np.float32):
    assert data.ndim == 2, "data must be 2D array [num, dim]"
    n, d = data.shape
    with open(filename, "wb") as fp:
        fp.write(struct.pack("II", n, d))
        fp.write(data.astype(vec_dtype).tobytes())


def fbin_read(fname):
    return bin_read(fname, vec_dtype=np.float32)


def fbin_write(filename, data):
    bin_write(filename, data, vec_dtype=np.float32)


def ibin_read(fname):
    return bin_read(fname, vec_dtype=np.int32)


def ibin_write(filename, data):
    bin_write(filename, data, vec_dtype=np.int32)


def attr_bin_write(path, arr, dtype=np.uint32):
    """带头自描述二进制写出：8 字节 [N, A]（int32 小端）+ N*A 个 dtype（行主序）。"""
    data = np.ascontiguousarray(arr, dtype=dtype)
    n, A = data.shape
    with open(path, "wb") as fp:
        fp.write(struct.pack("II", int(n), int(A)))   # 文件头
        fp.write(data.tobytes(order="C"))            # 数据区


def attr_bin_read(path, dtype=np.uint32, n=None, a=None):
    """属性 bin 读入并 reshape 为 (N, A)。

    自适应两种格式（与 C++ load_attr_bin 一致）：
      - 带头：前 8 字节 [N,A] int32 头 + 数据区（当 n/a 未提供时按头解析）；
      - 无头：纯 N*A 数据区（当给定 n/a 时按文件大小自动判定有无头）。
    """
    import os
    if n is not None and a is not None:
        itemsize = np.dtype(dtype).itemsize
        expected_noheader = int(n) * int(a) * itemsize
        expected_header = 8 + expected_noheader
        actual = os.path.getsize(path)
        if actual == expected_header:
            raw = np.fromfile(path, dtype=np.int32)
            assert int(raw[0]) == int(n) and int(raw[1]) == int(a), \
                f"attr header mismatch in {path}: {raw[0]},{raw[1]} != {n},{a}"
            arr = np.fromfile(path, dtype=dtype)[2:]
        elif actual == expected_noheader:
            arr = np.fromfile(path, dtype=dtype)
        else:
            raise ValueError(
                f"attr size mismatch {path}: {actual} not {expected_noheader} or {expected_header}")
        return arr.reshape(int(n), int(a)).copy()
    # 无 n/a：强制按带头格式解析
    raw = np.fromfile(path, dtype=np.int32)          # 头为 int32，便于取 N/A
    n, A = int(raw[0]), int(raw[1])
    out = raw[2:].view(dtype).reshape(n, A).copy()
    assert out.size == n * A, "属性文件数据区大小与头部 N*A 不一致"
    return out


def fvecs_write(filename, data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    row_bytes = 4 + d * 4
    buf = bytearray(n * row_bytes)
    for i in range(n):
        base = i * row_bytes
        buf[base:base + 4] = struct.pack('I', d)
        buf[base + 4:base + row_bytes] = data[i].tobytes()
    with open(filename, 'wb') as fp:
        fp.write(buf)


def fvecs_read(fname):
    x = np.memmap(fname, dtype=np.int32, mode='r')
    d = x[0]
    arr = x.view(np.float32).reshape(-1, d + 1)
    return arr[:, 1:]


def ivecs_write(filename, data):
    print(f"Writing File - {filename}")
    data = np.ascontiguousarray(data, dtype=np.int32)
    n, d = data.shape
    row_bytes = 4 + d * 4
    buf = bytearray(n * row_bytes)
    for i in range(n):
        base = i * row_bytes
        buf[base:base + 4] = struct.pack('I', d)
        buf[base + 4:base + row_bytes] = data[i].tobytes()
    with open(filename, "wb") as f:
        f.write(buf)
    print(f"\t{filename} wrote")


def ivecs_read(fname):
    x = np.memmap(fname, dtype=np.int32, mode='r')
    d = x[0]
    arr = x.reshape(-1, d + 1)
    return arr[:, 1:]


#######################################################
# Attribute rotation encoding (disjoint rotation)
#######################################################

def _rotated(x, angles):
    """对 x 的每对相邻维度 (2j,2j+1) 施加角度 angles[:,j]。
    x: (N,d) 或 (d,) float; angles: (N, d//2) 或 (1, d//2)。返回同形状 float32。"""
    x = np.array(x, dtype=np.float64)
    single = x.ndim == 1
    if single:
        x = x[None, :]
    P = x.shape[1] // 2
    c = np.cos(angles)
    s = np.sin(angles)
    x0 = x[:, 0::2].copy()
    x1 = x[:, 1::2].copy()
    nx0 = x0 * c - x1 * s
    nx1 = x0 * s + x1 * c
    out = x.copy()
    out[:, 0::2] = nx0
    out[:, 1::2] = nx1
    out = out.astype(np.float32)
    return out[0] if single else out


def _hash_to_phases(ids, d, seed):
    """把每个属性ID哈希为 (d//2) 个相位∈[0,2π)。相位只依赖于「值本身+seed」,
    与数据集中其它值无关 -> 数据端与查询端对同一值必生成相同相位。"""
    P = d // 2
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    base_rng = np.random.default_rng(seed)
    salt = base_rng.integers(0, 2 ** 31, size=P, dtype=np.int64)   # 每对维度一个盐
    out = np.empty((len(ids), P))
    for j in range(P):
        h = (ids * 2654435761) ^ (salt[j] * 40503)
        h = (h ^ (h >> 15)) & 0x7FFFFFFF
        out[:, j] = (h / float(0x7FFFFFFF)) * 2.0 * np.pi         # ∈[0,2π)
    return out


def block_split(A, P, scheme="contiguous"):
    """把 P 个维度对均分给 A 个属性，互不重叠。
    scheme='contiguous'：连续切分（默认，缓存/向量化友好）。
    返回 list[length A]，每项为 block 维度对索引的 np.ndarray。"""
    if scheme == "contiguous":
        base, rem = divmod(P, A)
        blocks, start = [], 0
        for a in range(A):
            sz = base + (1 if a < rem else 0)
            blocks.append(np.arange(start, start + sz))
            start += sz
        return blocks
    raise ValueError(f"unknown block scheme: {scheme}")


def encode_disjoint(base, attrs_cat, blocks, seeds):
    """disjoint-block 类别编码。返回 (N, d) float32。"""
    P = base.shape[1] // 2
    E = base.copy()
    for a in range(attrs_cat.shape[1]):
        full = _hash_to_phases(attrs_cat[:, a], base.shape[1], seeds[a])   # (N, P)
        ang = np.zeros((full.shape[0], P), dtype=np.float64)
        ang[:, blocks[a]] = full[:, blocks[a]]
        E = _rotated(E, ang)
    return E.astype(np.float32)


# ----------------------------- (A) RoPE 频率编码 (数值范围) -----------------------------

def rope_angles(a_cont, d, base_freq=1.0):
    """a_cont: (N,) 连续值∈[0,1); 返回 (N, d//2) 每对旋转角。"""
    P = d // 2
    j = np.arange(P, dtype=np.float64)
    freq = np.pi * base_freq * (j + 1.0) / P          # (P,)
    return a_cont[:, None] * freq[None, :]              # (N, P)


def rope_encode_disjoint(base, attrs_num, L, blocks, base_freq=1.0):
    """disjoint-block 数值 RoPE 编码。attrs_num: (N, A) int∈[0,L)。"""
    P = base.shape[1] // 2
    E = base.copy()
    for a in range(attrs_num.shape[1]):
        a_cont = attrs_num[:, a].astype(np.float64) / L
        full = rope_angles(a_cont, base.shape[1], base_freq)               # (N, P)
        ang = np.zeros_like(full)
        ang[:, blocks[a]] = full[:, blocks[a]]
        E = _rotated(E, ang)
    return E.astype(np.float32)


def rope_query_range_disjoint(q, lo, hi, L, block, base_freq=1.0):
    """disjoint 数值范围查询。仅对 block 维度对施加 RoPE 同相旋转。返回 (d,) float32。"""
    P = q.shape[0] // 2
    c = (lo + hi) / (2.0 * L)
    full = rope_angles(np.array([c], dtype=np.float64), q.shape[0], base_freq)[0]   # (P,) 1-D
    ang = np.zeros((1, P), dtype=np.float64)
    ang[:, block] = full[block]
    return _rotated(q, ang)


#######################################################
# Attribute FDM/CDMA encoding (ported from diskann-fdm)
#######################################################

# F0 修复：splitmix64 哈希 (GF(2) 非线性)
_MIX_C1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX_C2 = np.uint64(0x94D049BB133111EB)
_GOLDEN_RATIO64 = np.uint64(0x9E3779B97F4A7C15)
_U30, _U27, _U31 = np.uint64(30), np.uint64(27), np.uint64(31)
_PHASE_MASK = np.uint64(0x7FFFFFFF)


def _mix64(x):
    x = np.asarray(x, dtype=np.uint64)
    x = (x ^ (x >> _U30)) * _MIX_C1
    x = (x ^ (x >> _U27)) * _MIX_C2
    return x ^ (x >> _U31)


def _hash_to_phases_f0(ids, d, seed):
    """把每个属性ID哈希为 (d//2) 个相位 in [0,2π)。F0 修复版（splitmix64 + 63-bit uint64 盐）。

    独立于 rot 用的旧版 _hash_to_phases（线性哈希），避免改动 disjoint 编码行为。
    """
    P = d // 2
    ids = np.asarray(ids, dtype=np.uint64).reshape(-1)
    rng = np.random.default_rng(seed)
    salt = rng.integers(0, 2 ** 63, size=P, dtype=np.int64).astype(np.uint64)
    base = _mix64(ids * _GOLDEN_RATIO64)
    out = np.empty((len(ids), P), dtype=np.float64)
    for j in range(P):
        h = _mix64(base ^ salt[j]) & _PHASE_MASK
        out[:, j] = (h.astype(np.float64) / float(0x7FFFFFFF)) * 2.0 * np.pi
    return out


def encode_fdm(base, attrs_cat, seeds):
    """FDM/CDMA 布局：每个维度对同时承载全部 A 个属性的相位码。返回 (N, d) float32。

    total 旋转角(每对) = Σ_a ψ_a(v_a)[j]（数据端），查询端 = Σ_a ψ_a(t_a)[j]，
    相对角 Δ[j] = Σ_a (ψ_a(t_a)-ψ_a(v_a))[j]。全匹配 Δ≡0 保距；任一失配被 P 对平均压到≈0。
    """
    P = base.shape[1] // 2
    total = np.zeros((attrs_cat.shape[0], P), dtype=np.float64)
    for a in range(attrs_cat.shape[1]):
        total += _hash_to_phases_f0(attrs_cat[:, a], base.shape[1], seeds[a])
    return _rotated(base, total).astype(np.float32)


def query_fdm_single(q, targets, d, seeds):
    P = d // 2
    q = np.asarray(q, dtype=np.float64)
    total = np.zeros((1, P), dtype=np.float64)
    for a in range(len(targets)):
        total += _hash_to_phases_f0(np.array([int(targets[a])], dtype=np.int64), d, seeds[a])
    return _rotated(q, total).astype(np.float32)
