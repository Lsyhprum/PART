#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_dataset_utils.py -- data IO and RoPE rotation primitives (self-contained, only
depends on numpy).

Vendored from RaBitQ-range-inline-rot/_utils/_dataset_utils.py, with the faiss
dependency removed (the unused get_nearestneighbors) while keeping the fvecs/
ivecs/attribute-bin read/write and the disjoint-block RoPE numeric-range
rotation primitives.
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
    """Write a self-describing binary with header: 8 bytes [N, A] (little-endian
    int32) + N*A elements of dtype (row-major)."""
    data = np.ascontiguousarray(arr, dtype=dtype)
    n, A = data.shape
    with open(path, "wb") as fp:
        fp.write(struct.pack("II", int(n), int(A)))   # file header
        fp.write(data.tobytes(order="C"))            # data section


def attr_bin_read(path, dtype=np.uint32, n=None, a=None):
    """Read an attribute bin and reshape to (N, A).

    Adapts to two formats (consistent with C++ load_attr_bin):
      - with header: first 8 bytes [N,A] int32 header + data section (when n/a
        are not provided, parse from the header);
      - headerless: pure N*A data section (when n/a are given, auto-detect the
        presence of a header from the file size).
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
    # Without n/a: force parse in header format
    raw = np.fromfile(path, dtype=np.int32)          # header is int32 for easy N/A
    n, A = int(raw[0]), int(raw[1])
    out = raw[2:].view(dtype).reshape(n, A).copy()
    assert out.size == n * A, "attribute file data section size != header N*A"
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
# Attribute CPR encoding (disjoint rotation)
#######################################################

def _rotated(x, angles):
    """Apply angle angles[:,j] to each adjacent dimension pair (2j,2j+1) of x.
    x: (N,d) or (d,) float; angles: (N, d//2) or (1, d//2). Returns same-shape float32."""
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


def _hash_to_phases_cpr(ids, d, seed):
    """Hash each attribute ID into (d//2) phases in [0,2pi). The phase depends
    only on the value itself + seed, independent of other values in the dataset,
    so the data side and the query side generate the same phase for the same value."""
    P = d // 2
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    base_rng = np.random.default_rng(seed)
    salt = base_rng.integers(0, 2 ** 31, size=P, dtype=np.int64)   # one salt per dimension pair
    out = np.empty((len(ids), P))
    for j in range(P):
        h = (ids * 2654435761) ^ (salt[j] * 40503)
        h = (h ^ (h >> 15)) & 0x7FFFFFFF
        out[:, j] = (h / float(0x7FFFFFFF)) * 2.0 * np.pi         # in [0,2pi)
    return out


def cpr_block_split(A, P, scheme="contiguous"):
    """Split P dimension pairs evenly across A attributes with no overlap.
    scheme='contiguous': contiguous split (default; cache/vectorization friendly).
    Returns list[length A], each element a np.ndarray of block dimension-pair indices."""
    if scheme == "contiguous":
        base, rem = divmod(P, A)
        blocks, start = [], 0
        for a in range(A):
            sz = base + (1 if a < rem else 0)
            blocks.append(np.arange(start, start + sz))
            start += sz
        return blocks
    raise ValueError(f"unknown block scheme: {scheme}")


def encode_cpr(base, attrs_cat, blocks, seeds):
    """disjoint-block categorical encoding. Returns (N, d) float32."""
    P = base.shape[1] // 2
    E = base.copy()
    for a in range(attrs_cat.shape[1]):
        full = _hash_to_phases_cpr(attrs_cat[:, a], base.shape[1], seeds[a])   # (N, P)
        ang = np.zeros((full.shape[0], P), dtype=np.float64)
        ang[:, blocks[a]] = full[:, blocks[a]]
        E = _rotated(E, ang)
    return E.astype(np.float32)


# ----------------------------- (A) RoPE frequency encoding (numeric range) -----------------------------

def rope_angles(a_cont, d, base_freq=1.0):
    """a_cont: (N,) continuous value in [0,1); returns (N, d//2) per-pair rotation angle."""
    P = d // 2
    j = np.arange(P, dtype=np.float64)
    freq = np.pi * base_freq * (j + 1.0) / P          # (P,)
    return a_cont[:, None] * freq[None, :]              # (N, P)


def rope_encode_cpr(base, attrs_num, L, blocks, base_freq=1.0):
    """disjoint-block numeric RoPE encoding. attrs_num: (N, A) int in [0,L)."""
    P = base.shape[1] // 2
    E = base.copy()
    for a in range(attrs_num.shape[1]):
        a_cont = attrs_num[:, a].astype(np.float64) / L
        full = rope_angles(a_cont, base.shape[1], base_freq)               # (N, P)
        ang = np.zeros_like(full)
        ang[:, blocks[a]] = full[:, blocks[a]]
        E = _rotated(E, ang)
    return E.astype(np.float32)


def rope_query_range_cpr(q, lo, hi, L, block, base_freq=1.0):
    """disjoint numeric-range query. Apply only the same-phase RoPE rotation to
    the block dimension pairs. Returns (d,) float32."""
    P = q.shape[0] // 2
    c = (lo + hi) / (2.0 * L)
    full = rope_angles(np.array([c], dtype=np.float64), q.shape[0], base_freq)[0]   # (P,) 1-D
    ang = np.zeros((1, P), dtype=np.float64)
    ang[:, block] = full[block]
    return _rotated(q, ang)


#######################################################
# Attribute APR encoding (ported from diskann-fdm)
#######################################################

# F0 fix: splitmix64 hash (GF(2) nonlinear)
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


def _hash_to_phases_apr(ids, d, seed):
    """Hash each attribute ID into (d//2) phases in [0,2pi). F0-fixed version
    (splitmix64 + 63-bit uint64 salt).

    Independent of the old _hash_to_phases_cpr (linear hash) used by cpr, to avoid
    changing disjoint encoding behavior.
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


def encode_apr(base, attrs_cat, seeds):
    """APR layout: each dimension pair simultaneously carries the phase code
    of all A attributes. Returns (N, d) float32.

    total rotation angle (per pair) = sum_a psi_a(v_a)[j] (data side), query side
    = sum_a psi_a(t_a)[j]; relative angle Delta[j] = sum_a (psi_a(t_a)-psi_a(v_a))[j].
    Full match Delta==0 preserves distance; any mismatch is averaged over P pairs
    and pressed to ~0.
    """
    P = base.shape[1] // 2
    total = np.zeros((attrs_cat.shape[0], P), dtype=np.float64)
    for a in range(attrs_cat.shape[1]):
        total += _hash_to_phases_apr(attrs_cat[:, a], base.shape[1], seeds[a])
    return _rotated(base, total).astype(np.float32)


def query_apr_single(q, targets, d, seeds):
    P = d // 2
    q = np.asarray(q, dtype=np.float64)
    total = np.zeros((1, P), dtype=np.float64)
    for a in range(len(targets)):
        total += _hash_to_phases_apr(np.array([int(targets[a])], dtype=np.int64), d, seeds[a])
    return _rotated(q, total).astype(np.float32)
