#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_dataset_utils.py -- data IO helpers for the attribute-constraint pipeline.

Only the fvecs / ivecs / attribute-bin read/write helpers remain here, for the
range groundtruth generator (range_dataset.py). Attribute rotation encoding
(APR / CPR / range) is performed by the C++ library samples:

    bin/attribute_apr_encode  (APR equality + HAPR range, --layout eq|range)
    bin/attribute_cpr_encode  (CPR disjoint-block)

No Python encoding primitives live in this module anymore.
"""
import struct
import numpy as np


#######################################################
# Data file read/write
#######################################################

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
