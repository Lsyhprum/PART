import os
import struct
import numpy as np


#######################################################
# Data file I/O
#######################################################

# ====================== Attribute binary read/write (self-describing header) ======================
# File layout: 8-byte [N, A] (int32 little-endian) header + N*A payload elements.
# base attribute payload uses uint32, query attribute uses int32 (reserving the -1 sentinel bit).
def attr_bin_write(path, arr, dtype=np.uint32):
    """Write a self-describing binary: 8-byte [N, A] (int32 little-endian) header + N*A elements of dtype (row-major)."""
    data = np.ascontiguousarray(arr, dtype=dtype)
    n, A = data.shape
    with open(path, "wb") as fp:
        fp.write(struct.pack("II", int(n), int(A)))   # header
        fp.write(data.tobytes(order="C"))            # data payload


def attr_bin_read(path, dtype=np.uint32):
    """Read a self-describing binary and reshape to (N, A). N/A are read from the header, no external size needed."""
    raw = np.fromfile(path, dtype=np.int32)          # header is int32, convenient for N/A
    n, A = int(raw[0]), int(raw[1])
    out = raw[2:].view(dtype).reshape(n, A).copy()
    assert out.size == n * A, "attribute file data payload size does not match header N*A"
    return out


# ====================== Filtered ground-truth recomputation (shared by generators) ======================
# These functions are the single source of truth for filtered gt recomputation, used by
# gen_eq_attr_gt.py / gen_range_attr_gt.py. Generated gt is validated by the same recompute
# logic, so it always passes independent verification.

def load_attr(path, dtype, A_default=1):
    """Read an attribute bin, auto-detecting two layouts:
       - self-describing header: 8-byte [N,A](int32) + N*A elements of dtype (written by attr_bin_write)
       - headerless: N*A elements of dtype directly (legacy datasets, no [N,A] header)
    Presence of a header is judged by whether the file size matches '8 + N*A*4';
    when headerless, A is taken from A_default.
    """
    sz = os.path.getsize(path)
    raw = np.fromfile(path, dtype=np.int32)
    n_hdr, A_hdr = int(raw[0]), int(raw[1])
    if sz > 8 and (sz - 8) == n_hdr * A_hdr * 4:
        # has header
        return raw[2:].view(dtype).reshape(n_hdr, A_hdr).copy()
    # headerless: total elements = sz//4, A given externally
    total = sz // 4
    A = A_default
    N = total // A
    assert N * A == total, f'{path}: file size {sz} is not divisible by A={A}'
    return np.fromfile(path, dtype=dtype).reshape(N, A).copy()


def recompute_gt(X, Q, base_attr, query_attr, metric, K):
    """Equality filter: independent brute-force ground-truth recomputation. Returns (G, sel),
    same semantics as the dataset builder. Multi-attribute AND (every base attribute equals the
    corresponding query attribute); within the match set take top-K by metric; invalid slots are -1.
    Shared by gen_eq_attr_gt.py final candidate selection / self_check.
    """
    N, A = base_attr.shape
    QN = Q.shape[0]
    G = np.full((QN, K), -1, dtype=np.int32)
    sel = np.zeros(QN, dtype=np.float64)
    for i in range(QN):
        mask = np.all(base_attr == query_attr[i], axis=1)   # multi-attribute AND
        m_idx = np.where(mask)[0]
        sel[i] = m_idx.size / N
        if m_idx.size == 0:
            continue
        if metric == 'ip':
            score = Q[i] @ X[m_idx].T                       # inner product, larger = more similar
            order = np.argsort(-score)[:min(K, m_idx.size)]
        else:  # l2
            diff = X[m_idx] - Q[i]
            score = np.einsum('ij,ij->i', diff, diff)       # squared Euclidean, smaller = more similar
            order = np.argsort(score)[:min(K, m_idx.size)]
        G[i, :len(order)] = m_idx[order]
    return G, sel


def topk_legal_set(X, q, m_idx, metric, expect_n, tol=1e-6):
    """Tie-tolerant top-K legal set: within the match set m_idx take the top expect_n by metric,
    and return the set of absolute base indices whose score is no worse than the expect_n-th
    boundary (including ties). Returns an empty set when expect_n <= 0 or m_idx is empty.
    Shared by the self-checks of gen_eq_attr_gt.py / gen_range_attr_gt.py.
    """
    if expect_n <= 0 or m_idx.size == 0:
        return set()
    if metric == 'ip':
        score = q @ X[m_idx].T                         # inner product, larger = more similar
        boundary = np.sort(score)[::-1][expect_n - 1]  # expect_n-th score
        legal_rel = np.where(score >= boundary - tol)[0]
    else:  # l2
        diff = X[m_idx] - q
        score = np.einsum('ij,ij->i', diff, diff)     # squared Euclidean, smaller = more similar
        boundary = np.sort(score)[expect_n - 1]
        legal_rel = np.where(score <= boundary + tol)[0]
    return set(m_idx[legal_rel].tolist())


# ====================== fvecs / ivecs standard formats ======================
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
        buf[base : base + 4] = struct.pack('I', d)
        buf[base + 4 : base + row_bytes] = data[i].tobytes()
    with open(filename, "wb") as f:
        f.write(buf)
    print(f"\t{filename} wrote")

def ivecs_read(fname):
    x = np.memmap(fname, dtype=np.int32, mode='r')
    d = x[0]
    arr = x.reshape(-1, d + 1)
    return arr[:, 1:]
