import os
# Only pin the GPU if the user has not already chosen one (default device 0).
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import struct
import numpy as np
import time
import faiss


#######################################################
# Nearest neighbor search functions
#######################################################

def get_nearestneighbors(xq, xb, k, device, metric="ip", verbose=False):
    assert device in ["cpu", "cuda"], "device must be 'cpu' or 'cuda'"
    assert metric in ["l2", "ip"], "metric must be 'l2' or 'ip'"
    assert xq.dtype == np.float32 and xb.dtype == np.float32, "vectors must be float32"
    assert xq.shape[-1] == xb.shape[-1], "query and base dimensionality must match"

    if verbose:
        print(f"Computing Nearest Neighbors (Faiss), metric={metric}, device={device}")

    dim = xq.shape[1]
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    if device == 'cuda':
        index = faiss.index_cpu_to_all_gpus(index)

    start = time.time()
    index.add(xb)
    D, I = index.search(xq, k)
    if verbose:
        print(f"  NN search ({device}, {metric}) done in {time.time() - start:.2f} s")

    return I, D


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


def recompute_gt_range(X, Q, base_attr, query_ranges, metric, K):
    """Range filter: independent brute-force ground-truth recomputation. Returns (G, sel),
    same semantics as the range dataset builder.
    query_ranges: (QN, 2A) int, columns interleaved as [lo_0, hi_0, lo_1, hi_1, ...];
    a base vector matches when every attribute falls inside its query interval (multi-attribute AND);
    within the match set take top-K by metric.
    """
    N, A = base_attr.shape
    QN = Q.shape[0]
    G = np.full((QN, K), -1, dtype=np.int32)
    sel = np.zeros(QN, dtype=np.float64)
    for i in range(QN):
        rng_i = query_ranges[i].reshape(A, 2)               # (A,2) [lo,hi]
        mask = np.all((base_attr >= rng_i[:, 0]) & (base_attr <= rng_i[:, 1]), axis=1)
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
def fvecs_write(filename, data):
    data = np.ascontiguousarray(data, dtype=np.float32)
    n, d = data.shape
    row_bytes = 4 + d * 4
    buf = bytearray(n * row_bytes)
    for i in range(n):
        base = i * row_bytes
        buf[base : base + 4] = struct.pack('I', d)
        buf[base + 4 : base + row_bytes] = data[i].tobytes()
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


#######################################################
# Attribute rotation encoding (disjoint rotation)
#######################################################

def _rotated(x, angles):
    """Apply angle angles[:,j] to each adjacent dimension pair (2j, 2j+1) of x.
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


def _hash_to_phases(ids, d, seed):
    """Hash each attribute ID into (d//2) phases in [0,2π). A phase depends only on the
    value itself + seed, independent of other values in the dataset -> the data side and
    query side must produce the same phase for the same value (critical, otherwise the
    phase cannot cancel out)."""
    P = d // 2
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    base_rng = np.random.default_rng(seed)
    salt = base_rng.integers(0, 2 ** 31, size=P, dtype=np.int64)   # one salt per dimension pair
    out = np.empty((len(ids), P))
    for j in range(P):
        h = (ids * 2654435761) ^ (salt[j] * 40503)
        h = (h ^ (h >> 15)) & 0x7FFFFFFF
        out[:, j] = (h / float(0x7FFFFFFF)) * 2.0 * np.pi         # in [0,2π)
    return out


def block_split(A, P, scheme="contiguous"):
    """Split P dimension pairs evenly across A attributes, non-overlapping.
    scheme='contiguous': contiguous split (default, cache/vectorization friendly).
    Returns list[length A], each item an np.ndarray of block dimension-pair indices."""
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
    """Disjoint-block categorical encoding.
    attrs_cat: (N, A) int; blocks[a] is the array of dimension-pair indices occupied by attribute a;
    seeds[a] is the per-attribute salt. Each attribute block applies an independent hash rotation ->
    zero cross-talk between blocks -> AND / IN / OR(DNF) filters all hold.
    A=1, blocks=[[0..P-1]] is equivalent to phase_stack_encode_base. Returns (N, d) float32."""
    P = base.shape[1] // 2
    E = base.copy()
    for a in range(attrs_cat.shape[1]):
        full = _hash_to_phases(attrs_cat[:, a], base.shape[1], seeds[a])   # (N, P)
        ang = np.zeros((full.shape[0], P), dtype=np.float64)
        ang[:, blocks[a]] = full[:, blocks[a]]
        E = _rotated(E, ang)
    return E.astype(np.float32)


#######################################################
# Metric
#######################################################

def recall_at_k(res, gt, K):
    # Note: both ground-truth and predictions use -1 for "invalid/empty slot"; these must be
    # excluded, otherwise -1 counts in both denominator and numerator and recall goes out of
    # range (>1 or even 100x).
    rec_list = []
    for r, g in zip(res, gt):
        g_valid = [x for x in g[:K] if x >= 0]   # keep only valid gt indices
        if len(g_valid) == 0:
            continue                            # query has no valid gt, skip from averaging
        g_set = set(g_valid)
        hit = sum(1 for x in r[:K] if x in g_set)
        rec_list.append(hit / len(g_set))
    return float(np.mean(rec_list)) if rec_list else 0.0
