#!/usr/bin/env python3
"""
port_arxiv_em.py — Port the real arxiv-large em (Exact-Match) attribute labels 1:1 by index
onto the sift / msmarco / openai1536 datasets, reuse compute_eq_attr_gt.cpp + recompute_gt
to recompute attribute-filtered ground-truth, and emit the same artifact directory layout
as generate_one.

Porting semantics (consistent with the plan; NOT regenerating uniform attributes):
  - arxiv base vector i's attribute label (number_of_sub_categories) -> target dataset base vector i;
  - arxiv query j's label -> target dataset query j.
  Target datasets have base <= 2.7M and query = 10,000, so a precise prefix mapping is possible
  (arxiv labels are i.i.d., so the prefix distribution is identical). Thus the new dataset's em
  selectivity / distribution matches arxiv exactly, enabling a fair comparison.

Artifacts are written to <ds_dir>/Arxiv_Attr/ (replacing the synthetic name A1_C6):
    base_attr.bin  query_attr.bin  groundtruth.ivecs  queries.json
    + symlinks to the root base/query vectors (for convenient re-verification).

The directory name is decoupled from RaBitQ internal suffixes: files under Arxiv_Attr are
"no-suffix" names; the RaBitQ grid script internally uses ADIM=1 / ACARD=C(=max+1) only to
construct the suffix-bearing staging file name (groundtruth_A1_C*.ivecs) needed by search.cpp
and the --C radix for compute_eq_attr_gt; the two do not interfere.

This script is self-contained: it inlines the pure-numpy IO / recompute / topk_legal_set
portions of _dataset_utils and the candidate/verification logic of gen_eq_attr_gt, so it does
NOT depend on faiss (the top-level `import faiss` in _dataset_utils is unrelated to this pipeline)
and can be run directly with the default python3.

Usage:
  python3 port_arxiv_em.py --threads 0
  python3 port_arxiv_em.py --only sift
"""
import os
import sys
import json
import struct
import argparse
import subprocess

import numpy as np

# ---------------- arxiv source label paths ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
ARXIV_ROOT = os.path.abspath(os.path.join(HERE, '..', 'arxiv-large'))
ARXIV_BASE_JSONL = os.path.join(ARXIV_ROOT, 'database_attributes.jsonl')
ARXIV_QUERY_JSONL = os.path.join(ARXIV_ROOT, 'em', 'query_attributes.jsonl')

# ---------------- target dataset config: (ds_dir, base_name, query_name, metric) ----------------
DATASET_ROOT = os.path.abspath(os.path.join(HERE, '..'))
DATASETS = {
    'sift':      ('sift',      'sift_base.fvecs',      'sift_query.fvecs',      'l2'),
    'msmarco':   ('msmarco',   'msmarco_base.fvecs',   'msmarco_query.fvecs',   'l2'),
    'openai1536':('openai1536','openai1536_base.fvecs','openai1536_query.fvecs','ip'),
}


# =====================================================================
# Inlined: pure-numpy part of _dataset_utils (IO + recompute_gt + topk_legal_set)
# (kept in sync with _dataset_utils.py; avoids the top-level faiss dependency)
# =====================================================================
def attr_bin_write(path, arr, dtype=np.uint32):
    """Write a self-describing binary: 8-byte [N, A] (int32 little-endian) header + N*A elements of dtype (row-major)."""
    data = np.ascontiguousarray(arr, dtype=dtype)
    n, A = data.shape
    with open(path, "wb") as fp:
        fp.write(struct.pack("II", int(n), int(A)))   # header
        fp.write(data.tobytes(order="C"))            # data payload


def load_attr(path, dtype, A_default=1):
    """Read an attribute bin, auto-detecting the self-describing header / headerless layout."""
    sz = os.path.getsize(path)
    raw = np.fromfile(path, dtype=np.int32)
    n_hdr, A_hdr = int(raw[0]), int(raw[1])
    if sz > 8 and (sz - 8) == n_hdr * A_hdr * 4:
        return raw[2:].view(dtype).reshape(n_hdr, A_hdr).copy()
    total = sz // 4
    A = A_default
    N = total // A
    assert N * A == total, f'{path}: file size {sz} is not divisible by A={A}'
    return np.fromfile(path, dtype=dtype).reshape(N, A).copy()


def recompute_gt(X, Q, base_attr, query_attr, metric, K):
    """Equality filter: independent brute-force ground-truth recomputation. Returns (G, sel)."""
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
    """Tie-tolerant top-K legal set, shared by self-check and verifier."""
    if expect_n <= 0 or m_idx.size == 0:
        return set()
    if metric == 'ip':
        score = q @ X[m_idx].T
        boundary = np.sort(score)[::-1][expect_n - 1]
        legal_rel = np.where(score >= boundary - tol)[0]
    else:
        diff = X[m_idx] - q
        score = np.einsum('ij,ij->i', diff, diff)
        boundary = np.sort(score)[expect_n - 1]
        legal_rel = np.where(score <= boundary + tol)[0]
    return set(m_idx[legal_rel].tolist())


def fvecs_read(fname):
    x = np.memmap(fname, dtype=np.int32, mode='r')
    d = x[0]
    arr = x.view(np.float32).reshape(-1, d + 1)
    return arr[:, 1:]


def ivecs_write(filename, data):
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


def ivecs_read(fname):
    x = np.memmap(fname, dtype=np.int32, mode='r')
    d = x[0]
    arr = x.reshape(-1, d + 1)
    return arr[:, 1:]


# =====================================================================
# Inlined: candidate / verification logic of gen_eq_attr_gt (same source as generator/verifier)
# =====================================================================
def run_compute_eq_attr_gt(cpp_bin, base_path, query_path, base_attr_path,
                           query_attr_path, metric, K, C, margin, threads,
                           cand_path, count_path):
    """Invoke the multi-threaded C++ program to emit candidate files and per-query match counts.
    C is the attribute cardinality (signature radix); it must be >= max attribute value + 1,
    otherwise A>=2 causes signature collisions."""
    cmd = [cpp_bin,
           '--base', base_path, '--query', query_path,
           '--base-attr', base_attr_path, '--query-attr', query_attr_path,
           '--metric', metric, '--K', str(K), '--C', str(C),
           '--margin', str(margin),
           '--threads', str(threads),
           '--cand-out', cand_path, '--count-out', count_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('compute_eq_attr_gt failed:\n' + (r.stderr or r.stdout))
    if r.stderr.strip():
        print('   ', r.stderr.strip())


def finalize_via_recompute(cand_path, count_path, X, Q, query_attr, metric, K,
                           totalcap, N):
    """Read C++ candidates and compute the exact top-K on the candidate subset using
    recompute_gt (same logic as the verifier)."""
    QN = Q.shape[0]
    cand = np.fromfile(cand_path, dtype=np.int32)
    cand = cand.reshape(QN, totalcap)
    count = np.fromfile(count_path, dtype=np.int64)
    assert count.shape[0] == QN

    G = np.full((QN, K), -1, dtype=np.int32)
    sel = count.astype(np.float64) / N
    A = query_attr.shape[1]
    for i in range(QN):
        c = cand[i]
        c = c[c >= 0]
        if c.size == 0:
            continue
        Xc = np.ascontiguousarray(X[c], dtype=np.float32)
        bsub = np.broadcast_to(query_attr[i:i + 1], (c.size, A)).copy().astype(np.int64)
        Gc, _ = recompute_gt(Xc, Q[i:i + 1], bsub, bsub, metric, K)
        g0 = Gc[0][Gc[0] >= 0]
        G[i, :g0.size] = c[g0]
    return G, sel


def self_check(G, X, Q, base_attr, query_attr, metric, K, QN, n=256):
    """Verification identical to the independent verifier: 'subset + 1e-6 boundary' check,
    tie-tolerant."""
    rv = np.random.default_rng(20241010)
    sub = rv.choice(QN, size=min(n, QN), replace=False)
    Gref, _ = recompute_gt(X, Q[sub], base_attr, query_attr[sub], metric, K)
    for j, i in enumerate(sub):
        stored = G[i]
        mask = np.all(base_attr == query_attr[i], axis=1)
        m_idx = np.where(mask)[0]
        expect_n = min(K, m_idx.size)
        valid = stored[stored >= 0]
        if valid.size != expect_n:
            return False
        if m_idx.size > 0 and valid.size > 0:
            legal = topk_legal_set(X, Q[i], m_idx, metric, expect_n)
            if not set(valid.tolist()).issubset(legal):
                return False
    return True


# =====================================================================
# Porting main flow
# =====================================================================
def load_arxiv_labels():
    """Read arxiv em labels: base = number_of_sub_categories (scalar integer), query = label."""
    print(f'[arxiv] reading base attributes: {ARXIV_BASE_JSONL}')
    base_vals = []
    with open(ARXIV_BASE_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            base_vals.append(int(o['number_of_sub_categories']))
    print(f'[arxiv] reading query attributes: {ARXIV_QUERY_JSONL}')
    query_labels = []
    with open(ARXIV_QUERY_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            query_labels.append(int(json.loads(line)['label']))
    ab = np.array(base_vals, dtype=np.int64)
    aq = np.array(query_labels, dtype=np.int64)
    print(f'[arxiv] base N={ab.shape[0]}  query Nq={aq.shape[0]}  '
          f'base value range=[{ab.min()},{ab.max()}]  query value range=[{aq.min()},{aq.max()}]')
    return ab, aq


def port_one(ds_dir, base_name, query_name, metric, K, arxiv_base, arxiv_query,
             cpp_bin, margin, threads, no_verify=False):
    """For one target dataset: map arxiv labels by index -> write attribute bins ->
    compute_eq_attr_gt produces candidates -> recompute_gt finalizes fixed-length GT ->
    write ivecs/json -> symlink vectors -> self-check.
    Artifacts are written to <ds_dir>/Arxiv_Attr/. Returns a stats dict.
    """
    assert metric in ('ip', 'l2')
    A = 1
    out_dir = os.path.join(ds_dir, 'Arxiv_Attr')
    os.makedirs(out_dir, exist_ok=True)

    base_path = os.path.join(ds_dir, base_name)
    query_path = os.path.join(ds_dir, query_name)
    print(f'[load] base={base_name} query={query_name} (metric={metric}, K={K})')
    X = fvecs_read(base_path).astype(np.float32)
    Q = fvecs_read(query_path).astype(np.float32)
    N, D = X.shape
    QN = Q.shape[0]
    print(f'      X={X.shape} Q={Q.shape}')

    # -------- Map arxiv labels 1:1 by index (prefix) --------
    assert N <= arxiv_base.shape[0], f'target base({N}) > arxiv base({arxiv_base.shape[0]})'
    assert QN <= arxiv_query.shape[0], f'target query({QN}) > arxiv query({arxiv_query.shape[0]})'
    base_attr = arxiv_base[:N].reshape(N, A).astype(np.int64)
    query_attr = arxiv_query[:QN].reshape(QN, A).astype(np.int64)
    # radix C must be >= max attribute value + 1 (A>=2 prevents signature collision; A=1 prevents bucket overflow)
    max_val = int(max(base_attr.max(), query_attr.max()))
    C = max_val + 1
    base_attr_bin = os.path.join(out_dir, 'base_attr.bin')
    query_attr_bin = os.path.join(out_dir, 'query_attr.bin')
    attr_bin_write(base_attr_bin, base_attr, np.uint32)
    attr_bin_write(query_attr_bin, query_attr, np.int32)
    print(f'      base_attr={base_attr.shape} query_attr={query_attr.shape} '
          f'radix C={C} (max_val={max_val})')

    # -------- C++ multi-thread: top-(K+M) candidates in the match set + match counts --------
    totalcap = K + margin
    cand_path = os.path.join(out_dir, '.cand.int32')
    count_path = os.path.join(out_dir, '.count.int64')
    run_compute_eq_attr_gt(cpp_bin, base_path, query_path, base_attr_bin,
                           query_attr_bin, metric, K, C, margin, threads,
                           cand_path, count_path)

    # -------- Python final: exact top-K on candidates via recompute_gt (same as verifier) --------
    G, sel = finalize_via_recompute(cand_path, count_path, X, Q, query_attr,
                                    metric, K, totalcap, N)
    if os.path.exists(cand_path):
        os.remove(cand_path)
    if os.path.exists(count_path):
        os.remove(count_path)
    avg_sel = float(sel.mean())

    # -------- Write ivecs / json --------
    ivecs_write(os.path.join(out_dir, 'groundtruth.ivecs'), G)
    meta = [{'type': 'eq', 'attr': list(range(A)),
             'value': [int(v) for v in query_attr[i]],
             'selectivity': float(sel[i]), 'metric': metric}
            for i in range(QN)]
    with open(os.path.join(out_dir, 'queries.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'      avg selectivity = {avg_sel:.4f} (min {sel.min():.4f}, max {sel.max():.4f})')

    # -------- Symlink root vectors for re-verification --------
    for nm in (base_name, query_name):
        link = os.path.join(out_dir, nm)
        target = os.path.abspath(os.path.join(ds_dir, nm))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)

    # -------- Self-check (attribute IO round-trip + recompute_gt sample comparison) --------
    if not no_verify:
        ba = load_attr(base_attr_bin, np.uint32, A_default=A).astype(np.int64)
        qa = load_attr(query_attr_bin, np.int32, A_default=A).astype(np.int64)
        G_read = ivecs_read(os.path.join(out_dir, 'groundtruth.ivecs')).astype(np.int64)
        assert ba.shape == base_attr.shape and qa.shape == query_attr.shape, 'attribute shape mismatch on read-back'
        assert np.array_equal(ba, base_attr) and np.array_equal(qa, query_attr), 'attribute content mismatch on read-back'
        assert G_read.shape == G.shape, 'gt shape mismatch on read-back'
        ok = self_check(G, X, Q, base_attr, query_attr, metric, K, QN) and \
             self_check(G_read, X, Q, base_attr, query_attr, metric, K, QN)
        assert ok, 'self-check failed: sample differs from recompute_gt (verifier semantics)'
        print(f'      [self-check] PASS  (attribute IO round-trip + 256-sample subset of verifier legal set)')

    return {'ds_dir': ds_dir, 'metric': metric, 'K': K, 'N': N, 'QN': QN, 'D': D,
            'radix_C': C, 'avg_sel': avg_sel,
            'min_sel': float(sel.min()), 'max_sel': float(sel.max()),
            'out_dir': out_dir, 'self_check': 'SKIP' if no_verify else 'PASS'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds-dir', default=DATASET_ROOT, help='dataset root (contains sift/ msmarco/ openai1536/)')
    ap.add_argument('--only', default=None, choices=list(DATASETS.keys()),
                    help='only generate the specified dataset (default: all)')
    ap.add_argument('--K', type=int, default=100)
    ap.add_argument('--cpp-bin', default=os.path.join(HERE, 'compute_eq_attr_gt'))
    ap.add_argument('--margin', type=int, default=512,
                    help='candidate margin M: candidate count = K+M, covering float-boundary ties to ensure the true top-K is included')
    ap.add_argument('--threads', type=int, default=0, help='OpenMP threads, 0 = use all cores')
    ap.add_argument('--no-verify', action='store_true',
                    help='skip self-check (attribute IO round-trip + recompute_gt sample)')
    args = ap.parse_args()

    arxiv_base, arxiv_query = load_arxiv_labels()

    targets = [args.only] if args.only else list(DATASETS.keys())
    results = []
    for name in targets:
        rel, base_name, query_name, metric = DATASETS[name]
        ds_dir = os.path.join(args.ds_dir, rel)
        print(f'\n========== port -> {name} ({ds_dir}) ==========')
        r = port_one(ds_dir, base_name, query_name, metric, args.K,
                     arxiv_base, arxiv_query, args.cpp_bin, args.margin,
                     args.threads, no_verify=args.no_verify)
        results.append((name, r))
        print('[ok]', name, json.dumps({k: v for k, v in r.items() if k != 'ds_dir'}, ensure_ascii=False))

    print('\n==================== summary ====================')
    for name, r in results:
        print(f'  {name:12s} N={r["N"]:>9d} QN={r["QN"]:>6d} D={r["D"]:>5d} '
              f'radixC={r["radix_C"]:>3d} avg_sel={r["avg_sel"]:.4f} self_check={r["self_check"]}')
    print('==============================================')


if __name__ == '__main__':
    main()
