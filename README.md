# PART: Baking Attributes into High-Dimensional Vectors for Filtered Approximate Nearest Neighbor Search

**Attribute-aware vector search with phase rotation (APR / HAPR).**

PART is the reference implementation accompanying the paper of the same name. It
extends [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library) with
**Attribute Phase Rotation (APR)** and its hybrid layout **HAPR (Hybrid
Attribute Partitioning Rotation)**, enabling fast *filtered* approximate nearest
neighbor search in which each query carries categorical **equality** and/or
numeric **range** predicates over point attributes.

The idea is to encode a point's attribute tuple as a deterministic, orthogonal
rotation of the vector. On a full attribute match the rotated-space distance is
exactly the original distance, so existing RaBitQ indexes keep working; on a
mismatch the distance is inflated and a lightweight predicate gate discards
non-matching candidates. An optional reranking step recovers the true original
ordering for range/mixed predicates.

> Full method, numeric protocol, and layout contract are documented in
> [`docs/attribute-rotation.md`](docs/attribute-rotation.md). This README is the
> entry point for building, reproducing, and citing the system.

---

## Why PART

- **Filtered search without pre-filtering.** Equality and range predicates are
  folded into the vector geometry, so a single ANN index answers attribute-
  filtered queries with no separate filtered index or post-filtering fallback.
- **Exact for equality, soft for range.** Equality attributes are encoded with an
  isometric phase (rotated distance equals original distance); range attributes
  are encoded with an interval-centre phase (a soft filter that a predicate gate
  and optional rerank clean up).
- **Index-agnostic.** APR is a leaf of `rabitqlib::core`: it only rotates
  vectors. IVF-RaBitQ, SymphonyQG, and HNSW-RaBitQ all consume the rotated
  vectors through their existing builders, and additionally expose optional
  predicate-filtered `search` overloads.

## Method in brief

Let `P = D / 2`. For each dimension pair `j` and attribute `a`, an attribute
value `v` determines a phase `psi_a(v)[j]`. A base point with attribute tuple
`V` is rotated by `sum_a psi_a(v_a)[j]`; a query with target tuple `T` is rotated
by `sum_a psi_a(t_a)[j]`. The relative angle is therefore
`Delta[j] = sum_a (psi_a(t_a)[j] - psi_a(v_a)[j])`.

- On a **full match**, `Delta[j] == 0` for every pair, so rotated-space
  distances are *exactly* the original distances and the original ground truth
  stays valid.
- On a **mismatch**, phases are pseudo-random; averaging over `P` pairs drives
  the expected squared distance towards `||x||^2 + ||q||^2`, and a predicate
  gate (`attr_match`) discards candidates that violate the query's predicate.

The rotation is a block-diagonal Givens transform: it preserves the norm, is
orthogonal and invertible, and is applied **without normalization**. Do not
normalize the input — changing the scale would change the reranking metric and
break comparability with the reference protocol.

### Layouts

| Layout | Predicate | Encoding | Recall without rerank |
| --- | --- | --- | --- |
| `eq` | equality only | isometric phase per attribute value | exact (rotation preserves distance) |
| `range` | range only | interval-centre phase (soft) | soft ceiling (~0.95–0.96); rerank restores |
| `eq,range` (mixed) | equality **and** range | exact phase for `eq`, soft phase for `range` | soft ceiling; rerank restores |

`--layout` takes one token per attribute; its token count must equal
`--attr-dim`. Range length `range_l` (the number of distinct range values) is
auto-inferred from `base_attr` when `--range-l 0` (the default).

---

## Building

PART is a CMake project on top of RaBitQ-Library. Configure and build the core,
index, and the C++ sample tools (which include the APR encoder and the filtered
query drivers):

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_SAMPLES=ON \
  -DRABITQ_BUILD_INDEX=ON \
  -DRABITQ_BUILD_FASTSCAN=ON \
  -DRABITQ_USE_OPENMP=ON
cmake --build build --parallel
ctest --no-tests=error --test-dir build --output-on-failure
```

The CLI tools are produced under `build/bin/` (or `bin/` for an in-tree build):
`attribute_apr_encode`, `ivf_rabitq_indexing`, `ivf_rabitq_querying`, and the
SymphonyQG / HNSW equivalents.

Key CMake options: `RABITQ_BUILD_INDEX`, `RABITQ_BUILD_FASTSCAN`,
`RABITQ_BUILD_SAMPLES`, `RABITQ_BUILD_TESTS`, `RABITQ_BUILD_BENCHMARKS`,
`RABITQ_BUILD_PYTHON_BINDINGS`, and `RABITQ_USE_OPENMP` (required by the ANN
indexes). `src/attribute/apr_rotation.cpp` is compiled with
`-ffp-contract=off -fno-fast-math` so its float32 output matches the NumPy
reference protocol exactly.

> Build environment note: if your shell wraps commands or injects `PYTHONPATH`
> shims, build inside a clean environment (e.g. `env -i` with an explicit
> `PATH`/`CC`/`CXX`) to avoid interference from unrelated tooling.

---

## Attribute-rotation pipeline (end to end)

The paper's filtered-search recipe has four steps. **The index is built on the
rotated vectors, never on the original vectors.**

### 1. Encode attributes as rotation

```bash
./bin/attribute_apr_encode \
  --base    <base.fvecs> \
  --query   <query.fvecs> \
  --base-attr  <base_attr.bin> \
  --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 \
  --layout eq,range \
  --standalone-seed 1234 \
  --out-dir data/apr
```

This writes `apr_base.fvecs`, `apr_query.fvecs`, and (for HAPR layouts) a
plain-text `hapr_meta.txt` recording the layout and `range_l`. The rotation is
deterministic given `--standalone-seed` (PART's own salt derivation) or an
explicit `--salt-file`.

### 2. Cluster the rotated vectors

Cluster the rotated base with `k = 4096` (any k-means; faiss is convenient) and
write the centroids as `centroids.fvecs` and each point's nearest-centroid
assignment as `cluster_ids.ivecs` (one `int32` per point). The centroids and
assignments must live in the *same* rotated space as the index input.

### 3. Build the IVF-RaBitQ index on rotated vectors

```bash
./bin/ivf_rabitq_indexing \
  data/apr/apr_base.fvecs centroids.fvecs cluster_ids.ivecs \
  32 apr.index l2 false
```

(`32` is the code width in bits; `false` disables the faster raw-vector
rerank path inside the index — PART supplies its own reranking in step 4.)

### 4. Query with the predicate gate, optionally reranked

```bash
# No rerank: fast, but range/mixed recall is capped by the soft filter.
./bin/ivf_rabitq_querying apr.index data/apr/apr_query.fvecs GT.ivecs \
  --base-attr  <base_attr.bin> \
  --query-attr <query_attr.bin> \
  --attr-dim 2 --layout eq,range

# With rerank: rescored by the original L2 distance, restoring true top-K.
./bin/ivf_rabitq_querying apr.index data/apr/apr_query.fvecs GT.ivecs \
  --base-attr  <base_attr.bin> \
  --query-attr <query_attr.bin> \
  --attr-dim 2 --layout eq,range \
  --rerank --orig-base <orig_base.fvecs> --orig-query <orig_query.fvecs> \
  --rerank-factor 10
```

`--rerank` requires `--orig-base` / `--orig-query` (the **un-rotated** vectors,
row-aligned with the rotated inputs). The query sweeps `nprobe` adaptively and
reports QPS and recall@100 against the supplied ground truth.

> Gotchas: do **not** renormalize the vectors, and do **not** apply APR again
> inside the index — the query is already rotated by the encoder, so a second
> rotation would double-rotate it.

---

## Attribute binary format

`*.bin` attribute files carry an 8-byte little-endian header `[rows, cols]`
(`int32`) followed by row-major `int32` values:

- **base attributes** `<base_attr.bin>`: `[N, A]` where column `a` holds the
  actual attribute value for attribute `a` (an equality value, or the range
  value for range attributes).
- **query attributes** `<query_attr.bin>`: `[nq, qstride]` where `qstride` is
  `1` for an equality-only attribute and `3` for a range attribute
  (`[eq_value, lo, hi]` or `[lo, hi]` depending on layout).

The encoder validates that the layout and the query stride are consistent.

---

## Datasets and reproduction

**Unit / fixture tests** (self-contained, no external data) live under
`tests/fixtures/attribute/` and are exercised by `ctest` and
`tests/python/*`. They validate APR/HAPR against the NumPy reference protocol
(norm preservation, byte compatibility, and predicate-gate correctness for IVF
and SymphonyQG indexes).

**Paper-scale experiments** use the SIFT base/query vectors with generated
attribute columns:

- `A3_C100_w10` — three independent range attributes (layout `range,range,range`).
- `MIX_EQ_C10_R100_w10` — one equality attribute plus one range attribute
  (layout `eq,range`).

Each experiment rotates the vectors (`--layout` per above), clusters the rotated
base at `k = 4096`, builds a 32-bit IVF-RaBitQ index, and compares **no-rerank**
vs **rerank (F = 5 / 10)**. Representative, dataset-independent findings:

| Configuration | Recall@100 ceiling / convergence | Note |
| --- | --- | --- |
| No rerank (range) | ~0.956 | HAPR soft-filter ceiling in the original space |
| Rerank F = 5 / 10 (range) | ~0.997 | original L2 rerank recovers true top-K |
| No rerank (mixed) | ~0.948 | equality exact, range soft |
| Rerank F = 10 (mixed) | ~0.998 | |

The predicate gate reports `Predicate violations: 0` in every configuration,
confirming the equality/range/mixed filters are applied correctly. The exact
datasets and scripts used for the paper's figures are provided alongside the
submission; the commands above are the minimal reproducible recipe.

---

## Relationship to RaBitQ-Library

PART is layered on RaBitQ-Library: it reuses the quantization, indexing, and
search infrastructure and adds `rabitqlib::attribute` (APR/HAPR) plus the
filtered-search CLI tools. RaBitQ-Library's own README, API docs, and examples
remain valid for the underlying indexes and quantizers; this repository's
contribution is the attribute-rotation layer and the filtered-search workflow
above.

## Tests

```bash
ctest --no-tests=error --test-dir build --output-on-failure   # C++ tests
python -m pytest tests/python                                  # Python bindings tests
```

The Python suite includes path-handling tests that deliberately use non-ASCII
filenames, confirming the bindings accept international (e.g. UTF-8 / CJK) paths.

## Citation

If you use PART in your research, please cite the accompanying paper. The exact
title, authors, and venue should be filled in from the paper:

```bibtex
@inproceedings{part2027,
  title     = {PART: ...},   % fill in from the paper
  author    = {...},
  year      = {2027},
  note      = {To appear}
}
```

PART builds on RaBitQ-Library (Gao et al.). Please also cite RaBitQ-Library
where appropriate.

## License

PART is released under the [Apache License 2.0](LICENSE), consistent with
RaBitQ-Library. The vendored Eigen copy under `include/rabitqlib/third/Eigen`
retains its own MPL2/LGPL/BSD license.
