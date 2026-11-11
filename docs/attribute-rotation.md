# Attribute Phase Rotation (APR)

`rabitqlib::attribute` is an index-agnostic preprocessing layer. It encodes a
point's categorical attributes as a deterministic, orthogonal rotation of the
vector, and applies the same rotation to a query using the query's target
attribute tuple. It is a leaf of `rabitq_core`: it does not reference IVF,
SymphonyQG, HNSW, quantizers or any persisted index format, so both IVF and
SymphonyQG consume the rotated vectors through their existing builders. The
indexes additionally offer optional predicate-filtered `search` overloads
for equality, range and mixed attributes, backed by `attr_filter.hpp`; the
rotation itself is never applied inside an index.

## Why it works

Let `P = D / 2`. For every dimension pair `j` and attribute `a`, a value `v`
determines a phase `psi_a(v)[j]`. A base vector with attribute tuple
`V = (v_0 .. v_{A-1})` is rotated by `sum_a psi_a(v_a)[j]`; a query with target
tuple `T` is rotated by `sum_a psi_a(t_a)[j]`. The relative angle is therefore

```
Delta[j] = sum_a (psi_a(t_a)[j] - psi_a(v_a)[j])
```

* On a full match `Delta[j] == 0` for every pair, so distances in the rotated
  space are **exactly** the distances in the original space and the original
  ground truth stays valid.
* On a mismatch the phases are pseudo-random; averaging over `P` pairs drives
  `E[cos Delta]` towards zero, inflating the expected squared distance towards
  `||x||^2 + ||q||^2`. A later filtering gate (see `attr_match`) can then
  discard non-matching candidates.

The rotation is a block-diagonal Givens transform: it preserves the norm, is
orthogonal and invertible, and is applied **without normalization**. Do not
normalize the input: changing the scale would change the reranking metric and
break comparability with existing rotation experiments.

## Numeric protocol

All arithmetic in the phase computation is unsigned 64-bit (modulo `2^64`).

| Constant | Value |
| --- | --- |
| `mix` first multiplier | `0xBF58476D1CE4E5B9` |
| `mix` second multiplier | `0x94D049BB133111EB` |
| golden ratio | `0x9E3779B97F4A7C15` |
| phase mask | `0x7FFFFFFF` |

```
mix64(x) = let x = (x ^ (x >> 30)) * C1
           let x = (x ^ (x >> 27)) * C2
           in  x ^ (x >> 31)

base(v)  = mix64(v * GOLDEN)
h        = mix64(base(v) ^ salt[a * P + j]) & 0x7FFFFFFF
psi      = (double(h) / 2147483647.0) * (2 * pi)
total[j] = sum over a of psi_a(v_a)[j]
```

For each pair `j` of coordinates `(2j, 2j + 1)`:

```
c = cos(total[j]);  s = sin(total[j]);
out[2j]     = float32(double(x[2j])     * c - double(x[2j + 1]) * s);
out[2j + 1] = float32(double(x[2j])     * s + double(x[2j + 1]) * c);
```

`src/attribute/apr_rotation.cpp` is compiled with `-ffp-contract=off
-fno-fast-math` and keeps each multiply in a separate `double` statement. This
prevents FMA contraction from shifting the float32 output away from the NumPy
reference (tolerance `1e-5`).

## Salt

`AprRotationPlan::make` expects `A * P` little-endian `uint64` values: `A`
blocks of `P` salts, one block per attribute. To reproduce the salt of
published experiments, generate it with NumPy (this snippet is documentation
only; no Python source is shipped with the library):

```python
import numpy as np
D, A, seed = 128, 1, 1234
P = D // 2
np.concatenate([np.random.default_rng(seed + a).integers(0, 2**63, size=P, dtype=np.int64)
                for a in range(A)]).astype("<i8").tofile("apr_salt.bin")
```

`AprRotationPlan::make_standalone(config, seed)` instead derives the salt from
splitmix64 in pure C++, so the library is usable with no Python at all. Its
output is **not** comparable with published experiments; use it for
self-contained demos and tests only. Changing `A`, `D` or the salt changes the
base vectors, so an index must be rebuilt for each attribute configuration.

`tests/fixtures/attribute/apr_salt.bin` is a fixed salt for `D=8, A=1,
seed=1234`; the unit test embeds the same values as C++ constants.

## File formats

All files are little-endian.

| File | Layout |
| --- | --- |
| attribute `.bin` | `uint32 rows`, `uint32 cols`, then `rows * cols` values. |
| salt `.bin` | exactly `A * (D / 2)` `uint64` values (NumPy `int64` bits). |
| `.fvecs` | one `uint32` dimension per row followed by `D` `float32` values. |

Base attributes are `uint32`; query attributes are `int32`. A headerless
payload is accepted when both expected dimensions are supplied to
`read_attr_bin`.

For HAPR layouts the query file is wider than the base file: an `eq` column
occupies one target value, a `range` column occupies two (`[lo, hi]`), so
`qstride = sum(2 for range else 1)`. For `--layout eq,range` (`A=2`) the base
file is `(N,2) uint32` and the query file is `(Nq,3) int32`; the 8-byte header
always reports the real column count. Non-default layouts also produce a
plain-text `hapr_meta.txt` (`key=value` lines) recording the layout, `range_l`,
salt source, `dim`, `attr_dim`, `qstride` and output names. All-equality runs
write no meta.

This expanded query layout is also the retrieval-layer contract: the QG / IVF
gates and both query CLIs use `query_stride(layout, attr_dim)` as the single
source of truth for the row width, while `base_attr` always keeps one column
per attribute.

`attr_io.hpp` provides `read_attr_bin`, `write_attr_bin`, `read_salt_file` and
`write_fvecs`; `AttributeMatrix<T>` also satisfies the reader contract of
`rabitqlib::load_vecs` / `load_bin`.

## C++ API

```cpp
#include "rabitqlib/attribute/apr_rotation.hpp"
#include "rabitqlib/attribute/attr_io.hpp"

AprConfig config{128, 1};  // dim, attribute count
auto salt = read_salt_file("apr_salt.bin", config.attr_dim, config.dim / 2);
auto plan = AprRotationPlan::make(config, salt.data(), salt.size());

// src / dst are row-major float32 buffers of rows * dim; in-place is allowed.
rotate_batch(plan, base, num_base, base_attrs, rotated_base, 0);   // uint32 attrs
rotate_batch(plan, query, num_query, query_attrs, rotated_query, 0); // int32 targets
```

`rotate_batch`'s last argument requests row-wise workers (`0` = serial).
`attr_filter.hpp` provides three index-agnostic predicates: `attr_match`
(equality, `-1` wildcard), `attr_match_range` (one `[lo, hi]` pair per
attribute) and `attr_match_mix` (a per-attribute `AttributeKind` layout).
`predicate_matches` dispatches on the layout and is shared by the QG / IVF
gates and the query CLIs' violation counters. `query_stride` in `types.hpp`
derives the expanded query row width; `predicate_row_valid` rejects negative
values and inverted windows at gate entry points.

HAPR layouts use `AprRotationPlan::make_hapr` / `make_hapr_standalone` plus
`rotate_batch_hapr(plan, src, rows, values, dst, threads)`, where `values` is a
`rows * attr_dim` matrix of doubles holding integer equality values, range
values or interval centres. See the HAPR section below for the layout contract.

## Offline encoder

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
./bin/attribute_apr_encode \
    --base data/base.fvecs --query data/query.fvecs \
    --base-attr data/base_attr.bin --query-attr data/query_attr.bin \
    --dim 128 --attr-dim 1 --salt-file apr_salt.bin --out-dir data/apr
# or, with no Python/salt file at all:
./bin/attribute_apr_encode ... --standalone-seed 1234 --out-dir data/apr

# range-only and mixed HAPR layouts:
./bin/attribute_apr_encode \
    --base data/base.fvecs --query data/query.fvecs \
    --base-attr data/base_attr.bin --query-attr data/query_range.bin \
    --dim 128 --attr-dim 1 --layout range --range-l 100 \
    --standalone-seed 1234 --out-dir data/hapr_range
./bin/attribute_apr_encode \
    --base data/base.fvecs --query data/query.fvecs \
    --base-attr data/base_mix_attr.bin --query-attr data/query_mix.bin \
    --dim 128 --attr-dim 2 --layout eq,range \
    --standalone-seed 1234 --out-dir data/hapr_mixed
```

The encoder writes `apr_base.fvecs` and `apr_query.fvecs` (override with
`--base-out` / `--query-out`). `--range-l 0` (the default) infers `L` from the
base range columns as `max - min + 1`. New layouts also write `hapr_meta.txt`,
overridable with `--meta-out`. Existing index builders and query tools can
consume those files unchanged.

## HAPR range / mixed encoding

`--layout` declares one role per attribute column: `eq` (equality) or `range`
(numeric interval). `--layout range` is range-only, `--layout eq,range` is
mixed, and omitting `--layout` keeps the all-equality default byte-for-byte
unchanged. The token count must equal `--attr-dim`.

**Phase math.** With `P = D/2` pairwise subspaces, all columns share the full
`P` pairs and their phases are *summed* before one Givens rotation per pair:

- `eq` column with value `v`: the existing salted hash phase
  `total[j] += hash(v, salt[a][j]) / 2^31 * 2*pi`. Identical values on the base
  and query side cancel, so a full equality match is distance preserving.
- `range` column with value `v`: `total[j] += pi * (j/P) * (v/L)`, where
  `j = 0..P-1` is the **global** pair index (no `+1` offset, no `base_freq`)
  and `L` is the numeric value range. The relative angle is
  `pi * (j/P) * (a - t)/L`, so points near the window centre stay close while
  distant values are progressively scattered.

There is no `block_split`: disjoint per-attribute blocks would need a second
kernel and conflict with the shared additive phase summation used here. Both
the equality hash chain and the salt source (`--salt-file` reproducing published
experiments, or `--standalone-seed`) are **PART's own**; the reference HAPR
implementation's single-mix hash and per-column seed are deliberately not
copied, because an all-equality layout must stay byte-identical to the existing
golden fixtures and comparable with earlier experiments.

**Query contract (M=1).** A range predicate is a single probe at the interval
centre `t = (lo + hi)/2`; `.5` centres are kept as doubles and never truncated.
Query attribute rows therefore use `qstride = sum(2 for range else 1)` columns
while base rows use one column per attribute. For `--layout eq,range` the base
file is `(N,2) uint32` `[eq, range]` and the query file is `(Nq,3) int32`
`[eq, lo, hi]`. `0 <= lo <= hi` is required; `-1` wildcards and unknown layout
tokens are rejected rather than silently mis-encoded. Output query files keep
`Nq` rows.

**Geometry and files.** Rotation is orthogonal, so norms and L2 distances are
preserved within the float32 rounding of the existing `rotate_row` kernel, and
**inputs are not normalized** — normalizing would change the rerank metric and
break comparability with the published equality experiments. Encoded outputs
are ordinary `N*D` / `Nq*D` fvecs files; no index format or `.index` field
changes.

**Scope.** The encoder stays offline: it never rotates a query inside a search
call. Consuming its range / mixed output end to end is described in the
hard-filtering section below.

`tests/fixtures/attribute/hapr_*` are fixed fixtures generated once from the
existing unnormalised `qg_*_raw.fvecs` (D=64, N=256, NQ=16, L=100,
`--standalone-seed 1234`). `hapr_rotation_test.cpp` checks both golden layouts,
all-equality byte compatibility, norm preservation, the global-`j` phase
coefficients, half-integer centres, window-centre distance preservation,
progressive soft scattering, parallel/serial equality and invalid inputs.

## Constraints and known limitations

* `D` must be a positive even number. APR acts on the first `D` coordinates
  only; it is deliberately decoupled from the quantizer's internal padded
  dimension and from `FhtKacRotator` / `MatrixRotator`.
* Input vectors are **not** normalized.
* Equality is the default; range / mixed layouts are supported end to end.
  A query target of `-1` (wildcard) or any negative value is still rejected: a
  single rotated probe cannot represent "match any value", and a range window
  must satisfy `0 <= lo <= hi`. The match helpers keep `-1` wildcard semantics
  internally, but the QG / IVF gate entry points and both CLIs reject negative
  values; `WildcardPolicy::kMultiProbe` remains reserved in `types.hpp`.
* HAPR uses one probe per query (M=1). Range predicates are encoded at their
  interval centre, which may be a half integer; no `Nq*M` query expansion or
  `split_window` is performed. The hard gate still checks the full `[lo, hi]`
  window, so soft scattering never widens the result set.
* The QG / IVF gates support equality, range and mixed predicates through a
  per-attribute layout. A null layout is the original all-equality path; a
  non-null one uses expanded query rows of `query_stride` int32 values.
* The tuple table is bounded (64 MiB). Configurations with extremely many
  distinct attribute tuples fall back to immediate per-row computation with a
  thread-local scratch buffer.

## Performance design

* `rotate_batch` first scans attribute rows once and builds one `cos/sin` entry
  per distinct tuple. The hot loop is then a read-only, allocation-free lookup
  plus `P` independent 2D rotations: no trigonometric calls, no hashing of
  values, no per-row allocations.
* Vectors are accessed sequentially in row-major order, so the loop is
  bandwidth-bound and friendly to compiler auto-vectorization. Optional
  row-wise parallelism uses the library's OpenMP build option; the default is
  serial and the library never starts threads implicitly.
* No hand-written ISA dispatch is used: this keeps the numeric contract
  identical across backends and avoids the project's SIMD verification matrix.

Representative measurement (illustrative only, not a performance guarantee):
`rotate_batch` on 200,000 rows of dimension 128 with one attribute of
cardinality 16, GCC 13.3.0, `-O3` Release with
`RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF` (baseline x86-64 codegen, no
`-march=native`), on an AMD EPYC 9534. Best of 5 runs: 104.0 ms serial
(1.97 GB/s of read+write traffic, 520 ns/row) and 30.7 ms with 4 threads
(6.67 GB/s, 153 ns/row). Over 1,000 sampled rows the largest relative
squared-norm change was 2.3e-8 (float32 rounding). Results depend on cache,
memory bandwidth and host load; they only illustrate the design above.

## Reproducing from this repository alone

No external project, prebuilt binary or Python source is needed:

```bash
cmake -S . -B build -DRABITQ_BUILD_TESTS=ON -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --no-tests=error --test-dir build --output-on-failure
./bin/attribute_apr_encode \
    --base tests/fixtures/attribute/base.fvecs \
    --query tests/fixtures/attribute/query.fvecs \
    --base-attr tests/fixtures/attribute/base_attr.bin \
    --query-attr tests/fixtures/attribute/query_attr.bin \
    --dim 8 --attr-dim 1 \
    --salt-file tests/fixtures/attribute/apr_salt.bin \
    --out-dir data/apr-demo
```

The unit test compares the C++ output against the golden vectors in
`tests/fixtures/attribute/` (generated once from the NumPy protocol and checked
bit-for-bit against it), and also verifies norm/distance preservation,
parallel-vs-serial agreement, the standalone path and input validation.

## Range / mixed hard filtering

The QG and IVF search gates accept an optional per-attribute layout alongside
the existing attribute buffers. `AttributeKind::kEquality` columns read one
`int32` target, `AttributeKind::kRange` columns read a `[lo, hi]` pair; the
base attribute table always stays `[num_points, attr_dim] uint32`. The query
row width is `qstride = query_stride(layout, attr_dim)`, so `--layout eq,range`
uses `(N,2)` base rows and `(Nq,3)` query rows.

```cpp
const AttributeKind layout[2] = {
    AttributeKind::kEquality, AttributeKind::kRange
};
qg.search(query, k, ids, dists, base_attr, query_attr, 2, layout);
qg.search_batch(queries, nq, k, ids, dists,
                base_attr, query_attrs, 2, layout, /*num_threads=*/1);
ivf.search(query, k, nprobe, ids, dists, base_attr, query_attr, 2, layout);
ivf.search_batch(queries, nq, k, nprobe, ids, dists,
                 base_attr, query_attrs, 2, layout,
                 std::nullopt, /*num_threads=*/1);
```

The old equality-filtered overloads are unchanged and forward to the same
internal implementation with a null layout, so all-equality runs are
byte-for-byte identical. Entry points reject null layouts, negative values
(including `-1` wildcards), inverted or negative range bounds and unknown
`AttributeKind` values. `predicate_matches` dispatches to `attr_match` for a
null layout and to `attr_match_mix` otherwise; the same helper backs the CLIs'
`Predicate violations` counters, so hard and reported predicates cannot drift.

As before, the gate only runs at result-pool insertion. QG checks it in the
traversal loop and `update_results`; IVF checks it in
`insert_candidates_filtered` (one-bit estimates, kept in its own translation
unit to avoid register spills) and in the stage-2 refined loop. Exploration
queues, `visited` sets, cluster routing, distance estimation, `ef` / `nprobe`
budgets and the quantizers are untouched, and no index file field changes.

### Query CLIs

Both `symqg_querying` and `ivf_rabitq_querying` take `--layout eq,range`
(one comma-separated role per `--attr-dim`). Without it the tool behaves
exactly as before: `query_attrs.cols == attr_dim` and the equality gate is
used. With it the CLI validates `query_attrs.cols == qstride`, still requires
`base_attrs.cols == attr_dim`, and passes the layout array to the new search
overloads. `nprobe` sweeps, timed loops, `--rerank` oversampling and the
single-line `Predicate violations` output are shared by all layouts.

```bash
./bin/symqg_querying hapr.index hapr_query.fvecs hapr_mixed_gt.ivecs \
    --base-attr hapr_base_attr.bin --query-attr hapr_query_attr.bin \
    --attr-dim 2 --layout eq,range
./bin/ivf_rabitq_querying hapr_range.index hapr_range_query.fvecs \
    hapr_range_gt.ivecs --base-attr hapr_range_base_attr.bin \
    --query-attr hapr_range_query_attr.bin --attr-dim 1 --layout range
```

### Filtered ground truth

`tests/fixtures/attribute/hapr_mixed_gt.ivecs` and `hapr_range_gt.ivecs`
are fixed top-10 GTs computed inside the predicate-matching subset, padded
with `kPidMax`. They use **rotated-domain L2** between the HAPR-encoded base
and query fvecs, because that is the metric the index optimises: each base
vector is rotated with its own attribute value while the query uses the
interval centre, so a point away from the centre acquires a small phase
offset. The overlap with original-space L2 GT is 1.0 for the mixed fixture
and 0.9875 for the range fixture, but the rotated-domain GT is the exact
target. `qg_hapr_filter_test.cpp` and `ivf_hapr_filter_test.cpp` cover
soundness, batch/parallel agreement, recall against these GTs, all-equality
compatibility and invalid inputs; the IVF tests build 1, 5 and 32-bit
indexes to exercise both insertion paths.

## SymphonyQG integration

`QuantizedGraph<float>` has predicate-filtered overloads of `search` and
`search_batch`. The equality form takes three extra arguments, and the
range / mixed form adds a trailing `AttributeKind` layout (see the
hard-filtering section above):

```cpp
qg.search(query, k, ids, dists, base_attr, query_attr, attr_dim);
qg.search(query, k, ids, dists, base_attr, query_attr, attr_dim, layout);
qg.search_batch(
    queries, num_queries, k, ids, dists, base_attr, query_attrs, attr_dim
);
qg.search_batch(
    queries, num_queries, k, ids, dists, base_attr, query_attrs, attr_dim,
    layout, /*num_threads=*/1
);
```

`base_attr` is a row-major `uint32 [num_points(), attr_dim]` table; query
targets are `int32`, sliced as `query_attrs + i * qstride` for batch queries
(`qstride == attr_dim` for the equality overload). All targets must be
non-negative and range bounds ordered: `-1` wildcards, other negative values
and `lo > hi` raise `std::invalid_argument` because the offline encoder
cannot rotate a wildcard probe. The original `search` / `search_batch` /
`search_batch_with_ef` / `search_with_scratch` signatures and behaviour are
unchanged, and no `.index` field or format version was added.

Only points satisfying `predicate_matches` on every dimension enter the
result heap.
The gate is applied at the two result-pool insertion points (the traversal
loop and `update_results`); the exploration queue, `visited` set, FHT/Kac
rotation and stored graph data are untouched. `ef` remains the traversal
budget, the result heap capacity stays at `k` and no oversampling is performed,
so a query with fewer than `k` matches is padded with `kPidMax` and `+inf`.
Rotation still happens **offline**, before the query reaches `search*`;
applying APR inside the search implementation would double-rotate the query.

`sample/cpp/symqg_querying.cpp` keeps its three positional arguments and adds
the three attribute switches plus optional `--layout`:

```bash
./bin/symqg_indexing apr_base.fvecs 32 400 apr.index l2 8 pipnn 8
./bin/symqg_querying apr.index apr_query.fvecs filtered_gt.ivecs \
    --base-attr base_attr.bin --query-attr query_attr.bin --attr-dim 1
./bin/symqg_querying hapr.index hapr_query.fvecs hapr_mixed_gt.ivecs \
    --base-attr hapr_base_attr.bin --query-attr hapr_query_attr.bin \
    --attr-dim 2 --layout eq,range
```

The three attribute switches must be supplied together; `--layout` is
optional and additionally requires them. Without `--layout` the tool behaves
exactly as before. Recall loops skip `kPidMax` sentinels and the filtered run
prints `Predicate violations: N` (expected `0`) after the usual
`EF / QPS / Recall` table. The ground truth must be the **filtered** GT:
top-K within the matching attribute subset. For an all-equality layout APR
is an isometry, so original-space L2 GT is valid; for range / mixed layouts
use the rotated-domain GT described above.

### Exact reranking (optional)

`search*` returns at most `k` slots: `ef` is the traversal budget while the
result-pool capacity is always `k`. To let reranking change the top-K set, the
query tool oversamples whenever reranking is enabled:

```bash
./bin/symqg_querying apr.index apr_query.fvecs filtered_gt.ivecs \
    --base-attr base_attr.bin --query-attr query_attr.bin --attr-dim 1 \
    --rerank --orig-base base_raw.fvecs --orig-query query_raw.fvecs \
    --rerank-factor 10
```

* `--rerank` is a valueless switch. Without it nothing changes: no
  oversampling, no extra files and no extra output line.
* `--orig-base` / `--orig-query` must be the **unrotated** fvecs, row-aligned
  with the rotated ones. Both are required with `--rerank`; supplying them (or
  `--rerank-factor`) without the switch is an error.
* `--rerank-factor N` (default 10) sets the candidate count to
  `min(K * N, num_points())`. The tool raises `ef` to at least that count
  because `validate_search` requires `ef >= k`; QPS therefore includes the
  rerank work and is not directly comparable with a non-reranked run.
* The tool retrieves the oversampled matching candidates, computes exact L2
  squared in the **original** space with `euclidean_sqr`, sorts by
  `(distance, id)` for deterministic ties, and writes back exactly `K` results
  (padding with `kPidMax` / `+inf` when short). Output `distances` are then
  original-space L2 squared rather than QG estimated distances.
* Reranking only reorders candidates that already passed the attribute gate,
  so `Predicate violations` stays at 0. An L2 index is required: an IP index
  is rejected because the L2 ordering would not match its metric.
* The reference implementation's plain arm reranks only the final K. PART
  always oversamples when reranking is on, so the top-K set may change; keep
  the factor fixed when comparing runs.

### Fixed fixture

`tests/fixtures/attribute/qg_*` is one fixed sample used by
`tests/unit/rabitqlib/index/qg_attribute_filter_test.cpp`: 256 base vectors
and 16 queries of dimension 64, one attribute with cardinality 4, `K=10`,
degree 32. Base values come from Python's `random.Random(20260707)`; a query
is a randomly chosen base vector plus Gaussian noise (sigma 0.02), with a
target attribute different from that vector's own attribute so that the
unfiltered search demonstrably returns mismatches. The filtered GT is computed
by original-space L2 within the target subset. `qg_base.fvecs` and
`qg_query.fvecs` are produced by `attribute_apr_encode --standalone-seed 1234`;
that splitmix salt is self-contained but **not** comparable with
published-experiment salts. `qg_base_raw.fvecs` and `qg_query_raw.fvecs` are
the row-aligned unrotated counterparts used by reranking; re-encoding them
with `--standalone-seed 1234` reproduces the rotated files byte-for-byte. The
binaries ship with the repository; no Python source or runtime data generation
is needed to run the test.

## IVF integration

`rabitqlib::ivf::IVF` has predicate-filtered overloads of `search` and
`search_batch` alongside its original signatures. Each also has a layout-aware
form (see the hard-filtering section above):

```cpp
ivf.search(query, k, nprobe, ids, dists, base_attr, query_attr, attr_dim);
ivf.search(query, k, nprobe, ids, dists, base_attr, query_attr, attr_dim,
           layout);
ivf.search(query, k, nprobe, ids, dists, use_hacc,
           base_attr, query_attr, attr_dim);
ivf.search_batch(queries, num_queries, k, nprobe, ids, dists,
                 base_attr, query_attrs, attr_dim, use_hacc, num_threads);
ivf.search_batch(queries, num_queries, k, nprobe, ids, dists,
                 base_attr, query_attrs, attr_dim, layout, use_hacc,
                 num_threads);
```

The attribute contract is the SymphonyQG one: a row-major
`uint32 [num_stored_points(), A]` table, `int32` non-negative targets sliced
per query at `qstride` intervals, and no attribute data in the `.index` file.
IVF needs
`num_stored_points()` rather than `num_points()` rows because the gate indexes
by stored ID and `remove()` keeps those IDs addressable. `write_file` / `load`,
`kFormatVersion` and the frozen persistence fixtures are untouched.

The gate is applied at the two `result_pool.insert` points of the two-stage
batch scan: `detail::insert_candidates_filtered` (one-bit-only storage, no
reranking data) and the stage-2 refined-distance loop. Mismatches still take part in
cluster routing and distance estimation; they simply never enter the result
pool. `nprobe` remains the cluster-selection budget and is **not** attribute
aware. Fewer than `k` matches are padded with `kPidMax` / `+inf` by the
existing `copy_results` path.

**The index must be rebuilt in the rotated domain.** APR rotation is an
offline preprocessing step, but IVF is built from the rotated base fvecs,
rotated-domain centroids and rotated-domain cluster assignments. Feeding it
centroids or cluster IDs computed in the original space would make the
residual codes and centroid routing inconsistent, because `construct`
quantizes each point relative to its cluster centroid after applying the
index's own FHT/Kac rotation. That rotation is a second, attribute-independent
transform applied to both base and query; it is orthogonal, so the
rotated-domain assignment is preserved.

`sample/cpp/ivf_rabitq_querying.cpp` keeps its positional arguments
(`<index> <query.fvecs> <gt.ivecs> [hacc]`) and adds the same attribute
switches as QG, plus `--layout`:

```bash
./bin/ivf_rabitq_querying apr.index apr_query.fvecs filtered_gt.ivecs \
    --base-attr base_attr.bin --query-attr query_attr.bin --attr-dim 1
./bin/ivf_rabitq_querying apr.index apr_query.fvecs filtered_gt.ivecs \
    --base-attr base_attr.bin --query-attr query_attr.bin --attr-dim 1 \
    --rerank --orig-base base_raw.fvecs --orig-query query_raw.fvecs \
    --rerank-factor 10
./bin/ivf_rabitq_querying hapr_range.index hapr_range_query.fvecs \
    hapr_range_gt.ivecs --base-attr hapr_range_base_attr.bin \
    --query-attr hapr_range_query_attr.bin --attr-dim 1 --layout range
```

Reranking is again CLI-only. `--rerank` oversamples by passing
`search_k = min(K * factor, num_points())` as `k` to `search_batch`, then
`rerank_by_original_l2` reorders those matching candidates by original-space
L2 squared and truncates back to `K`. Because IVF's result-pool capacity is
`min(k, max_elements_)`, enlarging `k` is the whole oversampling mechanism;
unlike QG there is no `ef` to raise. **`nprobe` is unchanged**, so
oversampling collects more candidates from the same probed clusters; it does
not reach clusters that were never scanned. The nprobe sweep and the timed
measurement loop share one search path, so early stopping uses the same
predicates and rerank policy. QPS includes the rerank work, and the tool
prints `Predicate violations: N` (expected `0`) and one `Rerank: ...` line.

Two distinct reranking stages must not be confused: IVF's built-in stage 2
(`total_bits == 32` uses `euclidean_sqr` on the *rotated* stored vector;
extra-bit storage uses `split_distance_boosting`) corrects quantization error
and is left unchanged, while the CLI rerank uses the *unrotated* original
vectors.

`tests/fixtures/attribute/ivf_centroids.fvecs` and `ivf_cluster_ids.ivecs`
are fixed, generated once in the rotated domain from `qg_base.fvecs`, and
used by `tests/unit/rabitqlib/index/ivf_attribute_filter_test.cpp` to build
in-memory indexes at 1, 5 and 32 bits, covering both gate insertion points.
The filtered GT is reused unchanged because a full attribute match makes the
rotation an isometry. The reference IVF baseline only wires up offline
rotated fvecs and leaves its search sources untouched; the hard gate,
oversampling and predicate statistics here follow PART's QG protocol instead.
