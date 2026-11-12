# PART: Baking Attributes into High-Dimensional Vectors for Filtered Approximate Nearest Neighbor Search

PART encodes attribute predicates into high-dimensional vectors, thereby supporting filtered approximate nearest-neighbor search without a distinct attribute index. See the [research paper](https://github.com/Lsyhprum/PART/blob/main/technical_report.pdf) for further details.

## Why PART

- Attribute-aware rotation. PART rotates each vector by an attribute-dependent phase such that matching predicates preserve distance exactly while mismatches are pushed apart. The predicate check acts as a gate within the existing ANN path.
- Versatile support for diverse predicates. APR, CPR, SPR and HAPR can be deployed to handle equality, range, and mixed predicates.
- Built on mainstream ANNS indexes. It seamlessly integrates with prevailing ANNS indexes such as IVF-RaBitQ and SymphonyQG. It inherits the theoretical error bound and fast construction and query performance of the underlying indexes.

## Quick start

Configure and build the core, index, and the C++ sample tools:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DRABITQ_BUILD_SAMPLES=ON \
  -DRABITQ_BUILD_INDEX=ON \
  -DRABITQ_BUILD_FASTSCAN=ON \
  -DRABITQ_USE_OPENMP=ON
cmake --build build --parallel
```

The CLI tools are produced under `bin/` in the source tree:
`ivf_part_pipeline`, `attribute_apr_encode`, `attribute_cpr_encode`,
`attribute_spr_encode`, `ivf_rabitq_indexing`, `ivf_rabitq_querying`, and the
SymphonyQG equivalent.


```bash
# Build: rotate + train + route + save, with an optional filtered-search smoke test.
./bin/ivf_part_pipeline \
  --base <base.fvecs> --query <query.fvecs> \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 --standalone-seed 1234 \
  --num-clusters 4096 --bits 8 --metric l2 --threads 32 \
  --out-dir data/part_ivf \
  --gt GT.ivecs --nprobe 32 --topk 100

# Search: full nprobe/QPS/recall sweep, predicate gate and optional rerank.
./bin/ivf_rabitq_querying data/part_ivf/part_ivf.index \
  data/part_ivf/part_query.fvecs GT.ivecs \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> --attr-dim 2
```

Use `--encoder cpr` for CPR, `--encoder spr` for pure range, or
`--layout eq,range --range-l L` (equivalently `--encoder hapr`) for hybrid
HAPR. `--layout` / `--range-l` apply to `apr`, `hapr` and `spr`; `--encoder apr`
defaults to all-equality unless `--layout` is given. `--standalone-seed`, `--index-seed`
and `--cluster-seed` independently control the attribute salt, the IVF rotator
and the k-means training. The same two commands against the bundled fixture
are in the IVF branch of `sample/run_cpp.sh`.

### Predicate layouts and encoders

PART provides four offline encoders, selected by `--encoder apr|cpr|hapr|spr`
(default `apr`) in `ivf_part_pipeline`, or by the matching `attribute_*_encode`
CLI. All four rotate the vectors in the same way; they differ only in how the
attribute phases are assigned.

- **APR** — equality / multi-attribute AND. By default every column is an equality
  column (`--layout` omitted). An isometric phase is assigned per attribute value,
  so the rotation preserves distance and recall is exact without reranking. Range
  columns can still be mixed in via `--layout`, which activates HAPR below.
- **SPR** — pure continuous numeric **range** (Scaled Phase Rotation, the paper's
  Algorithm III). Selected with `--encoder spr` (standalone CLI
  `attribute_spr_encode`); it has no `--layout` because every column is a range
  column. A soft interval-centre phase is used; recall has a soft ceiling
  (~0.95–0.96) that the rerank step restores.
- **HAPR** — **hybrid** equality + range. Selected with `--encoder hapr`, or
  equivalently with `--encoder apr --layout eq,range`. Exact phase for `eq`
  columns, soft phase for `range` columns.
- **CPR** — categorical disjoint-block equality. A separate CLI with the same
  arguments (no `--layout`; every column is an equality column). Unlike APR, the
  `attr_dim` attributes own **disjoint** blocks of dimension pairs, so each pair
  is rotated by exactly one attribute.

Range length `range_l` (the number of distinct range values) is auto-inferred
from `base_attr` when `--range-l 0` (the default) for `spr` / `hapr` / `apr --layout`.

**CPR example** (same arguments as APR, no `--layout`):

```bash
./bin/attribute_cpr_encode \
  --base <base.fvecs> --query <query.fvecs> \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 \
  --standalone-seed 1234 \
  --out-dir data/cpr
```

It writes `cpr_base.fvecs` / `cpr_query.fvecs`. `ivf_part_pipeline` selects the
encoder with `--encoder apr|cpr|hapr|spr` (default `apr`). For SymphonyQG graph
indexes, use the standalone encoders together with the build/query commands in
`sample/run_cpp.sh`.

**SPR example** (pure range; `--range-l` required or auto-inferred):

```bash
./bin/attribute_spr_encode \
  --base <base.fvecs> --query <query.fvecs> \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 \
  --range-l 100 \
  --standalone-seed 1234 \
  --out-dir data/spr
```

It writes `spr_base.fvecs` / `spr_query.fvecs` and, like HAPR, a plain-text
`hapr_meta.txt` recording the all-range layout and `range_l`.

---

## Datasets and reproduction

Dataset preparation and preprocessing scripts live under `datasets/`. The
end-to-end workflow (attribute encoding, index build, and filtered search) is
covered in the Quick Start above.

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

## Acknowledgements

The quantization, indexing, and search infrastructure are taken from [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library). The attribute-rotation layer (`rabitqlib::attribute`, covering APR, SPR, HAPR and CPR) and the filtered-search workflow are our additions. Many thanks to
the RaBitQ authors for the foundation this project extends. Please cite RaBitQ-Library where appropriate.

## License

PART is released under the [Apache License 2.0](LICENSE).
