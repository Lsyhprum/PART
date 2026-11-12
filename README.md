# PART: Baking Attributes into High-Dimensional Vectors for Filtered Approximate Nearest Neighbor Search

PART encodes attribute predicates into high-dimensional vectors, thereby supporting filtered approximate nearest-neighbor search without a distinct attribute index. See the [research paper](https://github.com/Lsyhprum/PART/blob/main/technical_report.pdf) for further details.

## Why PART

- Attribute-aware rotation. PART rotates each vector by an attribute-dependent phase such that matching predicates preserve distance exactly while mismatches are pushed apart. The predicate check acts as a gate within the existing ANN path.
- Versatile support for diverse predicates. APR, CPR, SPR and HAPR can be deployed to handle equality, range, and mixed predicates.
- Built on mainstream ANNS indexes. It seamlessly integrates with prevailing ANNS indexes such as IVF-RaBitQ and SymphonyQG. It inherits the theoretical error bound and fast construction and query performance of the underlying indexes.

## Quick Start

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

Compiled CLI binaries are available under `bin/`.


```bash
# Build: rotate + train + route + save, with an optional filtered-search validation
./bin/ivf_part_pipeline \
  --base <base.fvecs> --query <query.fvecs> \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 --standalone-seed 1234 \
  --num-clusters 4096 --bits 8 --metric l2 --threads 32 \
  --out-dir data/part_ivf \
  --gt GT.ivecs --nprobe 32 --topk 100

# Search: evaluate recall / QPS with predicate filtering and optional rerank
./bin/ivf_rabitq_querying data/part_ivf/part_ivf.index \
  data/part_ivf/part_query.fvecs GT.ivecs \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> --attr-dim 2
```

To use other predicate encoders, configure `--encoder` and related flags described in **Predicate Encoders & Configurations**.
Preset end-to-end scripts are provided in `sample/run_cpp.sh``.

## Predicate Encoders & Configurations

PART provides four offline encoders, selected by `--encoder apr|cpr|hapr|spr` (default `apr`) in `ivf_part_pipeline`, or by the matching `attribute_*_encode` CLI. All four rotate the vectors in the same way; they differ only in how the attribute phases are assigned.

- **APR** — Equality / multi-attribute AND. Defaults to all-equality columns when `--layout` is omitted. An isometric phase is assigned per attribute value, preserving distances and exact recall without reranking. Mixed equality+range predicates can be enabled with `--layout eq,range`, which activates HAPR behavior.

- **SPR** — Pure continuous numeric range. Every attribute column is treated as a range; no `--layout` flag. It uses a soft interval-centre phase.

- **HAPR** — Hybrid equality + range. Equivalent to `--encoder apr --layout eq,range`. Exact phase for equality columns, soft phase for range columns.

- **CPR** — Categorical disjoint-block equality. No `--layout` flag. Each attribute occupies disjoint pairs of dimensions, so each dimension pair is rotated by exactly one attribute.

The flag `--range-l` specifies the number of distinct range values. Set `--range-l 0` (default) to auto-infer this value from `base_attr` for SPR / HAPR / APR with range layout.

Three independent random seeds are available:

- `--standalone-seed`: attribute rotation salt
- `--index-seed`: IVF rotator seed
- `--cluster-seed`: k-means clustering seed

CPR example:

```bash
./bin/attribute_cpr_encode \
  --base <base.fvecs> --query <query.fvecs> \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 \
  --standalone-seed 1234 \
  --out-dir data/cpr
```

SPR example:

```bash
./bin/attribute_spr_encode \
  --base <base.fvecs> --query <query.fvecs> \
  --base-attr <base_attr.bin> --query-attr <query_attr.bin> \
  --dim 128 --attr-dim 2 \
  --range-l 100 \
  --standalone-seed 1234 \
  --out-dir data/spr
```

## Datasets and Reproduction

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

## Acknowledgements

The quantization, indexing, and search infrastructure are taken from [RaBitQ-Library](https://github.com/VectorDB-NTU/RaBitQ-Library). The attribute-rotation layer (`rabitqlib::attribute`, covering APR, SPR, HAPR and CPR) and the filtered-search workflow are our additions. Many thanks to
the RaBitQ authors for the foundation this project extends. Please cite RaBitQ-Library where appropriate.

## License

PART is released under the [Apache License 2.0](LICENSE).
