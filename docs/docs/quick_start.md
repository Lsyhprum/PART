# Quick start

Build and query a PART IVF-RaBitQ index from C++ using the attribute fixture
shipped with this repository, so no dataset download is required.

## Build the library

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DRABITQ_BUILD_SAMPLES=ON
cmake --build build --parallel
```

`RABITQ_BUILD_INDEX`, `RABITQ_BUILD_FASTSCAN` and `RABITQ_USE_OPENMP` all
default to `ON`. The sample command-line tools are written to `bin/` in the
source tree (not `build/bin/`). For the full option list see
[C++ integration](cpp_integration.md) and [Installation](installation.md).

## Build and search an index

`ivf_part_pipeline` performs attribute rotation, centroid training, point
routing and index saving in one process:

```bash
fixture=./tests/fixtures/attribute

./bin/ivf_part_pipeline \
  --base "$fixture/qg_base_raw.fvecs" \
  --query "$fixture/qg_query_raw.fvecs" \
  --base-attr "$fixture/qg_base_attr.bin" \
  --query-attr "$fixture/qg_query_attr.bin" \
  --dim 64 --attr-dim 1 --standalone-seed 1234 \
  --num-clusters 8 --bits 8 --out-dir ./data/part_ivf \
  --gt "$fixture/qg_filtered_gt.ivecs" --nprobe 8 --topk 10

./bin/ivf_rabitq_querying ./data/part_ivf/part_ivf.index \
  ./data/part_ivf/part_query.fvecs "$fixture/qg_filtered_gt.ivecs" \
  --base-attr "$fixture/qg_base_attr.bin" \
  --query-attr "$fixture/qg_query_attr.bin" --attr-dim 1
```

`--gt` adds an in-process filtered-search smoke test (recall and predicate
violations); the querying CLI prints the full `nprobe` / QPS / recall sweep.
`sample/run_cpp.sh` runs the same commands. See the
[README](../../README.md) for CPR, HAPR and reranking options.

## Save and reload

`ivf_part_pipeline` writes a self-describing index file that
`ivf_rabitq_querying` reloads directly; no separate serialization step is
needed for the CLI workflow. The C++ API behind it is `rabitqlib::ivf::IVF`
(`rabitqlib/index/ivf/ivf.hpp`): construct it without a point count, call
`train(...)` and `add(...)`, then `save(...)`. A search process calls
`load(...)` before `search(...)` / `search_batch(...)`. See
[C++ integration](cpp_integration.md) for build setup and the exact signatures.
