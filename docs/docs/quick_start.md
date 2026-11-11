# Quick start

Build, search, save and reload a small IVF-RaBitQ index from C++. This guide
describes **0.8.3** and uses synthetic data, so no dataset download is required.
For the full API reference and build options, see
[C++ integration](cpp_integration.md). For platform details, see
[Installation](installation.md).

## Build the library

From the repository root:

```bash
cmake -S . -B build -DRABITQ_BUILD_INDEX=ON -DRABITQ_BUILD_SAMPLES=ON \
    -DCMAKE_BUILD_TYPE=Release -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF
cmake --build build --parallel
```

This builds the library and the `sample/cpp` command-line tools under `build/bin`
(or `bin/`).

## Build and search an IVF-RaBitQ index (CLI)

The C++ samples reproduce the paper's pipeline without Python. For an
attribute-aware (APR/HAPR) index, the workflow is:

```bash
# Learn attributes from the data, then encode them into the rotated space.
./build/bin/attribute_apr_encode data.fvecs attr.fvecs rotated.fvecs 5 16

# Build the IVF-RaBitQ index from the rotated (attribute-aware) vectors.
./build/bin/ivf_rabitq_indexing rotated.fvecs index.bin 5 16

# Search with an attribute predicate; only rows matching the predicate are returned.
./build/bin/ivf_rabitq_querying index.bin queries.fvecs predicates.txt 10 5
```

`attribute_apr_encode` rotates the data so that the attribute predicate is
enforced during search (HAPR); `ivf_rabitq_indexing` builds the quantizer and
clusters, and `ivf_rabitq_querying` performs hard-filtered search. See
[C++ integration](cpp_integration.md) for the exact arguments and the in-memory
C++ API (`rabitqlib::index::IvfIndex`).

## Save and reload

`ivf_rabitq_indexing` writes a self-describing index file that
`ivf_rabitq_querying` reloads directly; no separate serialization step is
needed for the CLI workflow. The in-memory C++ API exposes `save`/`load`
for embedding the index in your own service.
