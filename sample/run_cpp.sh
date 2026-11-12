#!/usr/bin/env bash

# Resolve data, build, and example paths from the repository root.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$repo_root" || exit 1

# compiling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Download the dataset
mkdir -p ./data/gist
if [ ! -f ./data/gist/gist.tar.gz ]; then
    wget -P ./data/gist ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz
fi
tar -xzvf ./data/gist/gist.tar.gz -C ./data

# indexing and querying for symqg
./bin/symqg_indexing ./data/gist/gist_base.fvecs 32 400 ./data/gist/symqg_32.index

./bin/symqg_querying ./data/gist/symqg_32.index ./data/gist/gist_query.fvecs ./data/gist/gist_groundtruth.ivecs

# One-shot PART IVF pipeline on the bundled attribute fixture: rotate the
# vectors, train centroids in-process and save the index, then evaluate it with
# the predicate-filtered query CLI. No clustering intermediate files needed.
fixture=./tests/fixtures/attribute
./bin/ivf_part_pipeline \
    --base "$fixture/qg_base_raw.fvecs" \
    --query "$fixture/qg_query_raw.fvecs" \
    --base-attr "$fixture/qg_base_attr.bin" \
    --query-attr "$fixture/qg_query_attr.bin" \
    --dim 64 --attr-dim 1 --standalone-seed 1234 \
    --num-clusters 8 --bits 8 --threads 4 \
    --gt "$fixture/qg_filtered_gt.ivecs" --nprobe 8 --topk 10 \
    --out-dir ./data/part_ivf

./bin/ivf_rabitq_querying ./data/part_ivf/part_ivf.index \
    ./data/part_ivf/part_query.fvecs "$fixture/qg_filtered_gt.ivecs" \
    --base-attr "$fixture/qg_base_attr.bin" \
    --query-attr "$fixture/qg_query_attr.bin" --attr-dim 1


