#!/usr/bin/env bash
# ============================================================
# run_ivf.sh — IVF build flow for RaBitQ-arp (reference: faiss/rabitq-inline/post):
#   prepare -> build -> encode(apr) -> cluster -> index -> verify(smoke)
#
# Consistent with the reference libraries: read config.cfg, env-var overrides (grid injection takes
# priority), taskset affinity, clean-env.sh build, and organizing by prepare/encode/index/search stages.
# Encoding keeps APR unchanged; index building uses the C++ ivf_rabitq_indexing (compiled as a C++ binary
# like symqg, bypassing the Python rabitqlib bindings); this round does not change search sources, and
# verify only does a pure-vector smoke test.
#
# Usage:
#   bash run_ivf.sh                       # full pipeline
#   bash run_ivf.sh prepare build encode cluster index verify
#   bash run_ivf.sh prepare                # single stage
# Env-var overrides are also supported (see DEFAULTS below): DATA_SOURCE / DATASET / D / ATTR_DIM /
#   ATTR_CARD / IVF_CLUSTERS / IVF_BITS / VEC_BASE / VEC_QUERY / ORIG_DATASET_DIR /
#   INDEX_DIR / LOG_DIR / NPROBE_CSV ...
# ============================================================
set -euo pipefail

# ---- Locate the directory containing this script (works from any invocation path) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.cfg"

# ---- Snapshot env-var overrides (must be done before reading config.cfg, otherwise config would override them) ----
declare -A _ENV_OVERRIDE=()
for _k in DATA_SOURCE DATASET D N NQ VEC_BASE VEC_QUERY ORIG_DATASET_DIR \
          INDEX_DIR LOG_DIR ATTR_DIM ATTR_CARD IVF_CLUSTERS IVF_BITS \
          NPROBE_CSV PY BUILD_THREADS SEARCH_THREADS RABITQLIB_HOME METRIC \
          ENC_SEED FORCE_REBUILD; do
  if [[ -n "${!_k:-}" ]]; then _ENV_OVERRIDE[$_k]="${!_k}"; fi
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# ---- Read config (only load KEY=VALUE lines; skip comments/blank lines; trim whitespace and quotes) ----
while IFS='=' read -r key value; do
  key="$(echo "$key" | xargs)"
  value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
  [[ -z "$key" || "$key" == \#* || -z "$value" ]] && continue
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
  declare -g "$key=$value"
done < "$CONFIG_FILE"

# ---- Apply env-var overrides (after reading config, ensuring defaults/config values are overridden) ----
for _k in "${!_ENV_OVERRIDE[@]}"; do declare -g "$_k=${_ENV_OVERRIDE[$_k]}"; done

# ---- DEFAULTS (small sift A1_C2 dataset; overridable via env vars) ----
PY="${PY:-/home/yq/miniconda3/envs/faiss_gpu/bin/python}"
BUILD_THREADS="${BUILD_THREADS:-32}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
BUILD_AFFINITY="0-$((BUILD_THREADS-1))"
SEARCH_AFFINITY="0-$((SEARCH_THREADS-1))"
# Critical: RABITQLIB_HOME must point to arp's own root so its IVF samples compile;
# only honored when the user sets it explicitly via env var (the legacy Graph value in config.cfg is ignored).
if [[ -n "${_ENV_OVERRIDE[RABITQLIB_HOME]:-}" ]]; then
  RABITQLIB_HOME="${_ENV_OVERRIDE[RABITQLIB_HOME]}"
else
  RABITQLIB_HOME="$SCRIPT_DIR"
fi

# clean-env.sh may not live under /home/yq/ (migrated to _old); resolve a usable path, fall back to direct invocation if missing
CLEAN_ENV=""
for _ce in /home/yq/clean-env.sh /home/yq/_old/clean-env.sh; do
  if [[ -x "$_ce" ]]; then CLEAN_ENV="bash $_ce"; break; fi
done
METRIC="${METRIC:-l2}"
ENC_SEED="${ENC_SEED:-1234}"
ATTR_DIM="${ATTR_DIM:-1}"
ATTR_CARD="${ATTR_CARD:-2}"
IVF_CLUSTERS="${IVF_CLUSTERS:-1024}"
IVF_BITS="${IVF_BITS:-8}"
NPROBE_CSV="${NPROBE_CSV:-5,10,20,40,80}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"

DATASET="${DATASET:-sift}"
DATA_SOURCE="${DATA_SOURCE:-/home/yq/SIGMOD2027/_dataset/sift/A1_C2}"
VEC_BASE="${VEC_BASE:-$DATA_SOURCE/${DATASET}_base.fvecs}"
VEC_QUERY="${VEC_QUERY:-$DATA_SOURCE/${DATASET}_query.fvecs}"
ORIG_DATASET_DIR="${ORIG_DATASET_DIR:-$DATA_SOURCE}"
D="${D:-128}"
INDEX_DIR="${INDEX_DIR:-/home/yq/SIGMOD2027/_index/RaBitQ-arp-ivf}"
LOG_DIR="${LOG_DIR:-/home/yq/SIGMOD2027/_log}"

ATTR_SUFFIX="_A${ATTR_DIM}_C${ATTR_CARD}"

# ---- Derived paths ----
BASE_FVECS="$DATA_SOURCE/${DATASET}_base.fvecs"
QUERY_FVECS="$DATA_SOURCE/${DATASET}_query.fvecs"
BASE_ATTR="$DATA_SOURCE/base_attr${ATTR_SUFFIX}.bin"
QUERY_ATTR="$DATA_SOURCE/query_attr${ATTR_SUFFIX}.bin"

APR_DIR="$DATA_SOURCE/apr_rotated${ATTR_SUFFIX}"
APR_BASE="$APR_DIR/${DATASET}_base.fvecs"
APR_QUERY="$APR_DIR/${DATASET}_query.fvecs"
# pure-vector GT (computed on the fly by faiss in verify, avoiding dependency on prepare's pure-GT symlink)
GT_PURE="$APR_DIR/${DATASET}_groundtruth_pure.ivecs"

BUILD_DIR="$SCRIPT_DIR/.build_ivf"
# Note: arp's CMakeLists sets the runtime output dir to ${PROJECT_SOURCE_DIR}/bin
INDEX_BIN="$SCRIPT_DIR/bin/ivf_rabitq_indexing"
QUERY_BIN="$SCRIPT_DIR/bin/ivf_rabitq_querying"

INDEX_FILE="$INDEX_DIR/ivf_${DATASET}${ATTR_SUFFIX}_b${IVF_BITS}.index"
CLUSTERS_NPZ="$APR_DIR/clusters_${IVF_CLUSTERS}.npz"
CENTROIDS_FVECS="$APR_DIR/centroids_${IVF_CLUSTERS}.fvecs"
CIDS_IVECS="$APR_DIR/cids_${IVF_CLUSTERS}.ivecs"

echo "[info] DATA_SOURCE=$DATA_SOURCE  DATASET=$DATASET  D=$D  A=$ATTR_DIM C=$ATTR_CARD"
echo "[info] RABITQLIB_HOME=$RABITQLIB_HOME  METRIC=$METRIC  IVF_CLUSTERS=$IVF_CLUSTERS  IVF_BITS=$IVF_BITS"
echo "[info] BUILD_THREADS=$BUILD_THREADS (affinity=$BUILD_AFFINITY)  SEARCH_THREADS=$SEARCH_THREADS (affinity=$SEARCH_AFFINITY)"
echo "[info] INDEX_FILE=$INDEX_FILE"

# =====================================================================
# Stage 1: prepare — invoke data/prepare.py to symlink vectors/attributes/GT into DATA_SOURCE
# =====================================================================
run_prepare() {
  echo "========== [1/6] prepare =========="
  mkdir -p "$DATA_SOURCE"
  env -u PYTHONPATH "$PY" "$RABITQLIB_HOME/data/prepare.py" \
    --vec-base "$VEC_BASE" \
    --vec-query "$VEC_QUERY" \
    --orig-dataset "$ORIG_DATASET_DIR" \
    --out-dir "$DATA_SOURCE" \
    --dataset "$DATASET" \
    --attr-dim "$ATTR_DIM" \
    --attr-card "$ATTR_CARD"
  echo "[prepare] done -> $DATA_SOURCE"
}

# =====================================================================
# Stage 2: build — out-of-source cmake + compile ivf_rabitq_indexing / ivf_rabitq_querying
#   (same structure as arp's run.sh build; RABITQLIB_HOME is forced to arp root,
#    a separate .build_ivf build dir is used, never touching the symqg_* binaries or run.sh)
# =====================================================================
run_build() {
  echo "========== [2/6] build (ivf rabitq) =========="
  mkdir -p "$BUILD_DIR"
  $CLEAN_ENV cmake -S "$RABITQLIB_HOME" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRABITQ_BUILD_SAMPLES=ON \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=ON
  echo "[build] cmake configured."
  $CLEAN_ENV taskset -c "$BUILD_AFFINITY" cmake --build "$BUILD_DIR" \
    --target ivf_rabitq_indexing ivf_rabitq_querying --parallel "$BUILD_THREADS"
  if [[ ! -x "$INDEX_BIN" || ! -x "$QUERY_BIN" ]]; then
    echo "[error] binaries not found after build: $INDEX_BIN or $QUERY_BIN" >&2
    exit 1
  fi
  echo "[build] done. binaries: $INDEX_BIN $QUERY_BIN"
}

# =====================================================================
# Stage 3: encode — APR rotation (reuse the script/encoding logic; encoding semantics unchanged)
# =====================================================================
run_encode() {
  echo "========== [3/6] encode (apr) =========="
  mkdir -p "$APR_DIR"
  ( cd "$SCRIPT_DIR/scripts" && taskset -c "$BUILD_AFFINITY" env -u PYTHONPATH \
      OMP_NUM_THREADS="$BUILD_THREADS" OPENBLAS_NUM_THREADS="$BUILD_THREADS" \
      MKL_NUM_THREADS="$BUILD_THREADS" NUMEXPR_NUM_THREADS="$BUILD_THREADS" \
      "$PY" encode_attr_apr.py \
      --ds "$DATA_SOURCE" \
      --out "$APR_DIR" \
      --d "$D" \
      --A "$ATTR_DIM" \
      --enc-seed "$ENC_SEED" \
      --base-name "${DATASET}_base.fvecs" \
      --query-name "${DATASET}_query.fvecs" \
      --base-attr-name "base_attr${ATTR_SUFFIX}.bin" \
      --query-attr-name "query_attr${ATTR_SUFFIX}.bin" \
      --groundtruth-name "${DATASET}_groundtruth${ATTR_SUFFIX}.ivecs" \
      --apr-base-name "${DATASET}_base.fvecs" \
      --apr-query-name "${DATASET}_query.fvecs" )
  echo "[encode] done -> $APR_BASE / $APR_QUERY"
}

# =====================================================================
# Stage 4: cluster — faiss clustering, producing clusters.npz + centroids.fvecs + cids.ivecs
#   (extended faiss_clustering.py; aligned with the C++ ivf_rabitq_indexing inputs)
# =====================================================================
run_cluster() {
  echo "========== [4/6] cluster (faiss) =========="
  ( cd "$SCRIPT_DIR/scripts" && taskset -c "$BUILD_AFFINITY" env -u PYTHONPATH \
      OMP_NUM_THREADS="$BUILD_THREADS" OPENBLAS_NUM_THREADS="$BUILD_THREADS" \
      MKL_NUM_THREADS="$BUILD_THREADS" NUMEXPR_NUM_THREADS="$BUILD_THREADS" \
      "$PY" faiss_clustering.py "$APR_BASE" "$CLUSTERS_NPZ" \
      --num-clusters "$IVF_CLUSTERS" \
      --metric "$METRIC" \
      --num-threads "$BUILD_THREADS" \
      --centroids-out "$CENTROIDS_FVECS" \
      --cids-out "$CIDS_IVECS" )
  echo "[cluster] done -> $CENTROIDS_FVECS / $CIDS_IVECS"
}

# =====================================================================
# Stage 5: index — C++ ivf_rabitq_indexing builds the IVF index
#   argv: <data.fvecs> <centroids.fvecs> <cids.ivecs> <total_bits> <index> [metric] [faster_quant]
# =====================================================================
run_index() {
  echo "========== [5/6] index (ivf_rabitq_indexing) =========="
  [[ -x "$INDEX_BIN" ]] || { echo "[error] $INDEX_BIN not found, run build first" >&2; exit 1; }
  [[ -f "$APR_BASE" ]]  || { echo "[error] $APR_BASE not found, run encode first" >&2; exit 1; }
  [[ -f "$CENTROIDS_FVECS" && -f "$CIDS_IVECS" ]] || { echo "[error] clustering outputs not found, run cluster first" >&2; exit 1; }
  if [[ -f "$INDEX_FILE" && "$FORCE_REBUILD" -eq 0 ]]; then
    echo "[index] index exists, skip build (use -f or FORCE_REBUILD=1 to force rebuild): $INDEX_FILE"
  else
    if [[ "$FORCE_REBUILD" -eq 1 && -f "$INDEX_FILE" ]]; then
      echo "[index] FORCE_REBUILD set, rebuilding index"
    fi
    mkdir -p "$INDEX_DIR"
    $CLEAN_ENV taskset -c "$BUILD_AFFINITY" "$INDEX_BIN" \
      "$APR_BASE" "$CENTROIDS_FVECS" "$CIDS_IVECS" "$IVF_BITS" "$INDEX_FILE" "$METRIC" false
    echo "[index] build complete: $INDEX_FILE"
  fi
}

# =====================================================================
# Stage 6: verify — pure-vector smoke test (search sources unchanged)
#   compute the pure-vector GT on the fly with faiss, then run the existing ivf_rabitq_querying
#   scanning nprobe to report recall
# =====================================================================
gen_pure_gt() {
  local out="$1"
  echo "[verify] generate pure-vector GT (faiss exact knn) -> $out"
  ( cd "$SCRIPT_DIR/scripts" && taskset -c "$BUILD_AFFINITY" env -u PYTHONPATH \
      OMP_NUM_THREADS="$BUILD_THREADS" \
      "$PY" -c "
import numpy as np, faiss
from utils import read_fvecs
base = read_fvecs('$APR_BASE').astype(np.float32)
q = read_fvecs('$APR_QUERY').astype(np.float32)
_, I = faiss.knn(q, base, 100, faiss.METRIC_L2)
I = I.astype(np.uint32)
with open('$out', 'wb') as f:
    for row in I:
        f.write(np.int32(100).tobytes()); f.write(row.tobytes())
print('[verify] GT rows', I.shape)
" )
}

run_verify() {
  echo "========== [6/6] verify (smoke, pure-vector) =========="
  [[ -x "$QUERY_BIN" ]]  || { echo "[error] $QUERY_BIN not found, run build first" >&2; exit 1; }
  [[ -f "$APR_QUERY" ]]  || { echo "[error] $APR_QUERY not found, run encode first" >&2; exit 1; }
  [[ -f "$INDEX_FILE" ]] || { echo "[error] index $INDEX_FILE not found, run index first" >&2; exit 1; }
  mkdir -p "$LOG_DIR"
  if [[ ! -s "$GT_PURE" ]]; then gen_pure_gt "$GT_PURE"; fi
  # the existing generic ivf_rabitq_querying: internally scans nprobe and prints the nprobe/QPS/recall table
  $CLEAN_ENV taskset -c "$SEARCH_AFFINITY" env \
    OMP_NUM_THREADS="$SEARCH_THREADS" OPENBLAS_NUM_THREADS="$SEARCH_THREADS" \
    MKL_NUM_THREADS="$SEARCH_THREADS" \
    "$QUERY_BIN" "$INDEX_FILE" "$APR_QUERY" "$GT_PURE" 2>&1 | tee "$LOG_DIR/ivf_${DATASET}${ATTR_SUFFIX}_b${IVF_BITS}_smoke.log"
  echo "[verify] smoke test complete -> $LOG_DIR/ivf_${DATASET}${ATTR_SUFFIX}_b${IVF_BITS}_smoke.log"
}

# ---- Main flow ----
if [[ $# -eq 0 ]]; then
  run_prepare
  run_build
  run_encode
  run_cluster
  run_index
  run_verify
  echo "[all done] IVF build flow complete."
  exit 0
fi

for stage in "$@"; do
  case "$stage" in
    prepare) run_prepare ;;
    build)   run_build   ;;
    encode)  run_encode  ;;
    cluster) run_cluster ;;
    index)   run_index   ;;
    verify)  run_verify  ;;
    *)
      echo "[warn] unknown stage: $stage (valid: prepare / build / encode / cluster / index / verify)" >&2
      ;;
  esac
done

echo "[all done] requested stages complete."
