#!/usr/bin/env bash
# ============================================================
# run.sh — reads config.cfg and runs the full RaBitQ-Library-inline-rot (symqg) baseline pipeline:
#   prepare -> build -> index -> search
# The enhanced branch additionally runs the encode_range stage first (produces range bin / range GT / rotated dataset).
#
# Conventions aligned with SIGMOD2027/baseline's RaBitQ-Library-post / RaBitQ-inline / NHQ / ACORN:
#   - parse config.cfg (only load KEY=VALUE lines; skip comments/blank lines; trim surrounding whitespace and quotes)
#   - resolve relative paths against SCRIPT_DIR (all paths here are absolute and kept as-is)
#   - derive names from ATTR_DIM/ATTR_CARD (e.g. base_attr_A*_C*.bin)
#   - inject thread/library paths (controlled by the scheduler layer; algorithm sources are untouched)
#   - unified LOG_DIR logging (clears previous [RESULT] lines before each search)
#
# Four retrieval modes (auto-selected from config):
#   pure           : both ATTR_BASE/ATTR_QUERY empty
#   inline         : both ATTR_BASE/ATTR_QUERY non-empty and ATTR_RANGE=0  (in-graph equality filtering)
#   range_hardgate : both ATTR_BASE/ATTR_QUERY non-empty and ATTR_RANGE=1  (-r kernel hard gate)
#   enhanced       : RANGE_QUERY=on  (RoPE rotation + M-subcenter union + original-space rerank gate)
#
# Usage:
#   bash run.sh                        # prepare -> build -> index -> search (per current config mode)
#   bash run.sh prepare                # only symlink data (vectors/GT/attributes -> DATA_SOURCE)
#   bash run.sh build                  # only cmake-build symqg_indexing / symqg_querying
#   bash run.sh index                  # only build the symqg graph index (skip if exists unless -f)
#   bash run.sh encode_range           # only produce range bin/GT and (for enhanced) the rotated dataset
#   bash run.sh search                 # only run search and emit [RESULT] logs
#   bash run.sh -f                     # force index rebuild (even if the index file already exists)
#   also combinable: bash run.sh prepare build index encode_range search
# ============================================================
set -euo pipefail

# ---- Locate the directory containing this script (works from any invocation path) ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.cfg"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] config file not found: $CONFIG_FILE" >&2
  exit 1
fi

# ---- Environment-variable override layer (used by sweep batch drivers) ----
# Note: the env-var overrides must be snapshotted BEFORE reading config.cfg — otherwise config's
# declare -g would overwrite the env values, so later ${!_k} would read the config value instead of the export.
# Mode switches (ATTR_BASE/ATTR_QUERY/ATTR_RANGE/RANGE_QUERY, etc.) are excluded here, to keep inline fixed.
OVERRIDE_KEYS=(DATASET D N NQ K \
               ATTR_DIM ATTR_CARD \
               VEC_BASE VEC_QUERY ORIG_DATASET_DIR ORIG_DATASET \
               DATA_SOURCE INDEX_DIR RABITQLIB_HOME LOG_DIR GT_SUFFIX \
               EF_SEARCH ROUNDS BUILD_THREADS SEARCH_THREADS \
               EF_CONSTRUCTION MAX_DEGREE METRIC QUANTIZATION_BITS INIT)
declare -A _ENV_OVERRIDE=()
for _k in "${OVERRIDE_KEYS[@]}"; do
  _v="$(printenv "$_k" 2>/dev/null || true)"
  if [[ -n "$_v" ]]; then
    _ENV_OVERRIDE["$_k"]="$_v"
  fi
done

# ---- Read config (only load KEY=VALUE lines; skip comments/blank lines; trim whitespace and quotes) ----
while IFS='=' read -r key value; do
  key="$(echo "$key" | xargs)"
  value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
  [[ -z "$key" || "$key" == \#* || -z "$value" ]] && continue
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
  declare -g "$key=$value"
done < "$CONFIG_FILE"

# ---- Apply environment-variable overrides (used by sweep batch drivers; no effect on single runs) ----
# The override values come from env vars snapshotted before reading config (see _ENV_OVERRIDE above).
for _k in "${OVERRIDE_KEYS[@]}"; do
  _e="${_ENV_OVERRIDE[$_k]:-}"
  if [[ -n "$_e" ]]; then
    declare -g "$_k=$_e"
  fi
done
# ORIG_DATASET defaults to the same source as DATASET (after prepare, filenames under DATA_SOURCE share the same stem)
ORIG_DATASET="${ORIG_DATASET:-$DATASET}"

# ---- Anchor possibly-relative paths to SCRIPT_DIR ----
for _p in DATA_SOURCE INDEX_DIR RABITQLIB_HOME LOG_DIR VEC_BASE VEC_QUERY ORIG_DATASET_DIR; do
  _v="${!_p}"
  case "$_v" in
    /*) ;;
    *)  _v="$SCRIPT_DIR/$_v" ;;
  esac
  declare -g "$_p=$_v"
done

# ---- Validate required parameters (ATTR_BASE/ATTR_QUERY/ATTR_RANGE/RANGE_QUERY are optional switches, not enforced) ----
for v in DATA_SOURCE DATASET RABITQLIB_HOME INDEX_DIR VEC_BASE VEC_QUERY ORIG_DATASET_DIR ORIG_DATASET \
         D N NQ K MAX_DEGREE EF_CONSTRUCTION METRIC QUANTIZATION_BITS INIT EF_SEARCH ROUNDS \
         BUILD_THREADS SEARCH_THREADS GT_SUFFIX ATTR_DIM ATTR_CARD \
         RANGE_M RANGE_DATASET RANGE_L RANGE_W LOG_DIR PY; do
  if [[ -z "${!v:-}" ]]; then
    echo "[error] missing config key: $v" >&2
    exit 1
  fi
done

# ---- Derive file paths (naming conventions aligned with SIGMOD2027) ----
ATTR_SUFFIX="_A${ATTR_DIM}_C${ATTR_CARD}"
DATA_DIR="$DATA_SOURCE"
BASE_FVECS="$DATA_DIR/${DATASET}_base.fvecs"
QUERY_FVECS="$DATA_DIR/${DATASET}_query.fvecs"
GT_PURE="$DATA_DIR/${DATASET}${GT_SUFFIX}"                                   # groundtruth without filtering
GT_INLINE="$DATA_DIR/${DATASET}_groundtruth${ATTR_SUFFIX}.ivecs"             # groundtruth for equality filtering
GT_RANGE="$DATA_DIR/${DATASET}_groundtruth_range${ATTR_SUFFIX}.ivecs"        # groundtruth for range filtering
BASE_ATTR="$DATA_DIR/base_attr${ATTR_SUFFIX}.bin"
QUERY_ATTR="$DATA_DIR/query_attr${ATTR_SUFFIX}.bin"
RANGE_BIN="$DATA_DIR/query_attr${ATTR_SUFFIX}_range.bin"                     # range-predicate bin (2A columns)

# Original base/query: used for exact reranking (rotation preserves distances, so original-space distance == GT metric, aligned with IVF stage 2).
# Note: indexing and traversal operate on (attribute-rotated) BASE_FVECS; exact reranking uses the original vectors — the two are numerically equivalent.
EXACT_BASE="$DATA_DIR/${DATASET}_base.fvecs"
EXACT_QUERY="$DATA_DIR/${DATASET}_query.fvecs"

# Attribute-rotated base/query (aligned with the IVF attribute rotation; orthogonal rotation preserves distances, so original-space GT stays valid).
# Inline indexing/traversal uses the rotated vectors; rotation depends on A/C, so outputs are isolated per (A,C) in their own directory.
APR_ROOT="${APR_ROOT:-$DATA_DIR}"
APR_DIR="$APR_ROOT/apr_rotated${ATTR_SUFFIX}"
APR_BASE="$APR_DIR/${DATASET}_base.fvecs"
APR_QUERY="$APR_DIR/${DATASET}_query.fvecs"

BUILD_DIR="$RABITQLIB_HOME/build"
BIN_DIR="$RABITQLIB_HOME/bin"
INDEX_BIN="$BIN_DIR/symqg_indexing"
QUERY_BIN="$BIN_DIR/symqg_querying"

# ---- Row-wise filter switch: enabled when both ATTR_BASE and ATTR_QUERY are non-empty ----
USE_INLINE=0
if [[ -n "${ATTR_BASE:-}" && -n "${ATTR_QUERY:-}" ]]; then
  USE_INLINE=1
fi

# ---- Mode selection (enhanced branch / -r hard-gate baseline / equality inline / pure vector) ----
MODE="pure"
if [[ "${RANGE_QUERY:-}" == "on" ]]; then
  MODE="enhanced"
elif [[ "$USE_INLINE" == 1 && "${ATTR_RANGE:-0}" == "1" ]]; then
  MODE="range_hardgate"
elif [[ "$USE_INLINE" == 1 ]]; then
  MODE="inline"
fi

# ---- Rerank switch (only effective in inline mode: 1=rerank with original vectors, 0=off, used for quantization-noise comparison) ----
RERANK="${RERANK:-1}"

# ---- Quantization-bit suffix: bit=0 appends nothing; bit>0 appends _bN, isolating index/log; bit=0/1 do not interfere ----
BITS_SUFFIX=""
if [[ -n "${QUANTIZATION_BITS:-}" && "${QUANTIZATION_BITS:-0}" != "0" ]]; then
  BITS_SUFFIX="_b${QUANTIZATION_BITS}"
fi

# ---- Per-mode dataset / index / groundtruth / attribute / log dependencies ----
case "$MODE" in
  enhanced)
    EFF_DATASET="$RANGE_DATASET"
    INDEX_FILE="$INDEX_DIR/${EFF_DATASET}_symqg_M${MAX_DEGREE}.index"
    SEARCH_QUERY="$DATA_DIR/${EFF_DATASET}_query.fvecs"
    GT_FILE="$GT_RANGE"
    LOG_FILE="$LOG_DIR/symqg_${EFF_DATASET}${ATTR_SUFFIX}_range.log"
    ;;
  range_hardgate)
    EFF_DATASET="$DATASET"
    INDEX_FILE="$INDEX_DIR/${EFF_DATASET}_symqg_M${MAX_DEGREE}.index"
    SEARCH_QUERY="$QUERY_FVECS"
    GT_FILE="$GT_RANGE"
    LOG_FILE="$LOG_DIR/symqg_${DATASET}${ATTR_SUFFIX}_range_hardgate.log"
    ;;
  inline)
    EFF_DATASET="$DATASET"
    INDEX_DIR="${INDEX_DIR}${ATTR_SUFFIX}${BITS_SUFFIX}"   # isolated per (A,C) (rotated base depends on A/C)
    INDEX_FILE="$INDEX_DIR/${EFF_DATASET}_symqg_M${MAX_DEGREE}.index"
    BASE_FVECS="$APR_BASE"          # index/traversal uses the attribute-APR rotated base
    QUERY_FVECS="$APR_QUERY"        # search query uses the attribute-APR rotated query
    SEARCH_QUERY="$QUERY_FVECS"
    GT_FILE="$GT_INLINE"
    rk_suf=""
    [[ "$RERANK" == "0" ]] && rk_suf="_norerank"
    LOG_FILE="$LOG_DIR/${DATASET}${ATTR_SUFFIX}/symqg_fdm_${DATASET}${ATTR_SUFFIX}_inline${BITS_SUFFIX}${rk_suf}.log"
    ;;
  pure)
    EFF_DATASET="$DATASET"
    INDEX_FILE="$INDEX_DIR/${EFF_DATASET}_symqg_M${MAX_DEGREE}.index"
    SEARCH_QUERY="$QUERY_FVECS"
    GT_FILE="$GT_PURE"
    LOG_FILE="$LOG_DIR/symqg_${DATASET}.log"
    ;;
esac

echo "[info] RABITQLIB_HOME=$RABITQLIB_HOME"
echo "[info] DATA_SOURCE=$DATA_SOURCE  DATASET=$DATASET  D=$D  N=$N  NQ=$NQ  K=$K"
echo "[info] ATTR_DIM=$ATTR_DIM ATTR_CARD=$ATTR_CARD  ATTR_SUFFIX=$ATTR_SUFFIX"
echo "[info] MODE=$MODE  eff_dataset=$EFF_DATASET  BITS_SUFFIX=$BITS_SUFFIX"
echo "[info] index: MAX_DEGREE=$MAX_DEGREE EF_CONSTRUCTION=$EF_CONSTRUCTION METRIC=$METRIC QUANTIZATION_BITS=$QUANTIZATION_BITS INIT=$INIT"
echo "[info] search: EF_SEARCH=$EF_SEARCH ROUNDS=$ROUNDS"
echo "[info] BUILD_THREADS=$BUILD_THREADS  SEARCH_THREADS=$SEARCH_THREADS"
echo "[info] LOG_FILE=$LOG_FILE"

# ---- Thread affinity (scheduler-layer control; algorithm sources untouched) ----
BUILD_AFFINITY="0-$((BUILD_THREADS-1))"
SEARCH_AFFINITY="0-$((SEARCH_THREADS-1))"

# ---- Parse -f (force index rebuild) and the stage list ----
FORCE_REBUILD=0
STAGES=()
for a in "$@"; do
  if [[ "$a" == "-f" ]]; then
    FORCE_REBUILD=1
  else
    STAGES+=("$a")
  fi
done
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(prepare build index search)
fi

# =====================================================================
# Stage 1: prepare — invoke data/prepare.py to symlink msmarco data into DATA_SOURCE
# =====================================================================
run_prepare() {
  echo "========== [1/4] prepare =========="
  mkdir -p "$DATA_DIR"
  env -u PYTHONPATH "$PY" "$RABITQLIB_HOME/data/prepare.py" \
    --vec-base "$VEC_BASE" \
    --vec-query "$VEC_QUERY" \
    --orig-dataset "$ORIG_DATASET_DIR" \
    --out-dir "$DATA_DIR" \
    --dataset "$DATASET" \
    --attr-dim "$ATTR_DIM" \
    --attr-card "$ATTR_CARD"
  echo "[prepare] done -> $DATA_DIR"
}

# =====================================================================
# Stage 2: build — out-of-source cmake + compile symqg_indexing / symqg_querying
# =====================================================================
run_build() {
  echo "========== [2/4] build (symqg) =========="
  mkdir -p "$BUILD_DIR"
  bash /home/yq/clean-env.sh cmake -S "$RABITQLIB_HOME" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRABITQ_BUILD_SAMPLES=ON \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=ON
  echo "[build] cmake configured."
  bash /home/yq/clean-env.sh taskset -c "$BUILD_AFFINITY" cmake --build "$BUILD_DIR" \
    --target symqg_indexing symqg_querying --parallel "$BUILD_THREADS"
  if [[ ! -x "$INDEX_BIN" || ! -x "$QUERY_BIN" ]]; then
    echo "[error] binaries not found after build: $INDEX_BIN or $QUERY_BIN" >&2
    exit 1
  fi
  echo "[build] done. binaries: $INDEX_BIN $QUERY_BIN"
}

# =====================================================================
# Stage 3: index — run symqg_indexing to build the symmetric-quantization graph index
#   The enhanced branch builds the graph on the rotated dataset (RANGE_DATASET); others on the original DATASET.
#   args: data.fvecs degree ef index_file [metric] [quant_bits] [init] [threads]
# =====================================================================
run_index() {
  echo "========== [3/4] index (symqg_indexing) =========="
  run_encode   # inline mode: produce the attribute-rotated base (idempotent)
  [[ -x "$INDEX_BIN" ]] || { echo "[error] $INDEX_BIN not found, run build first" >&2; exit 1; }
  [[ -f "$BASE_FVECS" ]] || { echo "[error] $BASE_FVECS not found, run prepare first" >&2; exit 1; }
  if [[ -f "$INDEX_FILE" && "$FORCE_REBUILD" -eq 0 ]]; then
    echo "[index] index exists, skip build (use -f to force rebuild): $INDEX_FILE"
  else
    if [[ "$FORCE_REBUILD" -eq 1 && -f "$INDEX_FILE" ]]; then
      echo "[index] -f given, rebuilding index"
    fi
    mkdir -p "$INDEX_DIR"
    bash /home/yq/clean-env.sh taskset -c "$BUILD_AFFINITY" "$INDEX_BIN" \
      "$BASE_FVECS" "$MAX_DEGREE" "$EF_CONSTRUCTION" "$INDEX_FILE" \
      "$METRIC" "$QUANTIZATION_BITS" "$INIT" "$BUILD_THREADS"
    echo "[index] build complete -> $INDEX_FILE"
  fi
}

# =====================================================================
# Stage 2b: encode_range — produce the range-predicate bin + range GT (+ rotated dataset for the enhanced branch)
#   range_dataset.py always runs (both range_hardgate and enhanced depend on its outputs);
#   encode_dataset_range.py runs only for the enhanced branch (RANGE_QUERY=on) to produce rotated base/query.
# =====================================================================
run_encode_range() {
  echo "========== [2b] encode_range =========="
  # 1) produce the range-predicate bin (Nq x 2A) and the range GT (reuse the existing base_attr as groundtruth)
  echo "---- [2b.1] range_dataset.py : range bin + range GT ----"
  bash /home/yq/clean-env.sh "$PY" "$RABITQLIB_HOME/scripts/range_dataset.py" \
      --src-base "$DATA_DIR/${ORIG_DATASET}_base.fvecs" \
      --src-query "$DATA_DIR/${ORIG_DATASET}_query.fvecs" \
      --out-dir "$DATA_DIR" \
      --base-attr-name "base_attr${ATTR_SUFFIX}.bin" \
      --query-attr-name "query_attr${ATTR_SUFFIX}_range.bin" \
      --groundtruth-name "${ORIG_DATASET}_groundtruth_range${ATTR_SUFFIX}.ivecs" \
      --A "$ATTR_DIM" --card "$ATTR_CARD" --range-w "${RANGE_W:-5}" \
      --K "$K" --n "$N" --nq "$NQ" --d "$D" --metric l2
  # 2) enhanced branch: produce the rotated dataset (rotated base + query M-subcenters)
  if [[ "$MODE" == "enhanced" ]]; then
    echo "---- [2b.2] encode_dataset_range.py : rotated base + M-subcenter query ----"
    bash /home/yq/clean-env.sh "$PY" "$RABITQLIB_HOME/scripts/encode_dataset_range.py" \
        --ds "$DATA_DIR" --out "$DATA_DIR" \
        --d "$D" --A "$ATTR_DIM" --L "${RANGE_L:-$ATTR_CARD}" --M "${RANGE_M:-1}" \
        --base-name "${ORIG_DATASET}_base.fvecs" \
        --query-name "${ORIG_DATASET}_query.fvecs" \
        --base-attr-name "base_attr${ATTR_SUFFIX}.bin" \
        --query-range-name "query_attr${ATTR_SUFFIX}_range.bin" \
        --rotated-base-name "${RANGE_DATASET}_base.fvecs" \
        --rotated-query-name "${RANGE_DATASET}_query.fvecs"
  else
    echo "[encode_range] non-enhanced mode, skip rotated dataset generation (only range bin / range GT produced)."
  fi
  echo "[encode_range] done."
}

# =====================================================================
# Stage 2c: encode — apply attribute-APR rotation to base/query (ported from diskann-fdm)
#   Only needed for inline mode: apply APR/CDMA phase-superposition encoding to the original base/query
#   (encode_apr, each dimension pair carries the phase code of all A attributes), producing APR-rotated
#   vectors for indexing and traversal.
#   Rotation preserves distances, so the original-space GT and exact reranking stay valid; outputs are
#   isolated per (A,C). Idempotent.
# =====================================================================
run_encode() {
  if [[ "$MODE" != "inline" ]]; then
    return 0
  fi
  if [[ -f "$APR_BASE" && -f "$APR_QUERY" ]]; then
    echo "[encode] APR base/query already exist, skip: $APR_DIR"
    return 0
  fi
  mkdir -p "$APR_DIR"
  local IVE_UTILS="$SCRIPT_DIR/scripts"
  echo "========== [2c] encode (attribute-APR rotated base/query) =========="
  echo "[encode] A=$ATTR_DIM C=$ATTR_CARD seed=1234 -> $APR_DIR"
  PYTHONPATH="$IVE_UTILS:${PYTHONPATH:-}" "$PY" "$SCRIPT_DIR/scripts/encode_attr_apr.py" \
      --ds "$DATA_DIR" --out "$APR_DIR" \
      --d "$D" --A "$ATTR_DIM" --enc-seed 1234 \
      --base-name "${DATASET}_base.fvecs" \
      --query-name "${DATASET}_query.fvecs" \
      --base-attr-name "base_attr${ATTR_SUFFIX}.bin" \
      --query-attr-name "query_attr${ATTR_SUFFIX}.bin" \
      --groundtruth-name "${DATASET}_groundtruth${ATTR_SUFFIX}.ivecs" \
      --apr-base-name "${DATASET}_base.fvecs" \
      --apr-query-name "${DATASET}_query.fvecs"
  echo "[encode] done -> $APR_BASE"
}

# =====================================================================
# Stage 4: search — run symqg_querying scanning ef, emit [RESULT] logs
#   args: index query.fvecs gt.ivecs [topk] [efs_csv] [algo] [rounds] [threads]
#         [-b base_attr -a query_attr -A attr_dim -r]
#         [--range_m M --orig_base base.fvecs --orig_query query.fvecs]
# =====================================================================
run_search() {
  echo "========== [4/4] search (symqg_querying) =========="
  run_encode   # inline mode: ensure attribute-rotated query exists (idempotent)
  [[ -x "$QUERY_BIN" ]] || { echo "[error] $QUERY_BIN not found, run build first" >&2; exit 1; }
  [[ -f "$INDEX_FILE" ]] || { echo "[error] index $INDEX_FILE not found, run index first" >&2; exit 1; }

  local attr_args=()
  local extra_args=()
  case "$MODE" in
    enhanced)
      attr_args=(-b "$BASE_ATTR" -a "$RANGE_BIN" -A "$ATTR_DIM")
      extra_args=(--range_m "${RANGE_M:-1}" --orig_base "$BASE_FVECS" --orig_query "$QUERY_FVECS")
      echo "[search] enhanced branch (rotation+M-subcenter+rerank gate): dataset=$EFF_DATASET range_m=${RANGE_M:-1}"
      ;;
    range_hardgate)
      attr_args=(-b "$BASE_ATTR" -a "$RANGE_BIN" -A "$ATTR_DIM" -r)
      echo "[search] -r hard-gate baseline (range): range_bin=$RANGE_BIN"
      ;;
    inline)
      attr_args=(-b "$BASE_ATTR" -a "$QUERY_ATTR" -A "$ATTR_DIM")
      if [[ "$RERANK" == "1" ]]; then
        attr_args+=(--orig_base "$EXACT_BASE" --orig_query "$EXACT_QUERY")
      fi
      echo "[search] row-wise attribute equality filtering (inline): base=$BASE_ATTR query=$QUERY_ATTR  rerank=$RERANK"
      ;;
    pure)
      echo "[search] attribute filtering disabled (pure vector search)"
      ;;
  esac

  # enhanced branch / -r baseline depend on the range bin and range GT
  if [[ "$MODE" == "enhanced" || "$MODE" == "range_hardgate" ]]; then
    [[ -f "$RANGE_BIN" ]] || { echo "[error] range bin $RANGE_BIN not found, run encode_range first" >&2; exit 1; }
    [[ -f "$GT_FILE" ]] || { echo "[error] range groundtruth $GT_FILE not found, run encode_range first" >&2; exit 1; }
  fi

  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"   # clear previous results before each run

  echo "[search] index=$INDEX_FILE query=$SEARCH_QUERY gt=$GT_FILE"
  bash /home/yq/clean-env.sh taskset -c "$SEARCH_AFFINITY" env \
      OMP_NUM_THREADS="$SEARCH_THREADS" \
      OPENBLAS_NUM_THREADS="$SEARCH_THREADS" \
      MKL_NUM_THREADS="$SEARCH_THREADS" \
      "$QUERY_BIN" "$INDEX_FILE" "$SEARCH_QUERY" "$GT_FILE" \
      "$K" "$EF_SEARCH" symqg "$ROUNDS" "$SEARCH_THREADS" \
      "${attr_args[@]}" "${extra_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"
  echo "[search] done -> result log: $LOG_FILE"
}

# ---- Main flow: execute the requested stages in order ----
for stage in "${STAGES[@]}"; do
  case "$stage" in
    prepare)      run_prepare ;;
    build)        run_build   ;;
    index)        run_index   ;;
    encode_range) run_encode_range ;;
    encode)       run_encode  ;;
    search)       run_search  ;;
    *)
      echo "[warn] unknown stage: $stage (valid: prepare / build / index / encode_range / search)" >&2
      ;;
  esac
done

echo "[all done] requested stages complete."
