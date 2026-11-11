#!/usr/bin/env bash
# Batch-generate range-attribute datasets (range attribute: query predicate is a
#   [lo,hi] interval). All runtime parameters live in config.cfg (same folder).
# Usage:
#   bash run_gen_range_attr_gt.sh [dataset ...]    # no args -> config.cfg RANGE_DATASETS
#   MARGIN=256 bash run_gen_range_attr_gt.sh       # env vars override config for one run
# Each dataset is generated config-by-config (RANGE_CONFIGS) in sequence.
# Detailed logs -> $LOG_DIR/gen_range_attr_gt_<ds>.log; progress -> gen_range_attr_gt_master.log.
# Note: the generator has a built-in self_check (256 samples, same logic as the
#   verifier) so no external verifier script is required.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

# ---------------- Read config (KEY=VALUE) ----------------
# Skip comments/blank lines/non-uppercase keys; strip surrounding whitespace and
# matching quotes. Variables already set in the environment are not overwritten
# (env takes precedence over this file).
CONFIG_FILE="${CONFIG_FILE:-$HERE/config.cfg}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] config file not found: $CONFIG_FILE" >&2
  exit 1
fi
while IFS='=' read -r key value; do
  key="$(echo "$key" | xargs 2>/dev/null)"
  value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
  [[ -z "$key" || "$key" == \#* || -z "$value" ]] && continue
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
  [[ -n "${!key+x}" ]] && continue
  declare -g "$key=$value"
done < "$CONFIG_FILE"

# ---------------- Apply parameters (config.cfg > built-in defaults) ----------------
PY="${PY:-python3}"
LOG_DIR="${LOG_DIR:-./_log}"
DATASET_ROOT="${DATASET_ROOT:-..}"
NO_VERIFY="${NO_VERIFY:-0}"
RANGE_CPP_BIN="${RANGE_CPP_BIN:-compute_range_attr_gt}"
RANGE_K="${RANGE_K:-100}"
RANGE_SEED="${RANGE_SEED:-1000}"
RANGE_MARGIN="${RANGE_MARGIN:-512}"
RANGE_THREADS="${RANGE_THREADS:-0}"
RANGE_DATASETS="${RANGE_DATASETS:-sift msmarco openai1536}"
RANGE_CONFIGS="${RANGE_CONFIGS:-1 100 50;1 100 10;2 100 10;3 100 10}"
RANGE_METRICS="${RANGE_METRICS:-sift=l2 msmarco=l2 openai1536=ip}"

# Binary name from config; script joins $HERE to an absolute path; env CPP_BIN overrides all.
CPP_BIN="${CPP_BIN:-$HERE/$RANGE_CPP_BIN}"

# ---------------- Light validation: catch config typos before running ----------------
die() { echo "[error] $*" >&2; exit 1; }
[[ "$RANGE_K" =~ ^[1-9][0-9]*$ ]]        || die "RANGE_K must be a positive integer, got '$RANGE_K'"
[[ "$RANGE_SEED" =~ ^[0-9]+$ ]]          || die "RANGE_SEED must be a non-negative integer, got '$RANGE_SEED'"
[[ "$RANGE_MARGIN" =~ ^[0-9]+$ ]]        || die "RANGE_MARGIN must be a non-negative integer, got '$RANGE_MARGIN'"
[[ "$RANGE_THREADS" =~ ^[0-9]+$ ]]       || die "RANGE_THREADS must be a non-negative integer, got '$RANGE_THREADS'"
[[ "$RANGE_CPP_BIN" != */* ]]            || die "RANGE_CPP_BIN should be a binary filename (relative to this folder), got '$RANGE_CPP_BIN'"

# metric map: ds=metric pairs from RANGE_METRICS (default l2 if absent)
declare -A METRIC=()
for kv in $RANGE_METRICS; do
  [[ "$kv" == *=* ]] || die "RANGE_METRICS entries must be ds=metric, got '$kv'"
  METRIC["${kv%%=*}"]="${kv##*=}"
done
for m in ${METRIC[@]+"${METRIC[@]}"}; do
  [[ "$m" == "ip" || "$m" == "l2" ]] || die "RANGE_METRICS metric must be ip|l2, got '$m'"
done

# range selectivity config: A C w triples, semicolon-separated (tolerate leading/
# trailing/consecutive semicolons and extra whitespace)
CONFIGS=()
IFS=';' read -ra _RAW <<< "$RANGE_CONFIGS"
for cfg in "${_RAW[@]}"; do
  cfg="$(echo "$cfg" | xargs 2>/dev/null)"
  [[ -n "$cfg" ]] && CONFIGS+=("$cfg")
done
[[ ${#CONFIGS[@]} -gt 0 ]] || die "RANGE_CONFIGS must not be empty"
for cfg in "${CONFIGS[@]}"; do
  read -r A C W <<< "$cfg"
  [[ "$A" =~ ^[1-9][0-9]*$ && "$C" =~ ^[1-9][0-9]*$ && "$W" =~ ^[1-9][0-9]*$ ]] \
    || die "RANGE_CONFIGS entries must be 'A C w' (positive-integer triple, semicolon-separated), got '$cfg'"
done

NVF=""
if [ "$NO_VERIFY" = "1" ]; then NVF="--no-verify"; fi

mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/gen_range_attr_gt_master.log"

# Auto-compile the C++ binary if missing (needs only g++ + OpenMP)
if [ ! -x "$CPP_BIN" ]; then
  echo "[$(date '+%F %T')] compiling $RANGE_CPP_BIN ..." | tee -a "$MASTER"
  ( cd "$HERE" && g++ -O3 -fopenmp -std=c++17 -o "$RANGE_CPP_BIN" "$RANGE_CPP_BIN.cpp" ) \
    || { echo "failed to compile $RANGE_CPP_BIN" | tee -a "$MASTER"; exit 1; }
fi

# ---------------- Dataset list: CLI positional args > config.cfg ----------------
DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  read -ra DATASETS <<< "$RANGE_DATASETS"
fi

echo "[$(date '+%F %T')] ===== start range-attribute batch generation: DATASETS=${DATASETS[*]} =====" | tee -a "$MASTER"
echo "[info] config: $CONFIG_FILE | K=$RANGE_K seed=$RANGE_SEED margin=$RANGE_MARGIN threads=$RANGE_THREADS no_verify=$NO_VERIFY" | tee -a "$MASTER"

for ds in "${DATASETS[@]}"; do
  metric="${METRIC[$ds]:-l2}"
  echo "[$(date '+%F %T')] ===== $ds (metric=$metric) begin =====" | tee -a "$MASTER"
  for cfg in "${CONFIGS[@]}"; do
    read -r A C W <<< "$cfg"
    echo "[$(date '+%F %T')]   $ds A${A}_C${C}_w${W} begin ..." | tee -a "$MASTER"
    if env -u PYTHONPATH "$PY" gen_range_attr_gt.py \
        --ds-dir "$DATASET_ROOT/$ds" \
        --base-name "${ds}_base.fvecs" \
        --query-name "${ds}_query.fvecs" \
        --A "$A" --C "$C" --w "$W" --metric "$metric" --K "$RANGE_K" --seed "$RANGE_SEED" \
        --cpp-bin "$CPP_BIN" --margin "$RANGE_MARGIN" --threads "$RANGE_THREADS" $NVF \
        >> "$LOG_DIR/gen_range_attr_gt_${ds}.log" 2>&1; then
      # pull the last [ok] summary line from the per-dataset log
      okline="$(grep -F '[ok]' "$LOG_DIR/gen_range_attr_gt_${ds}.log" | tail -1)"
      echo "[$(date '+%F %T')]   $ds A${A}_C${C}_w${W} done $okline" | tee -a "$MASTER"
    else
      echo "[$(date '+%F %T')]   $ds A${A}_C${C}_w${W} FAILED (see $LOG_DIR/gen_range_attr_gt_${ds}.log)" | tee -a "$MASTER"
    fi
  done
  echo "[$(date '+%F %T')] ===== $ds end =====" | tee -a "$MASTER"
done

echo "[$(date '+%F %T')] ===== all done =====" | tee -a "$MASTER"
