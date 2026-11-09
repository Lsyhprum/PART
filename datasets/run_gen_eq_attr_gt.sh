#!/usr/bin/env bash
# Batch-generate equality-attribute datasets (attribute + filtered gt + queries.json).
# All runtime parameters live in config.cfg (same folder) — tune there, not here.
# Usage:
#   bash run_gen_eq_attr_gt.sh [dataset ...]      # no args -> config.cfg EQ_DATASETS
#   MARGIN=256 bash run_gen_eq_attr_gt.sh         # env vars override config for one run
# Each dataset is generated config-by-config (EQ_CONFIGS) in sequence.
# Detailed logs -> $LOG_DIR/gen_eq_attr_gt_<ds>.log; progress -> gen_eq_attr_gt_master.log.
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
EQ_CPP_BIN="${EQ_CPP_BIN:-compute_eq_attr_gt}"
EQ_K="${EQ_K:-100}"
EQ_SEED="${EQ_SEED:-1000}"
EQ_MARGIN="${EQ_MARGIN:-512}"
EQ_THREADS="${EQ_THREADS:-0}"
EQ_DATASETS="${EQ_DATASETS:-sift deep1M msmarco openai1536 dbpedia1024 gist msong}"
EQ_CONFIGS="${EQ_CONFIGS:-1 2;1 10;2 10;3 10}"
EQ_METRICS="${EQ_METRICS:-sift=l2 deep1M=l2 msmarco=l2 gist=l2 msong=l2 openai1536=ip dbpedia1024=ip}"

# Binary name from config; script joins $HERE to an absolute path; env CPP_BIN overrides all.
CPP_BIN="${CPP_BIN:-$HERE/$EQ_CPP_BIN}"

# ---------------- Light validation: catch config typos before running ----------------
die() { echo "[error] $*" >&2; exit 1; }
[[ "$EQ_K" =~ ^[1-9][0-9]*$ ]]          || die "EQ_K must be a positive integer, got '$EQ_K'"
[[ "$EQ_SEED" =~ ^[0-9]+$ ]]            || die "EQ_SEED must be a non-negative integer, got '$EQ_SEED'"
[[ "$EQ_MARGIN" =~ ^[0-9]+$ ]]          || die "EQ_MARGIN must be a non-negative integer, got '$EQ_MARGIN'"
[[ "$EQ_THREADS" =~ ^[0-9]+$ ]]         || die "EQ_THREADS must be a non-negative integer, got '$EQ_THREADS'"
[[ "$EQ_CPP_BIN" != */* ]]              || die "EQ_CPP_BIN should be a binary filename (relative to this folder), got '$EQ_CPP_BIN'"

# metric map: ds=metric pairs from EQ_METRICS (default l2 if absent)
declare -A METRIC=()
for kv in $EQ_METRICS; do
  [[ "$kv" == *=* ]] || die "EQ_METRICS entries must be ds=metric, got '$kv'"
  METRIC["${kv%%=*}"]="${kv##*=}"
done
for m in ${METRIC[@]+"${METRIC[@]}"}; do
  [[ "$m" == "ip" || "$m" == "l2" ]] || die "EQ_METRICS metric must be ip|l2, got '$m'"
done

# selectivity config: A C pairs, semicolon-separated (tolerate leading/trailing/
# consecutive semicolons and extra whitespace)
CONFIGS=()
IFS=';' read -ra _RAW <<< "$EQ_CONFIGS"
for cfg in "${_RAW[@]}"; do
  cfg="$(echo "$cfg" | xargs 2>/dev/null)"
  [[ -n "$cfg" ]] && CONFIGS+=("$cfg")
done
[[ ${#CONFIGS[@]} -gt 0 ]] || die "EQ_CONFIGS must not be empty"
for cfg in "${CONFIGS[@]}"; do
  read -r A C <<< "$cfg"
  [[ "$A" =~ ^[1-9][0-9]*$ && "$C" =~ ^[1-9][0-9]*$ ]] \
    || die "EQ_CONFIGS entries must be 'A C' (positive-integer pair, semicolon-separated), got '$cfg'"
done

NVF=""
if [ "$NO_VERIFY" = "1" ]; then NVF="--no-verify"; fi

mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/gen_eq_attr_gt_master.log"

# Auto-compile the C++ binary if missing (needs only g++ + OpenMP)
if [ ! -x "$CPP_BIN" ]; then
  echo "[$(date '+%F %T')] compiling $EQ_CPP_BIN ..." | tee -a "$MASTER"
  ( cd "$HERE" && g++ -O3 -fopenmp -std=c++17 -o "$EQ_CPP_BIN" "$EQ_CPP_BIN.cpp" ) \
    || { echo "failed to compile $EQ_CPP_BIN" | tee -a "$MASTER"; exit 1; }
fi

# ---------------- Dataset list: CLI positional args > config.cfg ----------------
DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  read -ra DATASETS <<< "$EQ_DATASETS"
fi

echo "[$(date '+%F %T')] ===== start equality-attribute batch generation: DATASETS=${DATASETS[*]} =====" | tee -a "$MASTER"
echo "[info] config: $CONFIG_FILE | K=$EQ_K seed=$EQ_SEED margin=$EQ_MARGIN threads=$EQ_THREADS no_verify=$NO_VERIFY" | tee -a "$MASTER"

for ds in "${DATASETS[@]}"; do
  metric="${METRIC[$ds]:-l2}"
  echo "[$(date '+%F %T')] ===== $ds (metric=$metric) begin =====" | tee -a "$MASTER"
  for cfg in "${CONFIGS[@]}"; do
    read -r A C <<< "$cfg"
    echo "[$(date '+%F %T')]   $ds A${A}_C${C} begin ..." | tee -a "$MASTER"
    if env -u PYTHONPATH "$PY" gen_eq_attr_gt.py \
        --ds-dir "$DATASET_ROOT/$ds" \
        --base-name "${ds}_base.fvecs" \
        --query-name "${ds}_query.fvecs" \
        --A "$A" --C "$C" --metric "$metric" --K "$EQ_K" --seed "$EQ_SEED" \
        --cpp-bin "$CPP_BIN" --margin "$EQ_MARGIN" --threads "$EQ_THREADS" $NVF \
        >> "$LOG_DIR/gen_eq_attr_gt_${ds}.log" 2>&1; then
      # pull the last [ok] summary line from the per-dataset log
      okline="$(grep -F '[ok]' "$LOG_DIR/gen_eq_attr_gt_${ds}.log" | tail -1)"
      echo "[$(date '+%F %T')]   $ds A${A}_C${C} done $okline" | tee -a "$MASTER"
    else
      echo "[$(date '+%F %T')]   $ds A${A}_C${C} FAILED (see $LOG_DIR/gen_eq_attr_gt_${ds}.log)" | tee -a "$MASTER"
    fi
  done
  echo "[$(date '+%F %T')] ===== $ds end =====" | tee -a "$MASTER"
done

echo "[$(date '+%F %T')] ===== all done =====" | tee -a "$MASTER"
