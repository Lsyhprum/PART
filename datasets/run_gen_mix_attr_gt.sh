#!/usr/bin/env bash
# Batch-generate mixed-attribute datasets (each vector = 1 equality attribute +
#   1 range attribute, two-condition AND filter). All runtime parameters live in
#   config.cfg (same folder) — tune there, not here.
# Usage:
#   bash run_gen_mix_attr_gt.sh [dataset ...]    # no args -> config.cfg MIX_DATASETS
#   MIX_W=20 bash run_gen_mix_attr_gt.sh         # env vars override config for one run
# Each dataset is generated with one mixed config (fixed 1 equality + 1 range; the
#   selectivity rate is set by C_eq / C_range / w).
# Detailed logs -> $LOG_DIR/gen_mix_attr_gt_<ds>.log; progress -> gen_mix_attr_gt_master.log.
# Selectivity: sel = (1/MIX_C_EQ) * (MIX_W / MIX_C_RANGE); default 1/10 * 10/100 = 1%.
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
MIX_CPP_BIN="${MIX_CPP_BIN:-compute_mix_attr_gt}"
MIX_K="${MIX_K:-100}"
MIX_SEED="${MIX_SEED:-1000}"
MIX_MARGIN="${MIX_MARGIN:-512}"
MIX_THREADS="${MIX_THREADS:-0}"
MIX_C_EQ="${MIX_C_EQ:-10}"
MIX_C_RANGE="${MIX_C_RANGE:-100}"
MIX_W="${MIX_W:-10}"
MIX_DATASETS="${MIX_DATASETS:-sift deep1M msmarco openai1536 dbpedia1024 gist msong}"
MIX_METRICS="${MIX_METRICS:-sift=l2 deep1M=l2 msmarco=l2 gist=l2 msong=l2 openai1536=ip dbpedia1024=ip}"

# Binary name from config; script joins $HERE to an absolute path; env CPP_BIN overrides all.
CPP_BIN="${CPP_BIN:-$HERE/$MIX_CPP_BIN}"

# ---------------- Light validation: catch config typos before running ----------------
die() { echo "[error] $*" >&2; exit 1; }
[[ "$MIX_K" =~ ^[1-9][0-9]*$ ]]        || die "MIX_K must be a positive integer, got '$MIX_K'"
[[ "$MIX_SEED" =~ ^[0-9]+$ ]]          || die "MIX_SEED must be a non-negative integer, got '$MIX_SEED'"
[[ "$MIX_MARGIN" =~ ^[0-9]+$ ]]        || die "MIX_MARGIN must be a non-negative integer, got '$MIX_MARGIN'"
[[ "$MIX_THREADS" =~ ^[0-9]+$ ]]       || die "MIX_THREADS must be a non-negative integer, got '$MIX_THREADS'"
[[ "$MIX_CPP_BIN" != */* ]]            || die "MIX_CPP_BIN should be a binary filename (relative to this folder), got '$MIX_CPP_BIN'"
[[ "$MIX_C_EQ" =~ ^[0-9]+$ ]]          || die "MIX_C_EQ must be an integer (equality cardinality), got '$MIX_C_EQ'"
(( MIX_C_EQ >= 2 ))                    || die "MIX_C_EQ must be >= 2 (equality cardinality), got '$MIX_C_EQ'"
[[ "$MIX_C_RANGE" =~ ^[0-9]+$ ]]       || die "MIX_C_RANGE must be an integer (range cardinality), got '$MIX_C_RANGE'"
(( MIX_C_RANGE >= 2 ))                 || die "MIX_C_RANGE must be >= 2 (range cardinality), got '$MIX_C_RANGE'"
[[ "$MIX_W" =~ ^[1-9][0-9]*$ ]]        || die "MIX_W must be a positive integer (range width), got '$MIX_W'"
(( MIX_W <= MIX_C_RANGE ))             || die "MIX_W($MIX_W) cannot exceed MIX_C_RANGE($MIX_C_RANGE)"

# metric map: ds=metric pairs from MIX_METRICS (default l2 if absent)
declare -A METRIC=()
for kv in $MIX_METRICS; do
  [[ "$kv" == *=* ]] || die "MIX_METRICS entries must be ds=metric, got '$kv'"
  METRIC["${kv%%=*}"]="${kv##*=}"
done
for m in ${METRIC[@]+"${METRIC[@]}"}; do
  [[ "$m" == "ip" || "$m" == "l2" ]] || die "MIX_METRICS metric must be ip|l2, got '$m'"
done

NVF=""
if [ "$NO_VERIFY" = "1" ]; then NVF="--no-verify"; fi

mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/gen_mix_attr_gt_master.log"

# Auto-compile the C++ binary if missing (needs only g++ + OpenMP)
if [ ! -x "$CPP_BIN" ]; then
  echo "[$(date '+%F %T')] compiling $MIX_CPP_BIN ..." | tee -a "$MASTER"
  ( cd "$HERE" && g++ -O3 -fopenmp -std=c++17 -o "$MIX_CPP_BIN" "$MIX_CPP_BIN.cpp" ) \
    || { echo "failed to compile $MIX_CPP_BIN" | tee -a "$MASTER"; exit 1; }
fi

# ---------------- Dataset list: CLI positional args > config.cfg ----------------
DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  read -ra DATASETS <<< "$MIX_DATASETS"
fi

SEL="$(awk "BEGIN{printf \"%.4f\", (1/$MIX_C_EQ)*($MIX_W/$MIX_C_RANGE)}")"
echo "[$(date '+%F %T')] ===== start mixed-attribute batch generation: DATASETS=${DATASETS[*]} =====" | tee -a "$MASTER"
echo "[info] config: $CONFIG_FILE | 1 equality(C=$MIX_C_EQ) + 1 range(C=$MIX_C_RANGE,w=$MIX_W) -> sel~=$SEL | K=$MIX_K seed=$MIX_SEED margin=$MIX_MARGIN threads=$MIX_THREADS no_verify=$NO_VERIFY" | tee -a "$MASTER"

for ds in "${DATASETS[@]}"; do
  metric="${METRIC[$ds]:-l2}"
  echo "[$(date '+%F %T')] ===== $ds (metric=$metric) begin =====" | tee -a "$MASTER"
  if env -u PYTHONPATH "$PY" gen_mix_attr_gt.py \
      --ds-dir "$DATASET_ROOT/$ds" \
      --base-name "${ds}_base.fvecs" \
      --query-name "${ds}_query.fvecs" \
      --C-eq "$MIX_C_EQ" --C-range "$MIX_C_RANGE" --w "$MIX_W" \
      --metric "$metric" --K "$MIX_K" --seed "$MIX_SEED" \
      --cpp-bin "$CPP_BIN" --margin "$MIX_MARGIN" --threads "$MIX_THREADS" $NVF \
      >> "$LOG_DIR/gen_mix_attr_gt_${ds}.log" 2>&1; then
    # pull the last [ok] summary line from the per-dataset log
    okline="$(grep -F '[ok]' "$LOG_DIR/gen_mix_attr_gt_${ds}.log" | tail -1)"
    echo "[$(date '+%F %T')]   $ds MIX_EQ_C${MIX_C_EQ}_R${MIX_C_RANGE}_w${MIX_W} done $okline" | tee -a "$MASTER"
  else
    echo "[$(date '+%F %T')]   $ds MIX_EQ_C${MIX_C_EQ}_R${MIX_C_RANGE}_w${MIX_W} FAILED (see $LOG_DIR/gen_mix_attr_gt_${ds}.log)" | tee -a "$MASTER"
  fi
done

echo "[$(date '+%F %T')] ===== all done =====" | tee -a "$MASTER"
