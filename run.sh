#!/usr/bin/env bash
# ============================================================
# run.sh — 读取 config.cfg，执行 RaBitQ-Library-inline-rot (symqg) baseline 全流程：
#   prepare -> build -> index -> search
# 增强臂需额外先运行 encode_range 阶段（生成 range bin / range GT / 旋转数据集）。
#
# 约定对齐 SIGMOD2027/baseline 下 RaBitQ-Library-post / RaBitQ-inline / NHQ / ACORN：
#   - 解析 config.cfg（仅加载 KEY=VALUE 行，跳过注释/空行，去两端空白与引号）
#   - 相对路径锚定到 SCRIPT_DIR（本配置均为绝对路径，原样保留）
#   - 由 ATTR_DIM/ATTR_CARD 派生命名（base_attr_A*_C*.bin 等）
#   - 线程/库路径注入（调度层控制，不改算法源码）
#   - 统一 LOG_DIR 日志（每次 search 前清空历史 [RESULT] 行）
#
# 四种检索模式（由 config 自动选择）：
#   pure           : ATTR_BASE/ATTR_QUERY 均置空
#   inline         : ATTR_BASE/ATTR_QUERY 均非空 且 ATTR_RANGE=0  （图内等值过滤）
#   range_hardgate : ATTR_BASE/ATTR_QUERY 均非空 且 ATTR_RANGE=1  （-r 内核硬闸门）
#   enhanced       : RANGE_QUERY=on  （RoPE 旋转 + M 子中心并集 + 原始空间重排闸门）
#
# 用法：
#   bash run.sh                        # prepare -> build -> index -> search（按当前 config 模式）
#   bash run.sh prepare                # 仅软链数据（向量/真值/属性 -> DATA_SOURCE）
#   bash run.sh build                  # 仅 cmake 编译 symqg_indexing / symqg_querying
#   bash run.sh index                  # 仅构建 symqg 图索引（已存在则跳过，除非 -f）
#   bash run.sh encode_range           # 仅生成 range bin/GT 与（enhanced 时）旋转数据集
#   bash run.sh search                 # 仅检索并输出 [RESULT] 日志
#   bash run.sh -f                     # 强制重建索引（即使索引文件已存在）
#   也可组合：bash run.sh prepare build index encode_range search
# ============================================================
set -euo pipefail

# ---- 定位脚本所在目录（保证从任意位置调用都正确）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.cfg"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] 找不到配置文件: $CONFIG_FILE" >&2
  exit 1
fi

# ---- 环境变量覆盖层（sweep 批量驱动用）----
# 注意：必须在读取 config 之前先快照环境变量覆盖值——否则 config 的 declare -g 会把
# 环境值覆盖掉，导致后续 ${!_k} 读到的是 config 值而非 export 值。
# 模式开关（ATTR_BASE/ATTR_QUERY/ATTR_RANGE/RANGE_QUERY 等）不在此列，以保证 inline 固定。
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

# ---- 读取配置（仅加载 KEY=VALUE 行，跳过注释/空行，去两端空白与引号）----
while IFS='=' read -r key value; do
  key="$(echo "$key" | xargs)"
  value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
  [[ -z "$key" || "$key" == \#* || -z "$value" ]] && continue
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
  declare -g "$key=$value"
done < "$CONFIG_FILE"

# ---- 应用环境变量覆盖（sweep 批量驱动用；不影响单跑）----
# 覆盖值来自读取 config 前快照的环境变量（见上方 _ENV_OVERRIDE）。
for _k in "${OVERRIDE_KEYS[@]}"; do
  _e="${_ENV_OVERRIDE[$_k]:-}"
  if [[ -n "$_e" ]]; then
    declare -g "$_k=$_e"
  fi
done
# ORIG_DATASET 默认与 DATASET 同源（prepare 后 DATA_SOURCE 下文件名 stem 一致）
ORIG_DATASET="${ORIG_DATASET:-$DATASET}"

# ---- 将可能相对的路径锚定到脚本所在目录 SCRIPT_DIR ----
for _p in DATA_SOURCE INDEX_DIR RABITQLIB_HOME LOG_DIR VEC_BASE VEC_QUERY ORIG_DATASET_DIR; do
  _v="${!_p}"
  case "$_v" in
    /*) ;;
    *)  _v="$SCRIPT_DIR/$_v" ;;
  esac
  declare -g "$_p=$_v"
done

# ---- 校验必要参数（ATTR_BASE/ATTR_QUERY/ATTR_RANGE/RANGE_QUERY 为可选开关，不强制）----
for v in DATA_SOURCE DATASET RABITQLIB_HOME INDEX_DIR VEC_BASE VEC_QUERY ORIG_DATASET_DIR ORIG_DATASET \
         D N NQ K MAX_DEGREE EF_CONSTRUCTION METRIC QUANTIZATION_BITS INIT EF_SEARCH ROUNDS \
         BUILD_THREADS SEARCH_THREADS GT_SUFFIX ATTR_DIM ATTR_CARD \
         RANGE_M RANGE_DATASET RANGE_L RANGE_W LOG_DIR PY; do
  if [[ -z "${!v:-}" ]]; then
    echo "[error] 配置缺失: $v" >&2
    exit 1
  fi
done

# ---- 派生文件路径（命名规则对齐 SIGMOD2027 约定）----
ATTR_SUFFIX="_A${ATTR_DIM}_C${ATTR_CARD}"
DATA_DIR="$DATA_SOURCE"
BASE_FVECS="$DATA_DIR/${DATASET}_base.fvecs"
QUERY_FVECS="$DATA_DIR/${DATASET}_query.fvecs"
GT_PURE="$DATA_DIR/${DATASET}${GT_SUFFIX}"                                   # 无过滤真值
GT_INLINE="$DATA_DIR/${DATASET}_groundtruth${ATTR_SUFFIX}.ivecs"             # 等值过滤真值
GT_RANGE="$DATA_DIR/${DATASET}_groundtruth_range${ATTR_SUFFIX}.ivecs"        # 范围过滤真值
BASE_ATTR="$DATA_DIR/base_attr${ATTR_SUFFIX}.bin"
QUERY_ATTR="$DATA_DIR/query_attr${ATTR_SUFFIX}.bin"
RANGE_BIN="$DATA_DIR/query_attr${ATTR_SUFFIX}_range.bin"                     # 范围谓词 bin (2A 列)

# 原始 base/query：精确重排用（旋转保距，原始空间距离==GT 口径，对齐 IVF 阶段2）。
# 注意：索引与遍历在（属性旋转后的）BASE_FVECS 上进行；精确重排用原始向量，二者数值等价。
EXACT_BASE="$DATA_DIR/${DATASET}_base.fvecs"
EXACT_QUERY="$DATA_DIR/${DATASET}_query.fvecs"

# 属性旋转 base/query（对齐 IVF encode_disjoint；正交旋转保距，原始空间 GT 仍有效）。
# inline 索引与遍历用旋转向量；旋转依赖 A/C，故按 ac 隔离目录。
FDM_ROOT="${FDM_ROOT:-$DATA_DIR}"
FDM_DIR="$FDM_ROOT/fdm_rotated${ATTR_SUFFIX}"
FDM_BASE="$FDM_DIR/${DATASET}_base.fvecs"
FDM_QUERY="$FDM_DIR/${DATASET}_query.fvecs"

BUILD_DIR="$RABITQLIB_HOME/build"
BIN_DIR="$RABITQLIB_HOME/bin"
INDEX_BIN="$BIN_DIR/symqg_indexing"
QUERY_BIN="$BIN_DIR/symqg_querying"

# ---- 行间过滤开关：ATTR_BASE 与 ATTR_QUERY 均非空即启用 ----
USE_INLINE=0
if [[ -n "${ATTR_BASE:-}" && -n "${ATTR_QUERY:-}" ]]; then
  USE_INLINE=1
fi

# ---- 模式判定（增强臂 / -r 硬闸门 baseline / 等值 inline / 纯向量）----
MODE="pure"
if [[ "${RANGE_QUERY:-}" == "on" ]]; then
  MODE="enhanced"
elif [[ "$USE_INLINE" == 1 && "${ATTR_RANGE:-0}" == "1" ]]; then
  MODE="range_hardgate"
elif [[ "$USE_INLINE" == 1 ]]; then
  MODE="inline"
fi

# ---- 重排臂开关（仅 inline 模式生效：1=用原始向量重排，0=关闭，用于量化噪声对照）----
RERANK="${RERANK:-1}"

# ---- 量化比特后缀：bit=0 不追加；bit>0 追加 _bN，隔离索引/日志，bit=0/1 互不干扰 ----
BITS_SUFFIX=""
if [[ -n "${QUANTIZATION_BITS:-}" && "${QUANTIZATION_BITS:-0}" != "0" ]]; then
  BITS_SUFFIX="_b${QUANTIZATION_BITS}"
fi

# ---- 各模式依赖的 dataset / 索引 / 真值 / 属性 / 日志 ----
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
    INDEX_DIR="${INDEX_DIR}${ATTR_SUFFIX}${BITS_SUFFIX}"   # 按 ac 隔离（旋转 base 依赖 A/C）
    INDEX_FILE="$INDEX_DIR/${EFF_DATASET}_symqg_M${MAX_DEGREE}.index"
    BASE_FVECS="$FDM_BASE"          # 索引/遍历用属性 FDM 旋转 base
    QUERY_FVECS="$FDM_QUERY"        # 检索 query 用属性 FDM 旋转 query
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

# ---- 线程 affinity（调度层控制，不改算法源码）----
BUILD_AFFINITY="0-$((BUILD_THREADS-1))"
SEARCH_AFFINITY="0-$((SEARCH_THREADS-1))"

# ---- 解析 -f（强制重建索引）与阶段列表 ----
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
# 阶段 1：prepare —— 调用 data/prepare.py 软链 msmarco 数据到 DATA_SOURCE
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
# 阶段 2：build —— out-of-source cmake + 编译 symqg_indexing / symqg_querying
# =====================================================================
run_build() {
  echo "========== [2/4] build (symqg) =========="
  mkdir -p "$BUILD_DIR"
  bash /home/yq/clean-env.sh cmake -S "$RABITQLIB_HOME" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRABITQ_BUILD_SAMPLES=ON \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=OFF \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=ON
  echo "[build] cmake configured."
  bash /home/yq/clean-env.sh taskset -c "$BUILD_AFFINITY" cmake --build "$BUILD_DIR" \
    --target symqg_indexing symqg_querying --parallel "$BUILD_THREADS"
  if [[ ! -x "$INDEX_BIN" || ! -x "$QUERY_BIN" ]]; then
    echo "[error] 编译后未找到 $INDEX_BIN 或 $QUERY_BIN" >&2
    exit 1
  fi
  echo "[build] done. binaries: $INDEX_BIN $QUERY_BIN"
}

# =====================================================================
# 阶段 3：index —— 运行 symqg_indexing 构建对称量化图索引
#   增强臂在旋转数据集（RANGE_DATASET）上建图，其余在原 DATASET 上建图。
#   参数: data.fvecs degree ef index_file [metric] [quant_bits] [init] [threads]
# =====================================================================
run_index() {
  echo "========== [3/4] index (symqg_indexing) =========="
  run_encode   # inline 模式生成属性旋转 base（幂等）
  [[ -x "$INDEX_BIN" ]] || { echo "[error] 找不到 $INDEX_BIN，请先运行 build" >&2; exit 1; }
  [[ -f "$BASE_FVECS" ]] || { echo "[error] 找不到 $BASE_FVECS，请先运行 prepare" >&2; exit 1; }
  if [[ -f "$INDEX_FILE" && "$FORCE_REBUILD" -eq 0 ]]; then
    echo "[index] 索引已存在，跳过构建（使用 -f 强制重建）: $INDEX_FILE"
  else
    if [[ "$FORCE_REBUILD" -eq 1 && -f "$INDEX_FILE" ]]; then
      echo "[index] -f 指定，重新构建索引"
    fi
    mkdir -p "$INDEX_DIR"
    bash /home/yq/clean-env.sh taskset -c "$BUILD_AFFINITY" "$INDEX_BIN" \
      "$BASE_FVECS" "$MAX_DEGREE" "$EF_CONSTRUCTION" "$INDEX_FILE" \
      "$METRIC" "$QUANTIZATION_BITS" "$INIT" "$BUILD_THREADS"
    echo "[index] 构建完成 -> $INDEX_FILE"
  fi
}

# =====================================================================
# 阶段 2b：encode_range —— 生成范围谓词 bin + 范围 GT（+ 增强臂旋转数据集）
#   range_dataset.py 始终运行（range_hardgate / enhanced 都依赖其产物）；
#   encode_dataset_range.py 仅在增强臂（RANGE_QUERY=on）时运行（生成旋转 base/query）。
# =====================================================================
run_encode_range() {
  echo "========== [2b] encode_range =========="
  # 1) 生成范围谓词 bin（Nq x 2A）与范围 GT（复用现有 base_attr 作为真值）
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
  # 2) 增强臂：生成旋转数据集（base 旋转 + query M 子中心）
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
    echo "[encode_range] 非增强臂模式，跳过旋转数据集生成（仅生成 range bin / range GT）。"
  fi
  echo "[encode_range] done."
}

# =====================================================================
# 阶段 2c：encode（属性 FDM 旋转 base/query，移植自 diskann-fdm）
#   仅 inline 模式需要：对原始 base/query 施加 FDM/CDMA 相位叠加编码
#   （encode_fdm，每维对承载全部 A 属性相位码），产出 FDM 旋转向量供索引与遍历。
#   旋转保距，故原始空间 GT 与精确重排仍有效；产物按 ac 隔离。幂等。
# =====================================================================
run_encode() {
  if [[ "$MODE" != "inline" ]]; then
    return 0
  fi
  if [[ -f "$FDM_BASE" && -f "$FDM_QUERY" ]]; then
    echo "[encode] FDM base/query 已存在，跳过: $FDM_DIR"
    return 0
  fi
  mkdir -p "$FDM_DIR"
  local IVE_UTILS="$SCRIPT_DIR/scripts"
  echo "========== [2c] encode (属性 FDM 旋转 base/query) =========="
  echo "[encode] A=$ATTR_DIM C=$ATTR_CARD seed=1234 -> $FDM_DIR"
  PYTHONPATH="$IVE_UTILS:${PYTHONPATH:-}" "$PY" "$SCRIPT_DIR/scripts/encode_attr_fdm.py" \
      --ds "$DATA_DIR" --out "$FDM_DIR" \
      --d "$D" --A "$ATTR_DIM" --enc-seed 1234 \
      --base-name "${DATASET}_base.fvecs" \
      --query-name "${DATASET}_query.fvecs" \
      --base-attr-name "base_attr${ATTR_SUFFIX}.bin" \
      --query-attr-name "query_attr${ATTR_SUFFIX}.bin" \
      --groundtruth-name "${DATASET}_groundtruth${ATTR_SUFFIX}.ivecs" \
      --fdm-base-name "${DATASET}_base.fvecs" \
      --fdm-query-name "${DATASET}_query.fvecs"
  echo "[encode] done -> $FDM_BASE"
}

# =====================================================================
# 阶段 4：search —— 运行 symqg_querying 扫描 ef，输出 [RESULT] 日志
#   参数: index query.fvecs gt.ivecs [topk] [efs_csv] [algo] [rounds] [threads]
#         [-b base_attr -a query_attr -A attr_dim -r]
#         [--range_m M --orig_base base.fvecs --orig_query query.fvecs]
# =====================================================================
run_search() {
  echo "========== [4/4] search (symqg_querying) =========="
  run_encode   # inline 模式确保属性旋转 query 存在（幂等）
  [[ -x "$QUERY_BIN" ]] || { echo "[error] 找不到 $QUERY_BIN，请先运行 build" >&2; exit 1; }
  [[ -f "$INDEX_FILE" ]] || { echo "[error] 找不到索引 $INDEX_FILE，请先运行 index" >&2; exit 1; }

  local attr_args=()
  local extra_args=()
  case "$MODE" in
    enhanced)
      attr_args=(-b "$BASE_ATTR" -a "$RANGE_BIN" -A "$ATTR_DIM")
      extra_args=(--range_m "${RANGE_M:-1}" --orig_base "$BASE_FVECS" --orig_query "$QUERY_FVECS")
      echo "[search] 增强臂(旋转+M子中心+重排闸门): dataset=$EFF_DATASET range_m=${RANGE_M:-1}"
      ;;
    range_hardgate)
      attr_args=(-b "$BASE_ATTR" -a "$RANGE_BIN" -A "$ATTR_DIM" -r)
      echo "[search] -r 硬闸门 baseline(range): range_bin=$RANGE_BIN"
      ;;
    inline)
      attr_args=(-b "$BASE_ATTR" -a "$QUERY_ATTR" -A "$ATTR_DIM")
      if [[ "$RERANK" == "1" ]]; then
        attr_args+=(--orig_base "$EXACT_BASE" --orig_query "$EXACT_QUERY")
      fi
      echo "[search] 属性行间等值过滤(inline): base=$BASE_ATTR query=$QUERY_ATTR  rerank=$RERANK"
      ;;
    pure)
      echo "[search] 未启用属性过滤（纯向量检索）"
      ;;
  esac

  # 增强臂 / -r baseline 依赖 range bin 与 range GT
  if [[ "$MODE" == "enhanced" || "$MODE" == "range_hardgate" ]]; then
    [[ -f "$RANGE_BIN" ]] || { echo "[error] 找不到 range bin $RANGE_BIN，请先运行 encode_range" >&2; exit 1; }
    [[ -f "$GT_FILE" ]] || { echo "[error] 找不到范围真值 $GT_FILE，请先运行 encode_range" >&2; exit 1; }
  fi

  mkdir -p "$(dirname "$LOG_FILE")"
  : > "$LOG_FILE"   # 每次运行前清空历史结果

  echo "[search] index=$INDEX_FILE query=$SEARCH_QUERY gt=$GT_FILE"
  bash /home/yq/clean-env.sh taskset -c "$SEARCH_AFFINITY" env \
      OMP_NUM_THREADS="$SEARCH_THREADS" \
      OPENBLAS_NUM_THREADS="$SEARCH_THREADS" \
      MKL_NUM_THREADS="$SEARCH_THREADS" \
      "$QUERY_BIN" "$INDEX_FILE" "$SEARCH_QUERY" "$GT_FILE" \
      "$K" "$EF_SEARCH" symqg "$ROUNDS" "$SEARCH_THREADS" \
      "${attr_args[@]}" "${extra_args[@]}" \
    2>&1 | tee -a "$LOG_FILE"
  echo "[search] 完成 -> 结果日志: $LOG_FILE"
}

# ---- 主流程：按 STAGES 顺序执行指定阶段 ----
for stage in "${STAGES[@]}"; do
  case "$stage" in
    prepare)      run_prepare ;;
    build)        run_build   ;;
    index)        run_index   ;;
    encode_range) run_encode_range ;;
    encode)       run_encode  ;;
    search)       run_search  ;;
    *)
      echo "[warn] 未知阶段: $stage （可选: prepare / build / index / encode_range / search）" >&2
      ;;
  esac
done

echo "[all done] 指定阶段完成。"
