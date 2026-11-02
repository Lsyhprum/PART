#!/usr/bin/env bash
# ============================================================
# run_ivf.sh — RaBitQ-arp 的 IVF 版构建流程（参考 faiss/rabitq-inline/post）：
#   prepare -> build -> encode(fdm) -> cluster -> index -> verify(smoke)
#
# 与参考库一致：读 config.cfg、环境变量覆盖（网格注入优先）、taskset 亲和、
# clean-env.sh 构建、按 prepare/encode/index/search 阶段组织。
# 编码保持 FDM 不变；索引构建走 C++ ivf_rabitq_indexing（与 symqg 一样编译 C++ 二进制，
# 绕开 Python rabitqlib 绑定）；本轮不改搜索源码，verify 仅做纯向量冒烟。
#
# 用法：
#   bash run_ivf.sh                       # 全流程
#   bash run_ivf.sh prepare build encode cluster index verify
#   bash run_ivf.sh prepare                # 单阶段
# 也可用环境变量覆盖（见下方 DEFAULTS）：DATA_SOURCE / DATASET / D / ATTR_DIM /
#   ATTR_CARD / IVF_CLUSTERS / IVF_BITS / VEC_BASE / VEC_QUERY / ORIG_DATASET_DIR /
#   INDEX_DIR / LOG_DIR / NPROBE_CSV ...
# ============================================================
set -euo pipefail

# ---- 定位脚本所在目录（保证从任意位置调用都正确）----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/config.cfg"

# ---- 环境变量覆盖快照（必须在读取 config.cfg 之前，否则会被 config 覆盖）----
declare -A _ENV_OVERRIDE=()
for _k in DATA_SOURCE DATASET D N NQ VEC_BASE VEC_QUERY ORIG_DATASET_DIR \
          INDEX_DIR LOG_DIR ATTR_DIM ATTR_CARD IVF_CLUSTERS IVF_BITS \
          NPROBE_CSV PY BUILD_THREADS SEARCH_THREADS RABITQLIB_HOME METRIC \
          ENC_SEED FORCE_REBUILD; do
  if [[ -n "${!_k:-}" ]]; then _ENV_OVERRIDE[$_k]="${!_k}"; fi
done

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[error] 找不到配置文件: $CONFIG_FILE" >&2
  exit 1
fi

# ---- 读取配置（仅加载 KEY=VALUE 行，跳过注释/空行，去两端空白与引号）----
while IFS='=' read -r key value; do
  key="$(echo "$key" | xargs)"
  value="$(echo "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/^"//' -e 's/"$//')"
  [[ -z "$key" || "$key" == \#* || -z "$value" ]] && continue
  [[ "$key" =~ ^[A-Z_][A-Z0-9_]*$ ]] || continue
  declare -g "$key=$value"
done < "$CONFIG_FILE"

# ---- 应用环境变量覆盖（config 读取之后，确保覆盖默认/配置值）----
for _k in "${!_ENV_OVERRIDE[@]}"; do declare -g "$_k=${_ENV_OVERRIDE[$_k]}"; done

# ---- DEFAULTS（sift A1_C2 小数据集，可被环境变量覆盖）----
PY="${PY:-/home/yq/miniconda3/envs/faiss_gpu/bin/python}"
BUILD_THREADS="${BUILD_THREADS:-32}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
BUILD_AFFINITY="0-$((BUILD_THREADS-1))"
SEARCH_AFFINITY="0-$((SEARCH_THREADS-1))"
# 关键：RABITQLIB_HOME 必须指向 arp 自身根目录以编译本仓库 IVF 样本；
# 仅当用户通过环境变量显式指定时才尊重（config.cfg 里指向 Graph 的旧值忽略）。
if [[ -n "${_ENV_OVERRIDE[RABITQLIB_HOME]:-}" ]]; then
  RABITQLIB_HOME="${_ENV_OVERRIDE[RABITQLIB_HOME]}"
else
  RABITQLIB_HOME="$SCRIPT_DIR"
fi

# clean-env.sh 可能不在 /home/yq/ 下（已迁移到 _old），解析一个可用路径，缺失则直接调用
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

# ---- 派生路径 ----
BASE_FVECS="$DATA_SOURCE/${DATASET}_base.fvecs"
QUERY_FVECS="$DATA_SOURCE/${DATASET}_query.fvecs"
BASE_ATTR="$DATA_SOURCE/base_attr${ATTR_SUFFIX}.bin"
QUERY_ATTR="$DATA_SOURCE/query_attr${ATTR_SUFFIX}.bin"

FDM_DIR="$DATA_SOURCE/fdm_rotated${ATTR_SUFFIX}"
FDM_BASE="$FDM_DIR/${DATASET}_base.fvecs"
FDM_QUERY="$FDM_DIR/${DATASET}_query.fvecs"
# 纯向量 GT（verify 阶段用 faiss 现算，避免依赖 prepare 的纯 GT 软链）
GT_PURE="$FDM_DIR/${DATASET}_groundtruth_pure.ivecs"

BUILD_DIR="$SCRIPT_DIR/.build_ivf"
# 注意：arp 的 CMakeLists 将运行时输出目录设为 ${PROJECT_SOURCE_DIR}/bin
INDEX_BIN="$SCRIPT_DIR/bin/ivf_rabitq_indexing"
QUERY_BIN="$SCRIPT_DIR/bin/ivf_rabitq_querying"

INDEX_FILE="$INDEX_DIR/ivf_${DATASET}${ATTR_SUFFIX}_b${IVF_BITS}.index"
CLUSTERS_NPZ="$FDM_DIR/clusters_${IVF_CLUSTERS}.npz"
CENTROIDS_FVECS="$FDM_DIR/centroids_${IVF_CLUSTERS}.fvecs"
CIDS_IVECS="$FDM_DIR/cids_${IVF_CLUSTERS}.ivecs"

echo "[info] DATA_SOURCE=$DATA_SOURCE  DATASET=$DATASET  D=$D  A=$ATTR_DIM C=$ATTR_CARD"
echo "[info] RABITQLIB_HOME=$RABITQLIB_HOME  METRIC=$METRIC  IVF_CLUSTERS=$IVF_CLUSTERS  IVF_BITS=$IVF_BITS"
echo "[info] BUILD_THREADS=$BUILD_THREADS (affinity=$BUILD_AFFINITY)  SEARCH_THREADS=$SEARCH_THREADS (affinity=$SEARCH_AFFINITY)"
echo "[info] INDEX_FILE=$INDEX_FILE"

# =====================================================================
# 阶段 1：prepare —— 调用 data/prepare.py 软链向量/属性/真值到 DATA_SOURCE
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
# 阶段 2：build —— out-of-source cmake + 编译 ivf_rabitq_indexing / ivf_rabitq_querying
#   （与 arp run.sh 的 build 同构；RABITQLIB_HOME 已强制为 arp 根目录，
#    单独用 .build_ivf 构建目录，绝不触碰 symqg_* 二进制与 run.sh）
# =====================================================================
run_build() {
  echo "========== [2/6] build (ivf rabitq) =========="
  mkdir -p "$BUILD_DIR"
  $CLEAN_ENV cmake -S "$RABITQLIB_HOME" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRABITQ_BUILD_SAMPLES=ON \
    -DRABITQ_BUILD_TESTS=OFF \
    -DRABITQ_BUILD_PYTHON_BINDINGS=OFF \
    -DRABITQ_ENABLE_NATIVE_OPTIMIZATION=ON
  echo "[build] cmake configured."
  $CLEAN_ENV taskset -c "$BUILD_AFFINITY" cmake --build "$BUILD_DIR" \
    --target ivf_rabitq_indexing ivf_rabitq_querying --parallel "$BUILD_THREADS"
  if [[ ! -x "$INDEX_BIN" || ! -x "$QUERY_BIN" ]]; then
    echo "[error] 编译后未找到 $INDEX_BIN 或 $QUERY_BIN" >&2
    exit 1
  fi
  echo "[build] done. binaries: $INDEX_BIN $QUERY_BIN"
}

# =====================================================================
# 阶段 3：encode —— FDM 旋转（复用脚本/编码逻辑，不改编码语义）
# =====================================================================
run_encode() {
  echo "========== [3/6] encode (fdm) =========="
  mkdir -p "$FDM_DIR"
  ( cd "$SCRIPT_DIR/scripts" && taskset -c "$BUILD_AFFINITY" env -u PYTHONPATH \
      OMP_NUM_THREADS="$BUILD_THREADS" OPENBLAS_NUM_THREADS="$BUILD_THREADS" \
      MKL_NUM_THREADS="$BUILD_THREADS" NUMEXPR_NUM_THREADS="$BUILD_THREADS" \
      "$PY" encode_attr_fdm.py \
      --ds "$DATA_SOURCE" \
      --out "$FDM_DIR" \
      --d "$D" \
      --A "$ATTR_DIM" \
      --enc-seed "$ENC_SEED" \
      --base-name "${DATASET}_base.fvecs" \
      --query-name "${DATASET}_query.fvecs" \
      --base-attr-name "base_attr${ATTR_SUFFIX}.bin" \
      --query-attr-name "query_attr${ATTR_SUFFIX}.bin" \
      --groundtruth-name "${DATASET}_groundtruth${ATTR_SUFFIX}.ivecs" \
      --fdm-base-name "${DATASET}_base.fvecs" \
      --fdm-query-name "${DATASET}_query.fvecs" )
  echo "[encode] done -> $FDM_BASE / $FDM_QUERY"
}

# =====================================================================
# 阶段 4：cluster —— faiss 聚类，产出 clusters.npz + centroids.fvecs + cids.ivecs
#   （扩展后的 faiss_clustering.py；对齐 C++ ivf_rabitq_indexing 输入）
# =====================================================================
run_cluster() {
  echo "========== [4/6] cluster (faiss) =========="
  ( cd "$SCRIPT_DIR/sample/python" && taskset -c "$BUILD_AFFINITY" env -u PYTHONPATH \
      OMP_NUM_THREADS="$BUILD_THREADS" OPENBLAS_NUM_THREADS="$BUILD_THREADS" \
      MKL_NUM_THREADS="$BUILD_THREADS" NUMEXPR_NUM_THREADS="$BUILD_THREADS" \
      "$PY" faiss_clustering.py "$FDM_BASE" "$CLUSTERS_NPZ" \
      --num-clusters "$IVF_CLUSTERS" \
      --metric "$METRIC" \
      --num-threads "$BUILD_THREADS" \
      --centroids-out "$CENTROIDS_FVECS" \
      --cids-out "$CIDS_IVECS" )
  echo "[cluster] done -> $CENTROIDS_FVECS / $CIDS_IVECS"
}

# =====================================================================
# 阶段 5：index —— C++ ivf_rabitq_indexing 建 IVF 索引
#   argv: <data.fvecs> <centroids.fvecs> <cids.ivecs> <total_bits> <index> [metric] [faster_quant]
# =====================================================================
run_index() {
  echo "========== [5/6] index (ivf_rabitq_indexing) =========="
  [[ -x "$INDEX_BIN" ]] || { echo "[error] 找不到 $INDEX_BIN，请先运行 build" >&2; exit 1; }
  [[ -f "$FDM_BASE" ]]  || { echo "[error] 找不到 $FDM_BASE，请先运行 encode" >&2; exit 1; }
  [[ -f "$CENTROIDS_FVECS" && -f "$CIDS_IVECS" ]] || { echo "[error] 找不到聚类产物，请先运行 cluster" >&2; exit 1; }
  if [[ -f "$INDEX_FILE" && "$FORCE_REBUILD" -eq 0 ]]; then
    echo "[index] 索引已存在，跳过构建（使用 -f 或 FORCE_REBUILD=1 强制重建）: $INDEX_FILE"
  else
    if [[ "$FORCE_REBUILD" -eq 1 && -f "$INDEX_FILE" ]]; then
      echo "[index] FORCE_REBUILD，重新构建索引"
    fi
    mkdir -p "$INDEX_DIR"
    $CLEAN_ENV taskset -c "$BUILD_AFFINITY" "$INDEX_BIN" \
      "$FDM_BASE" "$CENTROIDS_FVECS" "$CIDS_IVECS" "$IVF_BITS" "$INDEX_FILE" "$METRIC" false
    echo "[index] 构建完成: $INDEX_FILE"
  fi
}

# =====================================================================
# 阶段 6：verify —— 纯向量冒烟（不改搜索源码）
#   用 faiss 现算纯向量 GT，再跑现有 ivf_rabitq_querying 扫描 nprobe 看 recall
# =====================================================================
gen_pure_gt() {
  local out="$1"
  echo "[verify] 生成纯向量 GT (faiss exact knn) -> $out"
  ( cd "$SCRIPT_DIR/sample/python" && taskset -c "$BUILD_AFFINITY" env -u PYTHONPATH \
      OMP_NUM_THREADS="$BUILD_THREADS" \
      "$PY" -c "
import numpy as np, faiss
from utils import read_fvecs
base = read_fvecs('$FDM_BASE').astype(np.float32)
q = read_fvecs('$FDM_QUERY').astype(np.float32)
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
  [[ -x "$QUERY_BIN" ]]  || { echo "[error] 找不到 $QUERY_BIN，请先运行 build" >&2; exit 1; }
  [[ -f "$FDM_QUERY" ]]  || { echo "[error] 找不到 $FDM_QUERY，请先运行 encode" >&2; exit 1; }
  [[ -f "$INDEX_FILE" ]] || { echo "[error] 找不到 $INDEX_FILE，请先运行 index" >&2; exit 1; }
  mkdir -p "$LOG_DIR"
  if [[ ! -s "$GT_PURE" ]]; then gen_pure_gt "$GT_PURE"; fi
  # 现有通用 ivf_rabitq_querying：内部扫描 nprobe 并打印 nprobe/QPS/recall 表
  $CLEAN_ENV taskset -c "$SEARCH_AFFINITY" env \
    OMP_NUM_THREADS="$SEARCH_THREADS" OPENBLAS_NUM_THREADS="$SEARCH_THREADS" \
    MKL_NUM_THREADS="$SEARCH_THREADS" \
    "$QUERY_BIN" "$INDEX_FILE" "$FDM_QUERY" "$GT_PURE" 2>&1 | tee "$LOG_DIR/ivf_${DATASET}${ATTR_SUFFIX}_b${IVF_BITS}_smoke.log"
  echo "[verify] 冒烟完成 -> $LOG_DIR/ivf_${DATASET}${ATTR_SUFFIX}_b${IVF_BITS}_smoke.log"
}

# ---- 主流程 ----
if [[ $# -eq 0 ]]; then
  run_prepare
  run_build
  run_encode
  run_cluster
  run_index
  run_verify
  echo "[all done] IVF 构建流程完成。"
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
      echo "[warn] 未知阶段: $stage （可选: prepare / build / encode / cluster / index / verify）" >&2
      ;;
  esac
done

echo "[all done] 指定阶段完成。"
