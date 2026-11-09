# Attribute Dataset Generation Toolkit — Usage

This folder holds the **batch attribute-dataset generation** toolkit used to produce
the filtered-search datasets and ground-truth (gt) files for the PART paper
experiments. It covers three attribute-filter types: equality, range, and mixed
(one equality + one range, two-condition AND).

| Workflow | Entry point | Purpose |
|---|---|---|
| A. Equality attribute (attribute + filtered gt + queries.json) | `run_gen_eq_attr_gt.sh` | 7 datasets × 4 selectivities, C++ multi-threaded acceleration |
| B. Range attribute (query predicate is a `[lo,hi]` interval) | `run_gen_range_attr_gt.sh` | 3 datasets × 4 selectivities |
| C. Mixed attribute (1 equality + 1 range, two-condition AND) | `run_gen_mix_attr_gt.sh` | default sel=1% (C_eq=10 × C_range=100, w=10) |

---

## 0. Environment setup

- **Python**: any interpreter providing `faiss` (1.12+) and `numpy`. Point to it via
  the `PY` environment variable; the scripts default to `python3`.
- **`PYTHONPATH` shim avoidance**: the entry scripts internally run with
  `env -u PYTHONPATH`; when invoking a single script manually you must add it too:
  ```bash
  env -u PYTHONPATH python3 gen_eq_attr_gt.py --help
  ```
- **C++ compilation**: only `g++` + OpenMP are needed — no GPU / MKL / faiss. The
  entry scripts auto-compile the `.cpp` candidates on first run.

---

## 1. File inventory and responsibilities

All Python scripts depend **only** on the shared file `_dataset_utils.py` (the rest
are stdlib / numpy / faiss).

### Shared library
| File | Role |
|---|---|
| `_dataset_utils.py` | The only shared dependency. faiss brute-force nearest neighbor; fvecs/ivecs read-write; self-describing header attribute-bin read-write and headerless-compatible read (`load_attr`); filtered-gt recomputation `recompute_gt` (equality) / `recompute_gt_range` (range); tie-tolerant top-K legal set `topk_legal_set`; disjoint rotation encoding; `recall_at_k` |
| `config.cfg` | **Unified parameter config**: all runtime parameters of the three entry scripts (interpreter / log dir / dataset root / dataset list / selectivity config / metric map / K / seed / margin / threads / self-check switch). Tune here only; no `.py`/`.sh` edits needed (see section 3) |
| `port_arxiv_em.py` | Ports the real arxiv EM (exact-match) attribute labels 1:1 onto sift / msmarco / openai1536, reusing `compute_eq_attr_gt.cpp` + `recompute_gt` to regenerate attribute-filtered gt; self-contained (no faiss dependency) |

### Workflow A: equality-attribute batch generation
| File | Role |
|---|---|
| `gen_eq_attr_gt.py` | Generate one (A,C) config: random attribute → call C++ for candidates → recompute exact top-K on candidates with `recompute_gt` → self-check. Also importable via `generate_one(...)` |
| `compute_eq_attr_gt.cpp` | OpenMP multi-threaded candidate generation: CSR grouping by attribute signature (radix=C) counting sort; per query take top-(K+M) candidates in the matching subset (double accumulation). Cardinality is passed via `--C` (must match the attribute cardinality) |
| `run_gen_eq_attr_gt.sh` | Batch entry: 7 datasets × 4 configs |

### Workflow B: range-attribute batch generation
| File | Role |
|---|---|
| `gen_range_attr_gt.py` | Same as gen_eq_attr_gt, but the query attribute is an interleaved `[lo_a,hi_a]` interval predicate `(QN,2A)`; candidates have heterogeneous attributes, final selection sorts by distance directly on the candidate subset |
| `compute_range_attr_gt.cpp` | Range candidate generation: per-query interval mask + top-(K+M) in the matching subset (float32 accumulation, consistent with the verifier). No signature grouping, unaffected by cardinality |
| `run_gen_range_attr_gt.sh` | Batch entry: 3 datasets × 4 configs |

### Workflow C: mixed-attribute batch generation (1 equality + 1 range)
| File | Role |
|---|---|
| `gen_mix_attr_gt.py` | Each vector carries one equality attribute + one range attribute (fixed 1+1): base attribute `(N,2)` uint32 `[eq_cat, range_cat]`, query attribute `(QN,3)` int32 `[eq_val, lo, hi]`; filter is a two-condition AND; final selection same as range (direct distance sort on candidates), self-check uses a mixed mask + `topk_legal_set` |
| `compute_mix_attr_gt.cpp` | Mixed candidate generation: per-query two-condition mask (equality match AND interval hit) + top-(K+M) in the matching subset (double accumulation). Structured like the range version, only the mask adds one equality condition |
| `run_gen_mix_attr_gt.sh` | Batch entry: default sel=1% (`MIX_C_EQ=10 × MIX_C_RANGE=100, MIX_W=10`) |

---

## 2. Workflows A/B/C: batch-generate attribute + filtered gt

```bash
cd <repo>/datasets

# Equality attribute: EQ_DATASETS × EQ_CONFIGS from config.cfg (default 7 × 4)
bash run_gen_eq_attr_gt.sh                  # run all datasets in config
bash run_gen_eq_attr_gt.sh sift openai1536  # specify datasets on CLI (override config)

# Range attribute: RANGE_DATASETS × RANGE_CONFIGS from config.cfg (default 3 × 4)
bash run_gen_range_attr_gt.sh

# Mixed attribute (1 equality + 1 range): MIX_DATASETS × (MIX_C_EQ, MIX_C_RANGE, MIX_W)
#   default sel = (1/10)*(10/100) = 1%
bash run_gen_mix_attr_gt.sh
```

**Tune only `config.cfg`** (key descriptions in section 3). Every key also supports a
one-shot environment-variable override (e.g. `MARGIN=256 bash run_gen_eq_attr_gt.sh`;
the env key name matches the config key, e.g. `EQ_MARGIN`).

Manual single-config invocation (drive the `.py` directly, bypassing the entry script):

```bash
python3 gen_eq_attr_gt.py --ds-dir ../sift --base-name sift_base.fvecs --query-name sift_query.fvecs \
    --A 1 --C 10 --metric l2 --K 100 --seed 1000 \
    --cpp-bin ./compute_eq_attr_gt --margin 512 --threads 0
# range: --A 1 --C 100 --w 10 --cpp-bin ./compute_range_attr_gt
# mixed: --C-eq 10 --C-range 100 --w 10 --cpp-bin ./compute_mix_attr_gt
```

Outputs are written under `<DATASET_ROOT>/<ds>/` (resolved relative to this folder by
the entry scripts):

| Workflow | Directory | Attribute files | Ground truth / metadata |
|---|---|---|---|
| A equality | `A{A}_C{C}/` | `base_attr.bin`(N,A uint32), `query_attr.bin`(QN,A int32) | `groundtruth.ivecs`, `queries.json` |
| B range | `A{A}_C{C}_w{W}/` | `base_attr.bin`(N,A uint32), `query_attr_range.bin`(QN,2A int32) | `groundtruth_range.ivecs`, `queries_range.json` |
| C mixed | `MIX_EQ_C{Ceq}_R{Crng}_w{W}/` | `base_attr.bin`(N,2 uint32 `[eq_cat, range_cat]`), `query_attr_mix.bin`(QN,3 int32 `[eq_val, lo, hi]`) | `groundtruth_mix.ivecs`, `queries_mix.json` |

All three additionally create **symlinks** pointing to the dataset-root base/query
vectors (for convenient re-verification); invalid gt slots are `-1`.

The metric map is maintained in `config.cfg`'s `EQ_METRICS` / `RANGE_METRICS` /
`MIX_METRICS` (datasets not listed default to `l2`; msmarco is measured as L2).

---

## 3. config.cfg parameter reference

At startup the entry script reads the `config.cfg` in the same folder
(`KEY=VALUE` format, `#` comments). **Precedence: CLI dataset args > environment
variables > config.cfg > script built-in defaults**. Every key has a built-in
default (matching the current config value), so removing a key does not change
behavior; a missing config file aborts with an error. Relative paths (`LOG_DIR` /
`DATASET_ROOT`) are resolved against this `datasets/` folder.

| Key | Default | Applies to | Description |
|---|---|---|---|
| `PY` | `python3` | all | Python interpreter (must provide faiss + numpy) |
| `LOG_DIR` | `./_log` | all | Log directory (master progress + per-dataset logs) |
| `DATASET_ROOT` | `..` | all | Dataset root; scripts address `<DATASET_ROOT>/<ds>/<ds>_base.fvecs` |
| `NO_VERIFY` | `0` | all | 1 = skip the 256-sample self-check inside the generator |
| `EQ_CPP_BIN` / `RANGE_CPP_BIN` / `MIX_CPP_BIN` | `compute_eq_attr_gt` / `compute_range_attr_gt` / `compute_mix_attr_gt` | each | C++ binary name (relative to this folder; env `CPP_BIN` overrides all) |
| `EQ_K` / `RANGE_K` / `MIX_K` | `100` | each | filtered gt / Recall@K |
| `EQ_SEED` / `RANGE_SEED` / `MIX_SEED` | `1000` | each | attribute random seed (derived independently inside each `gen_*`, three attribute streams are independent) |
| `EQ_MARGIN` / `RANGE_MARGIN` / `MIX_MARGIN` | `512` | each | candidate margin M: candidates = K+M, covering float-boundary ties to ensure the true top-K is included |
| `EQ_THREADS` / `RANGE_THREADS` / `MIX_THREADS` | `0` | each | OpenMP threads, 0 = all cores |
| `EQ_DATASETS` / `RANGE_DATASETS` / `MIX_DATASETS` | 7 / 3 / 7 datasets | each | default dataset list (space-separated; CLI datasets override) |
| `EQ_CONFIGS` | `1 2;1 10;2 10;3 10` | equality | selectivity config: `A C` pairs, semicolon-separated (sel = 1/C^A) |
| `RANGE_CONFIGS` | `1 100 50;1 100 10;2 100 10;3 100 10` | range | selectivity config: `A C w` triples, semicolon-separated (sel = (w/C)^A) |
| `MIX_C_EQ` / `MIX_C_RANGE` / `MIX_W` | `10` / `100` / `10` | mixed | equality cardinality / range cardinality / range width; sel = (1/MIX_C_EQ) × (MIX_W/MIX_C_RANGE), default = 1% |
| `EQ_METRICS` / `RANGE_METRICS` / `MIX_METRICS` | see config file | each | metric map: `ds=metric` space-separated (default l2 if absent) |

Changes take effect immediately. For example, to run only sift with two configs, a
different seed, and no self-check:

```
EQ_DATASETS=sift
EQ_CONFIGS=1 10;2 10
EQ_SEED=777
NO_VERIFY=1
```

At startup the script prints a `[info] config: ...` line summarizing the effective
key parameters; config typos (non-positive-integer, metric not ip|l2, malformed
entries, etc.) abort before running.

---

## 4. C++ / Python division of labor

- `compute_eq_attr_gt` (C++) does only the heavy lifting: per query it takes
  **top-(K+M) candidates** + match counts in the attribute-matching subset, writing
  intermediate files `.cand.int32` / `.count.int64` (deleted after use);
- `gen_eq_attr_gt.py` (Python) does everything else: attribute generation, calling
  C++ (passing `--C` cardinality), exact final top-K selection on candidates,
  writing all artifacts, and self-checking;
- final selection and verification share `_dataset_utils.recompute_gt` /
  `topk_legal_set` — generated gt always passes independent verification.

**After editing a `.cpp` you must manually delete the old binary** (the script only
checks existence, not timestamps):
```bash
rm -f compute_eq_attr_gt compute_range_attr_gt compute_mix_attr_gt   # auto-recompiled on next run
```

---

## 5. Data file format reference

| Format | Layout |
|---|---|
| `.fvecs` | each row `int32 dim + dim×float32` |
| `.ivecs` | each row `int32 dim + dim×int32` (gt invalid slots are `-1`) |
| `.bin` (attribute) | self-describing header: 8 bytes `[N,A]` (int32 little-endian) + `N×A` payload. base attribute `uint32`, query attribute `int32`, range query attribute is `(QN,2A)` interleaved `[lo_a,hi_a]`. Older datasets may be headerless; `load_attr` auto-detects by file size |
| `queries.json` | per-query predicate metadata: `type` (eq/range), attribute index, value/interval, realized selectivity, metric |

---

## 6. Notes

1. **`--C` must match the attribute cardinality**: `compute_eq_attr_gt`'s signature
   radix was originally hardcoded to 10; with `C>10 and A>=2` a signature collision
   causes wrong candidates. Now `gen_eq_attr_gt.py` passes it through automatically.
2. **Match set smaller than K**: `finalize_via_recompute` originally wrote `-1`
   padding slots from `recompute_gt` as negative indices (`c[-1]`) repeatedly. This
   is fixed; production configs have match sets far larger than K, but small
   datasets / high-cardinality configs could trigger it — the self-check catches it.
3. Logs are written to `./_log/` (i.e. this `datasets/_log` folder):
   `gen_eq_attr_gt_master.log` / `gen_range_attr_gt_master.log` are the overall
   progress, `gen_eq_attr_gt_<ds>.log` etc. are per-dataset details. This directory
   is git-ignored.
4. The metric must match the dataset (see the section 2 map).
5. Scripts involving faiss need a GPU environment; `_dataset_utils.py` sets
   `CUDA_VISIBLE_DEVICES` only if it is **not already set** in your environment
   (default `0`), so you can control the device via that environment variable.
