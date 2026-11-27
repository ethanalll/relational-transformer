# Relational Transformer (`relational-transformer/`)

PluRel-compatible Relational Transformer (RT) implementation, native Rustler preprocessor/sampler, multi-GPU DDP training harness, and leakage-hardened RelBench evaluation pipeline for the RelRel ablations.

## Scope & Repository Boundaries

- **This repo owns**: RT preprocessing (`DATA/synthetic/pre/`, `DATA/relbench/pre/forecast-v1/`), RelBench downloads (`DATA/relbench/raw/`), model training/resume, evaluation, and run summaries (`DATA/runs/`).
- **Synthetic corpus generation** lives in the sibling `../relrel/` repo (`DATA/synthetic/raw/` and `DATA/synthetic/rt/`). Pass the **same absolute `DATA` root** to both repos.
- **Active vs. legacy entrypoints**: Use `rt/experiment.py`, `rt/experiment_data.py`, and the `Makefile` targets for all RelRel–RT experiments. `rt/main.py`, `rt/embed.py`, and `scripts/example_*.py` are upstream `rt-v1` reference scripts.

## Module Map

| Path | Responsibility |
| --- | --- |
| `rt/model.py` | PluRel-compatible `RelationalTransformer`, `RelationalBlock`, `MaskedAttention` (per-head Q/K `RMSNorm`, `flex_attention` on CUDA / SDPA on CPU), SwiGLU `FFN`. |
| `rt/data.py` | `RelationalDataset` wrapper over native `rustler.Sampler`; resolves `column_index.json` / `table_info.json`, reinterprets `bfloat16` arrays, shapes `f2p_nbr_idxs` to `(B, S, 5)`, and attaches RelBench `forecast_policy` rules. |
| `rt/experiment_data.py` | `python -m rt.experiment_data`: `download` (RelBench datasets/tasks via `RELBENCH_CACHE_DIR`) and `preprocess` (invokes `rustler pre`, encodes `text.json` with `all-MiniLM-L12-v2` into `bfloat16` `.bin`, validates targets/splits, writes `source.json` and manifests). |
| `rt/experiment.py` | `python -m rt.experiment`: `train` (1-GPU or multi-GPU DDP via `torch.distributed.run`, exact `resume.pt`), `evaluate` (RelBench `amazon`/`all` with `forecast-v1` or `synthetic` holdout), `summarize` (`summary-*.csv`), and `check-model`. |
| `rt/tasks.py` | Canonical RelBench task tuples `(db, table, target_col, leakage_cols)`. `rt/experiment_data.py::REL_BENCH_TASKS` uses the 18 forecasting tasks excluding `rel-event`; `EVAL=amazon` selects the 4 `rel-amazon` tasks. |
| `rustler/src/common.rs` | Shared `rkyv` structs: `TableInfo`, `TableType` (`Db`, `Train`, `Val`, `Test`), `Edge`, `Adj`, `SemType` (`Number=0`, `Text=1`, `DateTime=2`, `Boolean=3`), `Node`, `Offsets`. |
| `rustler/src/pre.rs` | Native Parquet preprocessor CLI (`rustler pre`). Applies RelBench schema/type fixups, computes column normalization (optionally filtered by `--stats-cutoff`), and writes `nodes.rkyv`, `offsets.rkyv`, timestamp-sorted `p2f_adj.rkyv`, `text.json`, `text_map.json`, `column_index.json`, `table_info.json`. |
| `rustler/src/fly.rs` | PyO3 `Sampler` (`import rustler`). Mmaps `rkyv` caches and `bfloat16` text embeddings, shuffles target items per epoch, collects BFS contexts (`f2p_ftr` stack + breadth-first `p2f_ftr` with `max_bfs_width` subsampling), enforces `ForecastPolicy`, and packs column-sorted cell sequences. |
| `rustler/src/lib.rs`, `rustler/src/main.rs` | PyO3 module export (`rustler.Sampler`) and standalone `rustler` CLI entrypoint (`pre` / `fly`). |
| `test/test_relbench_forecast.py` | End-to-end leakage test exercising real `rustler pre`, `rustler.Sampler`, `RelationalTransformer`, and `score()` on a multi-table fixture. |

## Core Architecture & Invariants

### 1. PluRel RT Model Compatibility
- Fixed architecture in `rt/experiment.py::DIMS`: `num_blocks=12`, `d_model=256`, `d_text=384`, `num_heads=8`, `d_ff=1024` -> **exactly `19,189,635` parameters** (asserted by `make check-model`).
- `RelationalBlock` uses **only** `attn_types = ("col", "feat", "nbr")` with per-head `q_norm` and `k_norm` (`nn.RMSNorm(d_model // num_heads)`); `"full"` attention is omitted.
- Attention masks:
  - `"feat"`: `(same_node | kv_in_f2p) & pad`
  - `"nbr"`: `q_in_f2p & pad`
  - `"col"`: `same_col_table & pad`
- `load_weights()` strips `module.` / `_orig_mod.` wrappers and calls `load_state_dict(..., strict=True)`. Any parameter-name or shape change breaks strict loading of released PluRel checkpoints (e.g., `synthetic-pretrain_rdb_1024_size_4b.pt`).
- Losses: Huber loss on `number`/`datetime`, BCE-with-logits on `boolean` (`(y > 0).float()`), and masked `text` raises `ValueError`. Sequence loss averages over masked cells per non-empty context.

### 2. Native Rustler & Build Synchronization
- Rust toolchain is pinned to **`1.88.0`** in `rust-toolchain.toml` (edition 2024, `pyo3` `abi3-py312`).
- **`make native` builds two separate artifacts** that must remain in sync whenever `rustler/src/` changes:
  1. Python extension (`maturin develop --uv --release --locked`) loaded by `rt/data.py`.
  2. Standalone CLI binary (`cargo build --release --locked --bin rustler` -> `rustler/target/release/rustler`) invoked by `rt/experiment_data.py`.
- `MAX_F2P_NBRS = 5` is hardcoded in both `rustler/src/fly.rs` (asserted per node) and `rt/data.py` (`.view(-1, self.seq_len, 5)`).
- `SemType` discriminants (`Number=0`, `Text=1`, `DateTime=2`, `Boolean=3`) must stay identical across `common.rs`, `rt/model.py`, and `rt/experiment.py::score()`.
- Text embeddings are stored as raw `bfloat16` (`uint16` bytes) of shape `(len(text.json), 384)` at `text_emb_all-MiniLM-L12-v2.bin`. `rt/experiment_data.py::write_text_embeddings` streams embeddings in 65,536-string chunks (`batch_size=2048` on CUDA, `256` on CPU) directly to disk to prevent GPU VRAM OOM on large RelBench DBs (`rel-amazon`).
- Datetime normalization in `pre.rs` uses Chan's parallel online `(dt_cnt, dt_mean, dt_m2)` variance accumulator to avoid catastrophic floating-point cancellation on nanosecond epoch timestamps.
- In `fly.rs`, `bfs_collect_nodes` always retains the seed node (`!result.is_empty()`), `seq()` prioritizes the focal target cell on wide seed rows, and undated non-forecast seeds (e.g. synthetic Entity tables) can traverse `p2f` edges into dated child tables (`cutoff.or(start_time).is_none_or(...)`).

### 3. RelBench Forecasting Leakage Protocol (`forecast-v1`)
All RelBench preprocessing, sampling, and scoring follow `FORECAST_PROTOCOL = "forecast-v1"`:
- **Normalization (`pre.rs` + `experiment_data.py`)**: `--stats-cutoff` is set to the earliest task validation cutoff for the DB, and `--task-horizon <table=seconds>` is passed per task. Timed `Db` table stats use only rows with `timestamp <= cutoff`; `Train` task tables use only rows with `timestamp + horizon <= cutoff`; `Val` and `Test` task tables inherit `Train` column statistics; rows with null timestamps in dated tables map to `i32::MAX` under forecast mode so they cannot leak across the cutoff.
- **Context traversal (`fly.rs::ForecastPolicy`)**: `cutoff = min(split_cutoff, start_time)`. Applied to **both** `f2p` and `p2f` edges:
  - `Db`: allowed if undated or `timestamp <= cutoff`.
  - `Train`: allowed only if `timestamp + horizon_seconds <= cutoff`.
  - `Val`: allowed only during `test` evaluation (`allow_validation_labels=True`) when `timestamp + horizon_seconds <= cutoff`.
  - `Test`: never traversed into context (no peer test rows).
- **Focal label masking (`experiment.py::score`)**: when `forecast=True`, ground-truth `y` is extracted for scoring, `batch["masks"][focal].all()` is asserted, and focal target values in `batch[f"{channel}_values"]` are zeroed out before calling `model(batch)`.
- **Cache & result isolation**: RelBench caches live under `DATA/relbench/pre/forecast-v1/<db>/`, and evaluation/summary/contrast filenames include `-forecast-v1`. Always rebuild native code (`make native`) and re-run `make preprocess-relbench` if `pre.rs` or `fly.rs` policy logic changes.
- **Fixed-size batch wrapping**: `fly.rs` wraps the final batch modulo `items.len()`; `score()` slices `[:true_batch_size]` so every real evaluation row is scored once.

### 4. Training, Multi-GPU DDP, Resume & Summarization
- **Environment**: Experiments use `.venv` provisioned by `make setup` from `requirements-experiments.txt` (`torch==2.8.0`, Python 3.12). Do not mix `.venv` with the legacy `pyproject.toml` Pixi env (`torch==2.6.0`).
- **Optimization**: `AdamW(lr=5e-4, weight_decay=0.1, fused=cuda)`, `OneCycleLR(max_lr=5e-4, pct_start=0.2, anneal_strategy="linear")`, `clip_grad_norm_(..., 1.0, error_if_nonfinite=True)`, `bfloat16` on CUDA / `float32` on CPU. Minimum `STEPS >= 6`. Default budget: `STEPS=7629`, global `BATCH=128`, `CONTEXT=1024`, `WIDTH=128` (pilot read: `STEPS=1908`).
- **DDP & GPU selection**:
  - `GPUS` is the per-node worker count; global `BATCH` must divide evenly by `GPUS`.
  - Under Slurm: launch one task (`--ntasks=1 --gres=gpu:<N>`) and leave `GPU_IDS` unset.
  - On standalone machines: set `GPU_IDS=0,2` (maps to `CUDA_VISIBLE_DEVICES`); do not pass non-zero `DEVICE` indices when `GPUS > 1`.
  - DDP scales each rank's loss by `valid * (world_size / global_valid)` so rank-averaged gradients match exact global averaging over non-empty contexts.
- **Exact resume (`RESUME=1`)**:
  - Acquires an exclusive flock on `.seed-<seed>.lock`.
  - `resume.pt` stores `config` (including `steps`, `source_sha256` of `rt/*.py` + `rustler/Cargo.lock`, `train_corpus_sha256`, git `revision`, hyperparams), `model`, `optimizer`, `scheduler`, `step`, `epoch`, `cursor`, `tokens`, and `slots`.
  - Any mismatch in settings (including `STEPS`, because `OneCycleLR` depends on `total_steps`), code hash, or preprocessed corpus raises `ValueError`. Changing `GPUS` on resume is allowed if `BATCH % GPUS == 0`.
- **Holdout & summarization (`summarize`)**:
  - Flat corpus layout: `DATA/synthetic/pre/<CORPUS>/` and `DATA/runs/<RUN>/<CORPUS>/seed-<SEED>/` (no `SUITE`/`ARM`/`SPLIT` indirection).
  - Synthetic holdout evaluation (`EVAL=synthetic HOLDOUT=<holdout_corpus>`) rejects overlapping train/holdout database seeds.
  - `summarize` collects all `*/seed-*/eval-<eval>.json` under `DATA/runs/<RUN>/` (optionally filtered by `CORPORA="c1,c2,..."`), verifies matching `corpus_sha256`, `batch_limit`, sampling settings, and `forecast_protocol`, and writes `summary-<eval>.csv`.

## Setup, Commands & Verification

### One-Time Setup
```sh
make setup
make native
make check-model DEVICE=cpu
# Optional strict PluRel checkpoint check:
make check-model DEVICE=cpu CHECKPOINT=/path/to/synthetic-pretrain_rdb_1024_size_4b.pt
```
Set `CARGO_BUILD_JOBS` if Cargo memory/CPU usage must be capped.

### Data, Training, Evaluation & Summary
```sh
export DATA=/shared/scratch/relrel-rt
make download-relbench EVAL=amazon
make preprocess-relbench EVAL=amazon
make preprocess-synthetic CORPUS=local-orig
make preprocess-synthetic CORPUS=global-quant
make preprocess-synthetic CORPUS=holdout-val

make train CORPUS=global-quant SEED=0 RUN=pilot GPUS=4 BATCH=128 STEPS=1908
make evaluate CORPUS=global-quant SEED=0 RUN=pilot EVAL=amazon
make evaluate CORPUS=global-quant SEED=0 RUN=pilot EVAL=synthetic HOLDOUT=holdout-val
make summarize RUN=pilot EVAL=amazon CORPORA="local-orig,global-quant"
```
Use `EMBED_DEVICE=cpu` for CPU preprocessing and `MODEL_PATH=/path/to/all-MiniLM-L12-v2` on offline cluster nodes.

### Verification Checklist
1. **Model & checkpoint parity**:
   ```sh
   make check-model DEVICE=cpu
   ```
2. **RelBench forecast leakage tests** (requires `make native` first; override binary path via `RT_TEST_NATIVE_BIN`):
   ```sh
   .venv/bin/python -m unittest discover -s test
   ```
3. **End-to-end tiny smoke test** (separate `DATA` root; keep CPU smoke runs distinct from cluster runs):
   ```sh
   make preprocess-synthetic DATA=/tmp/relrel-smoke CORPUS=smoke-global EMBED_DEVICE=cpu
   make train DATA=/tmp/relrel-smoke CORPUS=smoke-global SEED=0 RUN=smoke DEVICE=cpu STEPS=10 BATCH=8 CONTEXT=128 CHECKPOINT_EVERY=2
   # Re-run with the exact same STEPS and RESUME=1 (interrupt after step >= CHECKPOINT_EVERY to exercise mid-run resume)
   make train DATA=/tmp/relrel-smoke CORPUS=smoke-global SEED=0 RUN=smoke DEVICE=cpu STEPS=10 BATCH=8 CONTEXT=128 CHECKPOINT_EVERY=2 RESUME=1
   ```
