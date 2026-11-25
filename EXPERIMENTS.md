# RelRel ablations on PluRel RT

Run **factorial** first. Its four arms estimate the effects of global RSCM, eager quantile decoding, and their interaction. Run **sampler** second to compare the original graph-family sampler with direct sampling at identical realized edge counts. Start with Amazon's four tasks and the common synthetic holdout. Use `EVAL=all` for the paper's 18 RelBench tasks.

| Suite | Arm names |
|---|---|
| `factorial` | `local-original`, `local-quantile`, `global-original`, `global-quantile` |
| `sampler` | `family`, `direct` |

The default training configuration starts from a fresh seed, uses 7,629 updates with 128 contexts per batch and 1,024 cells per context, and records actual nonpadding tokens separately from context slots. For a cheaper first read, set `STEPS=1908` for every arm, yielding about 250M context slots. Use the same steps, batch, context, data seed, and model seeds for every arm in a comparison.

## Shared storage layout

Set `DATA` to the same absolute path in both repositories. Every stage has a fixed location.

| Location under `DATA` | Contents |
|---|---|
| `synthetic/raw/<suite>/<split>/` | Paired generator output, causal diagnostics, and the accepted and rejected seed manifest |
| `synthetic/rt/<suite>/<split>/<arm>/` | RT-ready Parquet and `corpus.json` listing databases and targets |
| `synthetic/pre/<suite>/<split>/<arm>/` | Native Rust caches, MiniLM embeddings, and preprocessed manifest |
| `synthetic/pre/holdout/val/reference/` | Common evaluation corpus from disjoint synthesis seeds |
| `relbench/raw/<db>/db/` | Raw RelBench database Parquet |
| `relbench/raw/<db>/tasks/<task>/` | Official `train.parquet`, `val.parquet`, and `test.parquet` |
| `relbench/pre/forecast-v1/<db>/` | RelBench native caches and embeddings, shared by every run |
| `runs/<run>/<suite>/<arm>/seed-<seed>/` | Config, checkpoint, resume state, token totals, and evaluation JSON |

## One-time setup

On the cluster, make `uv` and `rustup` available. From this RT repository:

```sh
make setup
make native
make check-model DEVICE=cpu
```

The repository pins the Rust toolchain to 1.88.0 for the archived dependency lock. Set `CARGO_BUILD_JOBS` to limit the build's CPU and memory use. `make setup` installs the experiment dependencies into `.venv`. The native build installs the sampler extension and produces the preprocessing executable. Use `NATIVE_BIN=/path/to/rustler` if your Cargo build writes elsewhere. Pass `CHECKPOINT=/path/to/published.pt` to `make check-model DEVICE=cpu` to verify strict compatibility with the published PluRel weights.

## 1. Synthesize and export in RelRel

From the RelRel repo after `pixi install`:

```sh
export DATA=/shared/scratch/relrel-rt
make synthesize SUITE=factorial COUNT=128 WORKERS=8
make export SUITE=factorial

make synthesize SUITE=sampler COUNT=128 WORKERS=8
make export SUITE=sampler

make synthesize SUITE=holdout SPLIT=val COUNT=16 WORKERS=8
make export SUITE=holdout SPLIT=val
```

Use a different `DATA` root and add `TINY=1 COUNT=4` to the synthesis commands for a smoke test. Keep training and holdout seed ranges disjoint if overriding `SEED_START`. RelRel stages publish completed corpora atomically. For a new corpus with the same suite and split, use a new data root.

## 2. Prepare RT and RelBench caches

From this RT repository, with the same exported `DATA` variable:

```sh
make download-relbench EVAL=amazon
make preprocess-relbench EVAL=amazon
make preprocess-synthetic SUITE=factorial
make preprocess-synthetic SUITE=sampler
make preprocess-holdout
```

If your cluster already has the Parquet files, place or link them at the documented `relbench/raw` paths and start with preprocessing. Preprocessing defaults to `cuda:0` for MiniLM. Use `EMBED_DEVICE=cpu` on a CPU allocation or `MODEL_PATH=/shared/models/all-MiniLM-L12-v2` with a local copy of the model. Each ready database cache can be reused on retry. `make preprocess-arm SUITE=factorial ARM=local-original` prepares a single arm for parallel cluster jobs.

RelBench forecasting uses the official validation and test cutoffs. The sampler traverses only rows available at the earlier of the split cutoff and the query time. Validation uses completed training labels as historical context. Test can also use completed validation labels. The query's own outcome goes to the scorer, and peer rows from the evaluated split are excluded from context. Normalization for timed database features uses rows through the validation cutoff.

The `forecast-v1` cache and result names allow archived evaluations to coexist with corrected ones. Rebuild the native extension and run `make preprocess-relbench` before scoring an existing checkpoint with this policy.

After `make native`, run `.venv/bin/python -m unittest discover -s test` to exercise the forecast policy on a local fixture. Set `RT_TEST_NATIVE_BIN` if the native binary lives outside the default build path.

## 3. Train, evaluate, and summarize in RT

Run each arm in its own GPU allocation using the same seed. Repeat the complete suite for each additional model seed. Example:

```sh
make train SUITE=factorial ARM=local-original SEED=0 RUN=pilot STEPS=1908
make evaluate SUITE=factorial ARM=local-original SEED=0 RUN=pilot EVAL=amazon
make evaluate SUITE=factorial ARM=local-original SEED=0 RUN=pilot EVAL=synthetic
```

Evaluate PluRel's released Table 1 synthetic-only checkpoint with the same settings as the ablation checkpoints. The run name keeps its results separate from the ablation summaries.

```sh
make evaluate SUITE=factorial ARM=local-original RUN=plurel-paper EVAL=amazon \
  CHECKPOINT=/path/to/synthetic-pretrain_rdb_1024_size_4b.pt \
  BATCH=128 CONTEXT=1024 WIDTH=128
```

### GPU controls

`GPUS` defaults to 1. Larger values launch one distributed worker per GPU on the same node. `BATCH` is the global batch size and must divide evenly across the workers. For example, `GPUS=4 BATCH=128` gives each GPU 32 contexts per update. Workers share the training schedule. The lead worker writes checkpoints and global metrics.

```sh
# Four GPUs in a Slurm allocation
srun --ntasks=1 --gres=gpu:4 --cpus-per-task=16 \
  make train SUITE=factorial ARM=local-original SEED=0 RUN=pilot GPUS=4 BATCH=128 STEPS=1908

# Submit the same run from the RT repo
sbatch --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=16 --time=08:00:00 \
  --wrap='make train SUITE=factorial ARM=local-original SEED=0 RUN=pilot GPUS=4 BATCH=128 STEPS=1908'

# Select two specific GPUs on a standalone machine
make train SUITE=factorial ARM=local-original RUN=pilot GPUS=2 GPU_IDS=0,2 BATCH=128 STEPS=1908
```

In Slurm, leave `GPU_IDS` unset to use the scheduler's visible GPUs, and add the account and partition flags required by your cluster. On a standalone machine, `GPU_IDS` sets `CUDA_VISIBLE_DEVICES` and the workers use the listed devices in order. For single-GPU runs and evaluation, `DEVICE=cuda:0` selects the first visible GPU. `EMBED_DEVICE` controls preprocessing.

Resume with the same run identifiers, global batch size, and training settings plus `RESUME=1`. Checkpoints preserve model, optimizer, scheduler, and the position in the global sampler. You can change `GPUS` on resume if the global batch divides evenly. GPU count changes can affect floating-point rounding. Run metadata records the latest worker count and batch per worker.

Give smoke runs a distinct name such as `RUN=smoke STEPS=10 BATCH=8 CONTEXT=128 CHECKPOINT_EVERY=2`. Evaluation uses one GPU. Pass the same `CONTEXT` and `WIDTH` as training, and set `EVAL_BATCHES=2` for a short check or leave it unset for the full splits.

Once all arms have been evaluated for a seed:

```sh
make summarize SUITE=factorial RUN=pilot EVAL=amazon
make summarize SUITE=factorial RUN=pilot EVAL=synthetic
make summarize SUITE=sampler RUN=pilot EVAL=amazon
```

The summary and paired contrasts CSVs are written to `runs/<run>/<suite>/`. RelBench result, summary, and contrast filenames end in `-forecast-v1` before the extension. The factorial summary reports the two main effects and their interaction per seed. The sampler contrast reports direct minus family. Each evaluation JSON also records per-task validation/test metrics and masked loss.
