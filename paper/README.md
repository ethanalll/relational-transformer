# Paper-compatible RT experiments

Use this isolated subproject, not the repository's modern `src/rt` package.
The archived checkpoint has **19,189,635 parameters**. The exact model source and
legacy Rust stack come from PLuREL v1.0.0, commit
`2a273cfd21933ee4893dfcb862a3edaed45ac665`; originals and hashes are in
`reference/` and `SOURCE.json`. The runtime preserves the original parameter
names, shapes, RMSNorm and attention/loss equations. Runtime model edits only
provide the two tensor permutations without einops and CPU SDPA dispatch using
the same attention masks; forward parity against the archived source is tested.

## Native setup on the H100 machine

Use a separate Python 3.12 environment and Rust toolchain. From **this directory**:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]' 'maturin>=1.7,<2'
cd rustler
maturin develop --release --locked
cd ..
python -c 'import rt, rustler; print(rt.__file__, rustler.__file__)'
```

The `rt` path must point here. `reference/pixi.lock` and `rustler/Cargo.lock`
preserve the original locks; the Python project pins Torch 2.8.0 and bounds its
other dependencies. A newly resolved Python environment is **not** claimed to
be the exact historical environment. Record `pip freeze` with cluster results.
Cargo lock changes, if required by the compiler, must be reviewed separately.

## Generate data in the PLuREL environment

The sibling `plurel/scripts/experiment_gen.py` creates counted accepted corpora,
records rejected >5-FK schemas, and uses paper-scale activity rows and column
counts with this checkout's common mechanism priors. Pass mode, split, seed
range and destination explicitly. Train/val/test seed ranges must not overlap.

For each arm and split, the handoff is a directory containing `corpus.json` and
modern database subdirectories. Keep separate output roots; no input is modified.

## Preprocess each arm and split

```sh
python scripts/prepare_corpus.py --corpus /data/raw/CORPUS \
  --export-dir /data/legacy-export/ARM/SPLIT --pre-dir /data/pre/ARM/SPLIT \
  --backend native --embedding-device cuda:0
```

The adapter preserves table values and adds the three legacy Parquet schema
metadata keys. It validates row-ordinal PKs, nullable int64 FKs and the five-FK
limit, discovers numeric/Boolean tasks, and excludes constant/all-missing targets
without dropping context columns. Native preprocessing retains ordinary paper
normalization; degenerate Boolean columns use raw 0/1, and missing/constant dates
are safe. Native cache files remain the original rkyv layout.

Embeddings use the real frozen `all-MiniLM-L12-v2` (384 dimensions). The first
invocation needs network access or a cached model. Causal sidecars are not inputs.
All export/cache destinations must be fresh to prevent stale corpus reuse.

## Run paired continuation experiments, sequentially on one allocated GPU

First use `configs/cluster_smoke.json` with the same command below: 10 updates,
batch 2, context 128, real published checkpoint, native sampler and real MiniLM.
Run it for both arms before committing the continuation budget. The CPU
validation in this checkout does not replace this cluster check.


Download `paper/synthetic-pretrain_rdb_1024_size_4b.pt` from
`stanford-star/rt-plurel` using huggingface_hub or the HF CLI. Use the SAME local
checkpoint path for both arms. No partial loading or architecture conversion.

```sh
python -m rt.main --config configs/first_continuation.json \
  --checkpoint /data/weights/synthetic-pretrain_rdb_1024_size_4b.pt \
  --train-pre-dir /data/pre/legacy/train --val-pre-dir /data/pre/legacy/val \
  --output-dir /data/runs/legacy-seed0 --device cuda:0
python -m rt.main --config configs/first_continuation.json \
  --checkpoint /data/weights/synthetic-pretrain_rdb_1024_size_4b.pt \
  --train-pre-dir /data/pre/global/train --val-pre-dir /data/pre/global/val \
  --output-dir /data/runs/global-seed0 --device cuda:0
```

Use the cluster scheduler's GPU allocation. The code uses only the specified
CUDA device and has no automatic multi-GPU launch. It currently batches on CPU
synchronously; profile before assigning a large GPU budget. Batch 128 is the
actual batch size, not per-rank. Reduce only by a matched config change if OOM;
gradient accumulation/multi-GPU optimization are not implemented.

The config uses 1,024 updates, batch 128, context 1,024, BFS width 128, peak LR
1e-4 and paper AdamW/linear OneCycle scheduling. Actual nonpadding tokens,
context slots, target count, checkpoint hash, input fingerprints and task-level
metrics are recorded. Evaluation never dispatches by table name. Each run
validates on the configured corpus; also cross-evaluate both models on the SAME
old/new held-out corpora before comparing losses. There is no automatic real
RelBench task-table import in this first synthetic-corpus runner.

Outputs: `model.pt` (canonical weights), `resume.pt` (optimizer/scheduler and
sampler position), `config.json`, `history.json`, `report.json`. To resume, pass
`--resume /data/runs/ARM/resume.pt` with the same config, paths and output dir.
The same seed and sampler epoch/batch position are restored. Resume config and
corpus fingerprints must match. No automatic best-checkpoint or test-set tuning.

## Fixed cross-evaluation and the unchanged baseline

Use `scripts/evaluate.py` on both old/new held-out corpora for all three weights:
the downloaded unchanged checkpoint, legacy-trained `model.pt`, and global-trained
`model.pt`. This also provides the first strict real-checkpoint check on the cluster.

```sh
python scripts/evaluate.py --checkpoint /data/runs/global-seed0/model.pt \
  --pre-dir /data/pre/legacy/val --output /data/eval/global-on-legacy.json --device cuda:0
```

Keep evaluation batch/context/BFS settings equal. Report per-task AUROC/R2 and
loss; never compare a model only on its own generator's validation loss. The
configured evaluation batch limit is a pilot subset, not a full benchmark.

## CPU diagnostic smoke and tests (no downloaded weights required)

```sh
python scripts/smoke.py --plurel-root ../../plurel --output-dir outputs/smoke
python -m pytest tests -q
```

The smoke additionally needs modern PLuREL dependencies (for example, install
`python -m pip install -e ../../plurel` in a diagnostic environment). Normal
training does not import PLuREL or RelBench.

The smoke generates two training DBs and one unseen validation DB **per mode**,
uses the full 19M paper model, checks 10 updates per mode, both target types, and
strict checkpoint reload. It explicitly uses seeded random initial weights,
deterministic diagnostic name embeddings, and a portable Python sampler/cache.
The Python sampler follows paper traversal rules but NOT Rust RNG streams;
portable results are diagnostics and cannot be used for claimed paper experiments.
Production run validation rejects the portable backend and diagnostic embeddings.

This CPU test does NOT establish published-weight compatibility, real MiniLM
behavior, native compilation/sampling, CUDA kernels or H100 throughput. Those
must pass in the cluster environment before long experiments. The model source
and state layout match the paper; the real checkpoint still needs strict loading.

The current global generator exports static databases. Legacy data may include
temporal structure; a whole-pipeline comparison cannot attribute the effect solely
to DAG/hop semantics without matching that source/time behavior separately.
