"""The compiled Rust engine: symbols + preprocess end-to-end."""

from __future__ import annotations

import pytest


def test_extension_symbols():
    import rt._rustler as r

    assert hasattr(r, "Sampler")
    assert hasattr(r, "column_sem_types")
    assert hasattr(r, "preprocess")  # present only when built with --features pre


def test_preprocess_end_to_end(synthetic_dataset, tmp_path):
    from rt._rustler import preprocess

    out = tmp_path / "out"
    preprocess(str(synthetic_dataset), str(out), skip_tasks=True)

    # preprocess writes to <out>/<dataset name>/
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    files = {p.name for p in produced.rglob("*") if p.is_file()}
    # the on-disk format the Sampler consumes
    assert {"nodes.rkyv", "table_info.json", "column_index.json"} <= files


@pytest.mark.parametrize("constant", [False, True])
def test_constant_boolean_remains_a_finite_sampler_target(
    synthetic_dataset, tmp_path, constant
):
    """Constant Boolean features have zero variance, not missing labels."""
    import json

    import numpy as np
    import polars as pl
    from ml_dtypes import bfloat16
    from rt._rustler import Sampler, preprocess

    from rt.data import process_batch

    users_path = synthetic_dataset / "db" / "users.parquet"
    users = pl.read_parquet(users_path).with_columns(pl.lit(constant).alias("active"))
    users.write_parquet(users_path)
    out = tmp_path / "pre"
    preprocess(str(synthetic_dataset), str(out), skip_tasks=True)
    folder = out / "synth"
    texts = json.loads((folder / "text.json").read_text())
    # The text vectors are irrelevant to this numeric/Boolean normalization
    # check; write a valid tensor file for the ordinary native sampler.
    np.zeros((len(texts), 8), dtype=bfloat16).tofile(folder / "text_emb_test.bin")
    columns = json.loads((folder / "column_index.json").read_text())
    info = json.loads((folder / "table_info.json").read_text())["users:Db"]
    sampler = Sampler(
        dataset_tuples=[("synth", "users", info["node_idx_offset"], info["num_nodes"])],
        global_rank=0,
        local_rank=0,
        world_size=1,
        local_ctx_sizes=[16],
        bfs_widths=[4],
        num_walks=0,
        walk_length=0,
        prefer_latest=[False],
        mask_prob_max=0,
        embedding_model="test",
        pre_dir=str(out),
        d_text=8,
        shuffle_seed=0,
        context_seed=0,
        target_columns=[columns["active of users"]],
        columns_to_drop=[[]],
        items_per_task=-1,
        quiet=True,
        ignore_data_errors=False,
        num_prev_skipped=0,
        skip_text_cols=False,
        mmap_populate=False,
        balance_labels=[False],
        timeout_per_item=10,
        ablate_schema_semantics=False,
        vector_db_path=None,
        train_only_fallback=False,
    )
    batch = process_batch(sampler.batch_py(0, 10, 16), d_text=8, bool_as_num=False)
    assert batch["batch_mask"].all()
    assert (batch["is_targets"].sum(-1) == 1).all()
    labels = batch["boolean_values"].squeeze(-1)[batch["is_targets"]]
    assert labels.isfinite().all()
    assert (labels == 0).all()
