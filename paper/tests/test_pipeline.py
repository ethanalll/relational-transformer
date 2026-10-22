import json
from pathlib import Path
import sys
import types

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from rt.checkpoints import load_paper_checkpoint
from rt.corpus import export_legacy_database, validate_database, load_database_manifest
from rt.data import RelationalDataset
from rt.embed import embed_database
from rt.main import run, validate_run_config
from rt.model import RelationalTransformer, MaskedAttention
from rt.preprocess import column_stats, preprocess_database
from rt.tasks import load_tasks


def prepare(raw_database, tmp_path, name="raw", shift=0):
    raw = raw_database(name, shift)
    export = tmp_path / "export" / name
    export_legacy_database(raw, export)
    pre = tmp_path / "pre" / name
    preprocess_database(export, pre, backend="portable")
    embed_database(pre, name, diagnostic=True)
    return raw, export, pre


def test_export_preserves_values_and_adds_metadata(raw_database, tmp_path):
    raw, export, pre = prepare(raw_database, tmp_path)
    for source in (raw / "db").glob("*.parquet"):
        a, b = pq.read_table(source), pq.read_table(export / "db" / source.name)
        assert a.equals(b, check_metadata=False)
        for key in [b"pkey_col", b"fkey_col_to_pkey_table", b"time_col"]:
            assert key in b.schema.metadata
    tasks = load_tasks(pre)
    assert len([t for t in tasks if t.table_name == "users"]) == 3
    assert not any(t.target_column in ["always_true", "always_false", "constant"] for t in tasks)
    assert len({t.key() for t in tasks}) == len(tasks)


def test_boolean_and_missing_stats_are_finite():
    assert column_stats([True, True], boolean=True) == (0, 1)
    assert column_stats([False, False], boolean=True) == (0, 1)
    assert column_stats([np.nan]) == (0, 1)
    x = np.array([False, True, False, True], dtype=float)
    m, s = column_stats(x, boolean=True)
    assert np.allclose(
        (x - m) / s, [-np.sqrt(3) / 2, np.sqrt(3) / 2, -np.sqrt(3) / 2, np.sqrt(3) / 2]
    )


def test_invalid_fk_rejected(raw_database):
    raw = raw_database()
    table = pq.read_table(raw / "db/events.parquet").to_pandas()
    table.loc[0, "user_id"] = 100
    table.to_parquet(raw / "db/events.parquet", index=False)
    with pytest.raises(ValueError, match="invalid parent reference"):
        validate_database(raw, load_database_manifest(raw))


def test_static_nullable_context_and_constant_boolean(raw_database, tmp_path):
    raw, export, pre = prepare(raw_database, tmp_path)
    data = RelationalDataset(load_tasks(pre), pre, batch_size=3, ctx_len=32, backend="portable")
    seen = 0
    for i in range(len(data)):
        b = data[i]
        assert (b["is_targets"].sum(1) == 1).all()
        assert (b["timestamps"][~b["is_padding"]] == -(2**31)).all()
        assert torch.isfinite(b["boolean_values"]).all()
        seen += len(b["is_targets"])
    assert seen == len(data.items)  # last partial batch is real, not wrapped duplicates
    index = json.loads((pre / "raw/column_index.json").read_text())
    nodes = torch.load(pre / "raw/portable.pt", weights_only=True)["nodes"]
    values = {c["col"]: c["value"] for n in nodes for c in n["cells"]}
    assert values[index["always_true of users"]] == 1
    assert values[index["always_false of users"]] == 0


def test_strict_loading_and_attention_equation(tmp_path):
    dims = dict(num_blocks=1, d_model=16, d_text=384, num_heads=4, d_ff=32)
    model = RelationalTransformer(**dims)
    path = tmp_path / "model.pt"
    torch.save({"_orig_mod." + k: v for k, v in model.state_dict().items()}, path)
    load_paper_checkpoint(path, model)
    bad = dict(model.state_dict())
    bad.pop(next(iter(bad)))
    torch.save(bad, path)
    with pytest.raises(RuntimeError):
        load_paper_checkpoint(path, model)
    attn = MaskedAttention(16, 4).double()
    x = torch.randn(2, 6, 16, dtype=torch.float64)
    mask = torch.eye(6, dtype=torch.bool).expand(2, -1, -1)
    q = attn.q_norm(attn.wq(x).reshape(2, 6, 4, 4).transpose(1, 2))
    k = attn.k_norm(attn.wk(x).reshape(2, 6, 4, 4).transpose(1, 2))
    v = attn.wv(x).reshape(2, 6, 4, 4).transpose(1, 2)
    weights = (q @ k.transpose(-1, -2) / 2).masked_fill(~mask[:, None], -torch.inf).softmax(-1)
    expected = attn.wo((weights @ v).transpose(1, 2).reshape(2, 6, 16))
    torch.testing.assert_close(attn(x, mask), expected, rtol=1e-10, atol=1e-10)


def test_checkpoint_and_training_all_task_types(raw_database, tmp_path):
    _, _, pre = prepare(raw_database, tmp_path, "train", 0)
    _, _, val = prepare(raw_database, tmp_path, "val", 1)
    dims = dict(num_blocks=1, d_model=16, d_text=384, num_heads=4, d_ff=32)
    result = run(
        dict(
            train_pre_dir=str(pre),
            val_pre_dir=str(val),
            output_dir=str(tmp_path / "run"),
            initialization="scratch",
            diagnostic=True,
            backend="portable",
            model=dims,
            batch_size=3,
            ctx_len=32,
            bfs_width=8,
            lr=1e-4,
            steps=6,
            device="cpu",
            eval_every=6,
            eval_batches=1,
        )
    )
    assert result["steps"] == 6 and result["targets"] == 18
    model = RelationalTransformer(**dims)
    load_paper_checkpoint(tmp_path / "run/resume.pt", model)
    a = {k: v.clone() for k, v in model.state_dict().items()}
    load_paper_checkpoint(tmp_path / "run/model.pt", model)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in a.items())
    history = json.loads((tmp_path / "run/history.json").read_text())
    assert len(history[-1]["validation"]["tasks"]) == 4
    config = json.loads((tmp_path / "run/config.json").read_text())
    config["diagnostic"] = False
    with pytest.raises(ValueError, match="native sampler"):
        validate_run_config(config)


def test_archived_full_model_state_and_forward_parity(raw_database, tmp_path):
    # Execute the untouched source with only CPU attention/layout adapters.
    from rt import model as current
    from rt import layout

    einops = types.ModuleType("einops")
    einops.rearrange = layout.rearrange
    helper = types.ModuleType("einops._torch_specific")
    helper.allow_ops_in_compiled_graph = lambda: None
    previous = {k: sys.modules.get(k) for k in ["einops", "einops._torch_specific"]}
    sys.modules["einops"] = einops
    sys.modules["einops._torch_specific"] = helper
    ns = {"__name__": "paper_reference"}
    try:
        source = Path(__file__).parents[1] / "reference/rt/model.py"
        exec(compile(source.read_text(), str(source), "exec"), ns)
    finally:
        for k, v in previous.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    ns["_make_block_mask"] = lambda mask, **kwargs: mask
    ns["flex_attention"] = lambda q, k, v, block_mask: (
        torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=block_mask[:, None])
    )
    dims = dict(num_blocks=1, d_model=16, d_text=384, num_heads=4, d_ff=32)
    reference = ns["RelationalTransformer"](**dims)
    candidate = current.RelationalTransformer(**dims)
    candidate.load_state_dict(reference.state_dict(), strict=True)
    _, _, pre = prepare(raw_database, tmp_path)
    batch = RelationalDataset(load_tasks(pre), pre, batch_size=2, ctx_len=32, backend="portable")[0]
    la, pa = reference(batch)
    lb, pb = candidate(batch)
    torch.testing.assert_close(la, lb, rtol=0, atol=0)
    for kind in ["number", "boolean"]:
        torch.testing.assert_close(pa[kind], pb[kind], rtol=0, atol=0)


def test_reference_source_hashes_and_full_parameter_count():
    from rt.corpus import digest
    from rt.main import PAPER_DIMS

    root = Path(__file__).parents[1]
    source = json.loads((root / "SOURCE.json").read_text())
    for name, expected in source["reference_sha256"].items():
        assert digest(root / "reference" / name) == expected
    assert sum(p.numel() for p in RelationalTransformer(**PAPER_DIMS).parameters()) == 19189635


def test_interrupted_resume_matches_uninterrupted(raw_database, tmp_path, monkeypatch):
    import rt.main as training

    _, _, pre = prepare(raw_database, tmp_path, "train", 0)
    _, _, val = prepare(raw_database, tmp_path, "val", 1)
    cfg = dict(
        train_pre_dir=str(pre),
        val_pre_dir=str(val),
        output_dir=str(tmp_path / "straight"),
        initialization="scratch",
        diagnostic=True,
        backend="portable",
        model=dict(num_blocks=1, d_model=16, d_text=384, num_heads=4, d_ff=32),
        batch_size=3,
        ctx_len=32,
        bfs_width=8,
        lr=1e-4,
        steps=6,
        device="cpu",
        eval_every=3,
        eval_batches=1,
        seed=2,
    )
    run(cfg)
    interrupted = dict(cfg, output_dir=str(tmp_path / "interrupted"))
    original_save = training.save_checkpoint

    def stop_after_save(*args, **kwargs):
        original_save(*args, **kwargs)
        if kwargs["steps"] == 3:
            raise RuntimeError("simulated preemption")

    monkeypatch.setattr(training, "save_checkpoint", stop_after_save)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        run(interrupted)
    monkeypatch.setattr(training, "save_checkpoint", original_save)
    run(dict(interrupted, resume=str(tmp_path / "interrupted/resume.pt")))
    a = torch.load(tmp_path / "straight/model.pt", weights_only=True)
    b = torch.load(tmp_path / "interrupted/model.pt", weights_only=True)
    assert all(torch.equal(v, b[k]) for k, v in a.items())
    report = run(dict(interrupted, resume=str(tmp_path / "interrupted/resume.pt")))
    assert report["steps"] == 6
