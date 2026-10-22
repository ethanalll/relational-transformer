"""Portable diagnostic preprocessing plus launcher for the archived Rust backend.

Portable caches are deliberately separate from rkyv; do not interchange them.
"""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from .corpus import digest


def column_stats(values, *, boolean=False):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    mean = float(x.mean()) if len(x) else 0.0
    std = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    if not np.isfinite(std) or std == 0:
        # Raw 0/1 preserves Boolean identity, including constant True.
        return (0.0 if boolean else mean), 1.0
    return mean, std


def preprocess_database(export_dir, pre_dir, *, backend="native"):
    raw = Path(export_dir).resolve()
    record = json.loads((raw / "export.json").read_text())
    out = Path(pre_dir).resolve() / record["name"]
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite preprocessed cache {out}")
    out.mkdir(parents=True)
    if backend == "native":
        crate = Path(__file__).resolve().parents[1] / "rustler"
        subprocess.run(
            [
                "cargo",
                "run",
                "--locked",
                "--release",
                "--",
                "pre",
                record["name"],
                "--input-dir",
                str(raw),
                "--output-dir",
                str(out),
            ],
            cwd=crate,
            check=True,
        )
    elif backend == "portable":
        _preprocess_portable(raw, out, record)
    else:
        raise ValueError(f"Unknown backend {backend}")
    shutil.copy2(raw / "export.json", out / "export.json")
    meta = dict(
        format="paper-input-v1",
        backend=backend,
        source_fingerprint=record["fingerprint"],
        paper_revision="2a273cfd21933ee4893dfcb862a3edaed45ac665",
        boolean_policy="paper_zscore_constant_raw_01",
        source_hash=digest(raw / "export.json"),
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


def _preprocess_portable(raw, out, record):
    frames = {
        name: pq.read_table(raw / "db" / f"{name}.parquet").to_pandas()
        for name in sorted(record["tables"])
    }
    texts, text_index, column_index, nodes, infos = [], {}, {}, [], {}

    def text_id(value):
        if value not in text_index:
            text_index[value] = len(texts)
            texts.append(value)
        return text_index[value]

    offsets = {}
    for name, df in frames.items():
        offsets[name] = len(nodes)
        tid = text_id(name)
        infos[f"{name}:Db"] = dict(node_idx_offset=len(nodes), num_nodes=len(df))
        for _ in range(len(df)):
            nodes.append(dict(table=tid, timestamp=None, f2p=[], p2f=[], cells=[]))
    dates = []
    for name, df in frames.items():
        for c in df:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                dates.extend(df[c].dropna().astype("datetime64[ns]").astype("int64").tolist())
    dt_mean = float(np.mean(dates)) if dates else 0.0
    dt_std = float(np.std(dates)) if dates else 1.0
    if not np.isfinite(dt_std) or dt_std == 0:
        dt_std = 1.0
    for name, df in frames.items():
        spec = record["tables"][name]
        pk, fks, tc = spec.get("pkey"), spec.get("fkeys") or {}, spec.get("time_col")
        for row in range(len(df)):
            node = nodes[offsets[name] + row]
            if tc and pd.notna(df[tc].iloc[row]):
                node["timestamp"] = int(pd.Timestamp(df[tc].iloc[row]).value // 1_000_000_000)
            for fk, parent in fks.items():
                value = df[fk].iloc[row]
                if pd.notna(value):
                    node["f2p"].append(offsets[parent] + int(value))
        for c in df:
            ci = text_id(f"{c} of {name}")
            column_index[f"{c} of {name}"] = ci
            if c == pk or c in fks:
                continue
            s = df[c]
            if pd.api.types.is_bool_dtype(s):
                kind = 3
                mean, std = column_stats(s.dropna(), boolean=True)
            elif pd.api.types.is_numeric_dtype(s):
                kind = 0
                mean, std = column_stats(s.dropna())
            elif pd.api.types.is_datetime64_any_dtype(s):
                kind, mean, std = 2, dt_mean, dt_std
            else:
                kind, mean, std = 1, 0.0, 1.0
            for row, value in enumerate(s):
                if pd.isna(value):
                    continue
                if kind == 1:
                    value = text_id(str(value))
                elif kind == 2:
                    value = (pd.Timestamp(value).value - mean) / std
                else:
                    value = (float(value) - mean) / std
                if not np.isfinite(value):
                    raise ValueError(f"Nonfinite {name}.{c}")
                nodes[offsets[name] + row]["cells"].append(dict(col=ci, kind=kind, value=value))
    for node_idx, node in enumerate(nodes):
        for parent in node["f2p"]:
            nodes[parent]["p2f"].append(node_idx)
    for node in nodes:
        node["p2f"].sort(
            key=lambda n: (nodes[n]["timestamp"] is not None, nodes[n]["timestamp"] or 0)
        )
    torch.save(dict(nodes=nodes), out / "portable.pt")
    (out / "text.json").write_text(json.dumps(texts))
    (out / "column_index.json").write_text(json.dumps(column_index))
    (out / "table_info.json").write_text(json.dumps(infos))
