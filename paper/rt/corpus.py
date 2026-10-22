"""Lossless modern-RelBench -> paper-RT table export (no generator imports)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_database_manifest(db_dir):
    root = Path(db_dir).resolve()
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    if not manifest.get("tables"):
        raise ValueError(f"No tables in {root}")
    for name in manifest["tables"]:
        if Path(name).name != name or name in (".", ".."):
            raise ValueError(f"Unsafe table name: {name}")
    return manifest


def validate_database(db_dir, manifest):
    root = Path(db_dir)
    frames = {
        name: pq.read_table(root / "db" / f"{name}.parquet").to_pandas()
        for name in manifest["tables"]
    }
    tasks, excluded = [], []
    for name, spec in manifest["tables"].items():
        df = frames[name]
        if not len(df):
            raise ValueError(f"{name}: empty tables are unsupported")
        pk, fk, tc = spec.get("pkey"), spec.get("fkeys") or {}, spec.get("time_col")
        if len(fk) > 5:
            raise ValueError(f"{name}: {len(fk)} outgoing FKs exceeds paper sampler limit 5")
        if pk and (pk not in df or not np.array_equal(df[pk].to_numpy(), np.arange(len(df)))):
            raise ValueError(f"{name}: PK must equal physical row ordinals 0..N-1")
        if tc and (tc not in df or not pd.api.types.is_datetime64_any_dtype(df[tc])):
            raise ValueError(f"{name}: invalid timestamp column {tc}")
        for col, parent in fk.items():
            if parent not in frames or col not in df:
                raise ValueError(f"{name}.{col}: missing FK column or parent {parent}")
            field = pq.read_schema(root / "db" / f"{name}.parquet").field(col)
            vals = df[col].dropna()
            if field.type != pa.int64():
                raise ValueError(f"{name}.{col}: expected nullable int64 FK, got {field.type}")
            parent_pk = manifest["tables"][parent].get("pkey")
            if not parent_pk or np.any((vals < 0) | (vals >= len(frames[parent]))):
                raise ValueError(f"{name}.{col}: invalid parent reference")
        for col in df:
            if col in {pk, tc, *fk}:
                continue
            arrow_type = pq.read_schema(root / "db" / f"{name}.parquet").field(col).type
            supported = (
                pa.types.is_null(arrow_type)
                or pa.types.is_boolean(arrow_type)
                or pa.types.is_string(arrow_type)
                or pa.types.is_large_string(arrow_type)
                or pa.types.is_timestamp(arrow_type)
                or arrow_type
                in [pa.int16(), pa.int32(), pa.int64(), pa.uint32(), pa.float32(), pa.float64()]
            )
            if not supported:
                raise ValueError(f"{name}.{col}: unsupported legacy feature type {arrow_type}")
            s = df[col].dropna()
            if pd.api.types.is_numeric_dtype(s) and not np.isfinite(s.to_numpy(dtype=float)).all():
                raise ValueError(f"{name}.{col}: infinite numeric feature")
            reason = None
            if not len(s):
                reason = "all_missing"
            elif s.nunique() < 2:
                reason = "constant"
            elif pd.api.types.is_bool_dtype(df[col]):
                kind = "clf"
            elif pd.api.types.is_numeric_dtype(df[col]):
                kind = "reg"
            else:
                reason = "context_only_type"
            if reason:
                excluded.append(dict(table=name, column=col, reason=reason))
            else:
                tasks.append(
                    dict(table_name=name, target_column=col, task_type=kind, columns_to_drop=[])
                )
    return tasks, excluded


def write_legacy_table(source, destination, table_spec):
    table = pq.read_table(source)
    md = dict(table.schema.metadata or {})
    md.update(
        {
            b"pkey_col": json.dumps(table_spec.get("pkey")).encode(),
            b"fkey_col_to_pkey_table": json.dumps(table_spec.get("fkeys") or {}).encode(),
            b"time_col": json.dumps(table_spec.get("time_col")).encode(),
        }
    )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table.replace_schema_metadata(md), destination)
    if not table.equals(pq.read_table(destination), check_metadata=False):
        raise ValueError(f"Export changed table values: {source}")


def export_legacy_database(db_dir, output_dir):
    root, out = Path(db_dir).resolve(), Path(output_dir).resolve()
    if out == root or root in out.parents or out in root.parents:
        raise ValueError("Export destination must be separate from source database")
    manifest = load_database_manifest(root)
    tasks, excluded = validate_database(root, manifest)
    hashes = {
        str(p.relative_to(root)): digest(p)
        for p in [root / "manifest.yaml", *sorted((root / "db").glob("*.parquet"))]
    }
    fingerprint = hashlib.sha256(
        json.dumps(
            dict(
                tables=manifest["tables"],
                files={k: v for k, v in hashes.items() if k.endswith(".parquet")},
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    record = dict(
        name=root.name,
        source=str(root),
        fingerprint=fingerprint,
        source_files=hashes,
        tables=manifest["tables"],
        tasks=tasks,
        excluded=excluded,
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh output directory")
    out.mkdir(parents=True)
    for name, spec in manifest["tables"].items():
        write_legacy_table(root / "db" / f"{name}.parquet", out / "db" / f"{name}.parquet", spec)
    (out / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def export_legacy_corpus(corpus_path, output_dir):
    root, out = Path(corpus_path).resolve(), Path(output_dir).resolve()
    if (root / "manifest.yaml").exists():
        databases = [root]
    elif (root / "corpus.json").exists():
        corpus = json.loads((root / "corpus.json").read_text())
        if not corpus.get("complete", True):
            raise ValueError("Generation corpus is incomplete")
        names = [item["name"] for item in corpus["databases"]]
        if any(Path(n).name != n or n in (".", "..") for n in names):
            raise ValueError("Unsafe database name in corpus")
        databases = [root / name for name in names]
    else:
        raise ValueError("Expected a database manifest.yaml or a generated corpus.json")
    if len({p.name for p in databases}) != len(databases):
        raise ValueError("Duplicate database identities")
    records = [export_legacy_database(p, out / p.name) for p in databases]
    (out / "corpus.json").write_text(
        json.dumps(dict(format="paper-export-v1", databases=records), indent=2) + "\n"
    )
    return records
