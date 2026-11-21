"""Download and preprocess data for the RelRel experiments."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from .tasks import forecast_clf_tasks, forecast_reg_tasks

ROOT = Path(__file__).resolve().parents[1]
MINILM = "all-MiniLM-L12-v2"
REL_BENCH_TASKS = [
    [*task, kind]
    for kind, group in (("clf", forecast_clf_tasks), ("reg", forecast_reg_tasks))
    for task in group
    if task[0] != "rel-event"
]
NATIVE_FILES = (
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "column_index.json",
    "table_info.json",
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as tmp:
        json.dump(value, tmp, indent=2, allow_nan=False)
        tmp.write("\n")
    Path(tmp.name).replace(path)


def read_json(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relbench_tasks(selection):
    if selection == "amazon":
        return [task for task in REL_BENCH_TASKS if task[0] == "rel-amazon"]
    return REL_BENCH_TASKS


def download_relbench(root, selection):
    raw = root / "relbench/raw"
    os.environ["RELBENCH_CACHE_DIR"] = str(raw)
    from relbench.datasets import download_dataset
    from relbench.tasks import download_task

    tasks = relbench_tasks(selection)
    for db in sorted({task[0] for task in tasks}):
        download_dataset(db)
    for db, table, *_ in tasks:
        download_task(db, table)
        for split in ("train", "val", "test"):
            path = raw / db / "tasks" / table / f"{split}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)


def source_signature(path):
    tables = sorted(path.glob("db/*.parquet")) + sorted(path.glob("tasks/*/*.parquet"))
    if not any(p.parent.name == "db" for p in tables):
        raise FileNotFoundError(f"No Parquet database in {path}")
    return [
        [str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns]
        for p in tables
    ]


def preprocess(root, suite, split, arm, selection, device, native_bin, model_path):
    if suite == "relbench":
        raw, pre = root / "relbench/raw", root / "relbench/pre"
        tasks = relbench_tasks(selection)
        names = sorted({task[0] for task in tasks})
        origin = None
        manifest = None
    else:
        relative = Path(suite) / split / arm
        raw, pre = root / "synthetic/rt" / relative, root / "synthetic/pre" / relative
        manifest = read_json(raw / "corpus.json")
        tasks = manifest["tasks"]
        names = [entry["id"] for entry in manifest["databases"]]
        origin = digest(raw / "corpus.json")
    native_bin = native_bin.expanduser().resolve()
    if not native_bin.is_file():
        raise FileNotFoundError(f"{native_bin} is missing. Run make native.")
    pre.mkdir(parents=True, exist_ok=True)
    import torch
    from sentence_transformers import SentenceTransformer

    model = None
    embedding_source = (
        str(model_path.resolve()) if model_path else f"sentence-transformers/{MINILM}"
    )
    embedding = {"model": embedding_source, "device_type": torch.device(device).type}
    sources = {}
    preprocessor = digest(native_bin)
    for name in names:
        source, output = raw / name, pre / name
        signature = source_signature(source)
        meta = {
            "source": signature,
            "embedding": embedding,
            "paper_cache": 1,
            "preprocessor_sha256": preprocessor,
        }
        required = (*NATIVE_FILES, f"text_emb_{MINILM}.bin")
        if output.exists():
            if read_json(output / "source.json") != meta or not all(
                (output / filename).is_file() for filename in required
            ):
                raise ValueError(
                    f"Stale or incomplete preprocessing in {output}. Use a fresh DATA root."
                )
        else:
            if model is None:
                model = SentenceTransformer(
                    embedding_source,
                    device=device,
                    model_kwargs={
                        "dtype": torch.bfloat16
                        if embedding["device_type"] == "cuda"
                        else torch.float32
                    },
                )
            with tempfile.TemporaryDirectory(dir=pre) as temp:
                stage = Path(temp) / name
                subprocess.run(
                    [
                        str(native_bin),
                        "pre",
                        name,
                        "--input-dir",
                        str(source),
                        "--output-dir",
                        str(stage),
                    ],
                    check=True,
                )
                texts = read_json(stage / "text.json")
                vectors = model.encode(
                    texts, batch_size=256, convert_to_tensor=True
                ).cpu()
                if (
                    vectors.shape != (len(texts), 384)
                    or not torch.isfinite(vectors).all()
                ):
                    raise ValueError(f"Invalid MiniLM embeddings for {name}")
                path = stage / f"text_emb_{MINILM}.bin"
                path.write_bytes(
                    vectors.to(torch.bfloat16)
                    .contiguous()
                    .view(torch.uint16)
                    .numpy()
                    .tobytes()
                )
                write_json(stage / "source.json", meta)
                stage.rename(output)
        cols, tables = (
            read_json(output / key) for key in ("column_index.json", "table_info.json")
        )
        for _, table, target, drops, _ in (t for t in tasks if t[0] == name):
            if not all(f"{col} of {table}" in cols for col in [target, *drops]):
                raise ValueError(
                    f"Missing target or leakage column: {name}/{table}/{target}"
                )
            expected = ("Val", "Test") if suite == "relbench" else ("Db",)
            if any(f"{table}:{part}" not in tables for part in expected):
                raise ValueError(f"Missing table or split for {name}/{table}")
        sources[name] = meta
        print(f"Ready {output}", flush=True)
    ready = f"{selection}.json" if suite == "relbench" else "corpus.json"
    write_json(
        pre / ready,
        {
            "origin": origin,
            "tasks": tasks,
            "sources": sources,
            "databases": manifest["databases"] if manifest else names,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "preprocess"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--suite",
        choices=("factorial", "sampler", "holdout", "relbench"),
        default="relbench",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--arm")
    parser.add_argument("--relbench", choices=("amazon", "all"), default="amazon")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--native-bin", type=Path, default=ROOT / "rustler/target/release/rustler"
    )
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if args.command == "download":
        download_relbench(root, args.relbench)
    else:
        if args.suite != "relbench" and not args.arm:
            parser.error("Synthetic preprocessing requires --arm")
        preprocess(
            root,
            args.suite,
            args.split,
            args.arm,
            args.relbench,
            args.device,
            args.native_bin,
            args.model_path,
        )


if __name__ == "__main__":
    main()
