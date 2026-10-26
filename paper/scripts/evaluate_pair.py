"""Compare original and continued weights on fresh global/legacy test corpora."""

import argparse
import json
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    home = Path.home() / "rt-data"
    ap.add_argument("--test-root", type=Path, default=home / "test-v1")
    ap.add_argument("--original-checkpoint", type=Path,
                    default=home / "weights/paper/synthetic-pretrain_rdb_1024_size_4b.pt")
    ap.add_argument("--continued-checkpoint", type=Path,
                    default=home / "runs/global-seed0/model.pt")
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--expected-dbs", type=int, default=64)
    ap.add_argument("--tasks-per-db", type=int, default=4)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--bfs-width", type=int, default=128)
    ap.add_argument("--eval-seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    for key in ["expected_dbs", "tasks_per_db", "eval_batches", "batch_size", "ctx_len", "bfs_width"]:
        if getattr(args, key) < 1:
            ap.error(f"--{key.replace('_', '-')} must be positive")

    import torch
    from rt.checkpoints import load_paper_checkpoint
    from rt.corpus import digest
    from rt.main import PAPER_DIMS, evaluate
    from rt.model import RelationalTransformer
    from rt.tasks import load_tasks, select_validation_tasks

    torch.set_num_threads(4)
    output = args.output_dir or args.test_root / "evaluation"
    checkpoints = {"original": args.original_checkpoint, "continued": args.continued_checkpoint}
    for path in checkpoints.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config = dict(backend="native", batch_size=args.batch_size,
                  eval_batch_size=args.batch_size, ctx_len=args.ctx_len,
                  bfs_width=args.bfs_width, eval_batches=args.eval_batches,
                  eval_tasks_per_db=args.tasks_per_db, eval_seed=args.eval_seed)

    # Check both corpora and every output path before starting GPU work.
    corpora = {}
    for mode in ["global_rscm", "legacy_scm"]:
        pre = args.test_root / "pre" / mode
        metas = sorted(pre.glob("*/meta.json"))
        if len(metas) != args.expected_dbs:
            raise ValueError(f"{mode}: expected {args.expected_dbs} prepared databases, found {len(metas)}")
        provenance = [json.loads(p.read_text()) for p in metas]
        for meta in provenance:
            if (meta.get("backend") != "native"
                    or meta.get("diagnostic_embeddings") is not False
                    or not meta.get("embedding_model")):
                raise ValueError(f"{mode}: native caches and real embeddings are required")
        tasks = select_validation_tasks(load_tasks(pre), args.tasks_per_db, args.eval_seed)
        corpora[mode] = (pre, provenance, [list(task.key()) for task in tasks])
        for name in checkpoints:
            target = output / f"{mode}-{name}.json"
            if target.exists():
                raise FileExistsError(f"{target}; choose a fresh --output-dir")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    for mode, (pre, provenance, task_ids) in corpora.items():
        for name, checkpoint in checkpoints.items():
            print(f"START: {mode}, {name}, {len(task_ids)} tasks", flush=True)
            model = RelationalTransformer(**PAPER_DIMS)
            load_paper_checkpoint(checkpoint, model)
            print(f"Checkpoint loaded strictly: {checkpoint}", flush=True)
            model = model.to(device=device, dtype=dtype).eval()
            result = evaluate(model, pre, config)
            result.update(config=config, checkpoint=str(checkpoint),
                          checkpoint_sha256=digest(checkpoint), corpus=provenance,
                          selected_tasks=task_ids, device=str(device), dtype=str(dtype))
            target = output / f"{mode}-{name}.json"
            target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            print(f"DONE: {mode}, {name}; loss={result['mean_loss']:.5f}; "
                  f"tasks={result['task_count']}; seconds={result['seconds']:.1f}", flush=True)
            for kind, metric in [("clf", "AUROC"), ("reg", "R2")]:
                tasks = [t for t in result["tasks"].values() if t["task_type"] == kind]
                values = [t["metric"] for t in tasks if t["metric"] is not None]
                score = f"{mean(values):.5f}" if values else "undefined"
                print(f"  {metric}: {score}; defined tasks: {len(values)}/{len(tasks)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
