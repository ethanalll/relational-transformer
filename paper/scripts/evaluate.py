"""Evaluate a strict-loaded paper checkpoint on a fixed held-out corpus."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rt.checkpoints import load_paper_checkpoint
from rt.corpus import digest
from rt.main import PAPER_DIMS, evaluate
from rt.model import RelationalTransformer


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pre-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ctx-len", type=int, default=1024)
    ap.add_argument("--bfs-width", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=4)
    ap.add_argument("--diagnostic", action="store_true")
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    metas = [json.loads(p.read_text()) for p in sorted(Path(args.pre_dir).glob("*/meta.json"))]
    if not metas:
        raise ValueError("Empty evaluation corpus")
    backends = {m["backend"] for m in metas}
    if len(backends) != 1:
        raise ValueError("Mixed cache backends")
    backend = backends.pop()
    if not args.diagnostic and (
        backend != "native" or any(m.get("diagnostic_embeddings", True) for m in metas)
    ):
        raise ValueError("Real evaluation requires native caches and real MiniLM embeddings")
    cfg_path = Path(args.checkpoint).parent / "config.json"
    dims = (
        json.loads(cfg_path.read_text()).get("model", PAPER_DIMS)
        if cfg_path.exists()
        else PAPER_DIMS
    )
    model = RelationalTransformer(**dims)
    load_paper_checkpoint(args.checkpoint, model)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model = model.to(
        device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )
    config = dict(
        batch_size=args.batch_size,
        ctx_len=args.ctx_len,
        bfs_width=args.bfs_width,
        eval_batches=args.eval_batches,
        backend=backend,
    )
    report = evaluate(model, args.pre_dir, config)
    report.update(
        checkpoint_sha256=digest(args.checkpoint),
        corpus=metas,
        config=config,
        diagnostic=args.diagnostic,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            dict(output=str(output), mean_loss=report["mean_loss"], tasks=len(report["tasks"]))
        )
    )


if __name__ == "__main__":
    main()
