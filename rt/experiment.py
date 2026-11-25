"""Train, resume, and evaluate the PluRel RT for a specific RelRel ablation arm."""

import argparse
import csv
import fcntl
import hashlib
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import r2_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel

from .experiment_data import (
    FORECAST_PROTOCOL,
    MINILM,
    ROOT,
    digest,
    forecast_cache,
    read_json,
    write_json,
)
from .model import RelationalTransformer

DIMS = {"num_blocks": 12, "d_model": 256, "d_text": 384, "num_heads": 8, "d_ff": 1024}
ARMS = {
    "factorial": (
        "local-original",
        "local-quantile",
        "global-original",
        "global-quantile",
    ),
    "sampler": ("family", "direct"),
}


def build_model(device):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return RelationalTransformer(**DIMS).to(device=device, dtype=dtype)


def load_weights(model, state):
    if "model" in state:
        state = state["model"]
    for _ in range(2):
        for prefix in ("module.", "_orig_mod."):
            if state and all(name.startswith(prefix) for name in state):
                state = {name[len(prefix) :]: value for name, value in state.items()}
    model.load_state_dict(state, strict=True)


def batch_for(model, batch):
    batch.pop("true_batch_size")
    parameter = next(model.parameters())
    return {
        name: tensor.to(
            device=parameter.device,
            dtype=parameter.dtype if tensor.is_floating_point() else tensor.dtype,
        )
        for name, tensor in batch.items()
    }


def dataset(
    path,
    tasks,
    split,
    config,
    *,
    shuffle_seed,
    rank=0,
    world_size=1,
    forecast_policies=None,
):
    from .data import RelationalDataset

    converted = [(db, table, col, split, drops) for db, table, col, drops, _ in tasks]
    data = RelationalDataset(
        tasks=converted,
        batch_size=config["batch"] // world_size,
        seq_len=config["context"],
        rank=rank,
        world_size=world_size,
        max_bfs_width=config["width"],
        embedding_model=MINILM,
        d_text=384,
        seed=shuffle_seed,
        pre_dir=path,
        forecast_policies=forecast_policies,
    )
    if not len(data):
        raise ValueError(f"No eligible target rows in {path}")
    return data


def run_dir(root, suite, arm, seed, run):
    if Path(run).name != run or run in (".", "..", ""):
        raise ValueError("RUN must be a single directory name")
    return root / "runs" / run / suite / arm / f"seed-{seed}"


def save_state(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args, root, out, device, rank, world_size):
    primary = rank == 0
    layout = {"workers": world_size, "batch_per_worker": args.batch // world_size}
    pre = root / "synthetic/pre" / args.suite / "train" / args.arm
    corpus = pre / "corpus.json"
    manifest = read_json(corpus)
    torch.manual_seed(args.seed)
    data = dataset(
        pre,
        manifest["tasks"],
        "train",
        vars(args),
        shuffle_seed=args.data_seed,
        rank=rank,
        world_size=world_size,
    )
    model = build_model(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.1, fused=device.type == "cuda"
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        total_steps=args.steps,
        pct_start=0.2,
        anneal_strategy="linear",
    )
    config = {
        key: getattr(args, key)
        for key in (
            "arm",
            "suite",
            "steps",
            "seed",
            "data_seed",
            "batch",
            "context",
            "width",
            "compile",
        )
    }
    source = hashlib.sha256()
    for path in sorted((ROOT / "rt").glob("*.py")):
        source.update(path.read_bytes())
    source.update((ROOT / "rustler/Cargo.lock").read_bytes())
    config.update(
        source_sha256=source.hexdigest(),
        train_corpus_sha256=digest(corpus),
        torch=str(torch.__version__),
        device_type=device.type,
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        model=DIMS,
        peak_lr=5e-4,
        weight_decay=0.1,
        warmup_fraction=0.2,
    )
    state_path = out / "resume.pt"
    step, epoch, cursor, tokens, slots = 0, 0, 0, 0, 0
    if args.resume and state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if config != state["config"]:
            raise ValueError(
                "Run settings or preprocessing changed since the checkpoint"
            )
        load_weights(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step, epoch, cursor, tokens, slots = (
            state[key] for key in ("step", "epoch", "cursor", "tokens", "slots")
        )
        data.sampler.shuffle_py(epoch)
        del state
    elif args.resume:
        previous = read_json(out / "config.json")
        previous.pop("last_launch", None)
        if config != previous:
            raise ValueError("Run settings or preprocessing changed")
        if primary:
            print("No checkpoint yet. Restarting from its recorded seed.", flush=True)
    if primary:
        if not args.resume:
            out.mkdir(parents=True, exist_ok=False)
        write_json(out / "config.json", {**config, "last_launch": layout})
        print(
            f"Training on {world_size} {device.type.upper()} worker(s), global batch {args.batch}, "
            f"batch per worker {layout['batch_per_worker']}",
            flush=True,
        )
    forward = model
    if world_size > 1:
        forward = DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None
        )
    if args.compile:
        forward = torch.compile(forward, dynamic=False)
    while step < args.steps:
        if cursor >= len(data):
            epoch += 1
            cursor = 0
            data.sampler.shuffle_py(epoch)
        batch = batch_for(model, data[cursor])
        optimizer.zero_grad(set_to_none=True)
        loss, _ = forward(batch)
        valid = batch["masks"].any(dim=1).sum()
        stats = torch.stack(
            (
                loss.detach().double() * valid,
                valid,
                (~batch["is_padding"]).sum(),
            )
        )
        if world_size > 1:
            dist.all_reduce(stats)
        loss_sum, global_valid, batch_tokens = stats.tolist()
        if not np.isfinite(loss_sum) or not global_valid:
            raise FloatingPointError(
                f"Invalid loss or empty target batch at step {step + 1}"
            )
        # DDP averages gradients, so weight ranks by their number of valid contexts.
        if world_size > 1:
            loss = loss * (valid * (world_size / global_valid))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step, cursor = step + 1, cursor + 1
        tokens += int(batch_tokens)
        slots += batch["is_padding"].numel() * world_size
        if primary and (step == 1 or step % 50 == 0 or step == args.steps):
            print(
                f"step {step}/{args.steps} loss={loss_sum / global_valid:.5f} tokens={tokens} slots={slots}",
                flush=True,
            )
        if primary and (step % args.checkpoint_every == 0 or step == args.steps):
            save_state(
                state_path,
                {
                    "config": config,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "launch": layout,
                    "epoch": epoch,
                    "cursor": cursor,
                    "tokens": tokens,
                    "slots": slots,
                },
            )
    if primary:
        save_state(out / "model.pt", model.state_dict())
        write_json(
            out / "training.json",
            {
                "steps": step,
                "nonpadding_tokens": tokens,
                "context_slots": slots,
                **layout,
            },
        )
    if world_size > 1:
        dist.barrier()


def launch_training(args, root, out, device):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.batch % world_size:
        raise ValueError(
            f"Global BATCH={args.batch} must be divisible by {world_size} workers"
        )

    rank = int(os.environ["RANK"]) if world_size > 1 else 0
    if device.type == "cuda":
        if world_size > 1:
            if device.index not in (None, 0):
                raise ValueError(
                    "Select GPUs with GPU_IDS when running multiple workers"
                )
            local_workers = int(os.environ["LOCAL_WORLD_SIZE"])
            visible = torch.cuda.device_count()
            if local_workers > visible:
                raise ValueError(
                    f"{local_workers} workers requested but only {visible} GPUs are visible. "
                    "Check GPUS, GPU_IDS, or the Slurm GPU allocation."
                )
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        if device.index is not None:
            torch.cuda.set_device(device)

    with ExitStack() as resources:
        if world_size > 1:
            dist.init_process_group(
                backend="nccl" if device.type == "cuda" else "gloo",
                device_id=device if device.type == "cuda" else None,
            )
            resources.callback(dist.destroy_process_group)

        error = None
        if rank == 0:
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                lock = resources.enter_context(
                    (out.parent / f".{out.name}.lock").open("a")
                )
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                error = f"A training process is already using {out}"
            except OSError as exc:
                error = str(exc)
            if error is None and out.exists() != args.resume:
                error = (
                    f"No run to resume at {out}"
                    if args.resume
                    else f"Run already exists at {out}. Set RESUME=1 or choose another RUN."
                )
        if world_size > 1:
            status = [error]
            dist.broadcast_object_list(status, src=0, device=device)
            error = status[0]
        if error is not None:
            raise RuntimeError(error)
        train(args, root, out, device, rank, world_size)


@torch.inference_mode()
def score(model, data, kind, limit, *, forecast=False):
    preds, targets = [], []
    weighted_loss = 0.0
    channel = "boolean" if kind == "clf" else "number"
    for index in range(min(len(data), limit) if limit else len(data)):
        raw_batch = data[index]
        actual = raw_batch["true_batch_size"]
        batch = batch_for(model, raw_batch)
        # The paper sampler wraps the last batch. Evaluate each real row once.
        mask = batch["is_targets"][:actual]
        if not (mask.sum(dim=1) == 1).all():
            raise ValueError("Evaluation batch has a missing or duplicated target")
        expected_type = 3 if kind == "clf" else 0
        if not (batch["sem_types"][:actual][mask] == expected_type).all():
            raise ValueError(f"Sampler target type disagrees with manifest ({kind})")
        y = batch[f"{channel}_values"][:actual][mask].flatten().float()
        if forecast:
            # Keep the label for the scorer and pass only a masked placeholder to RT.
            focal = batch["is_targets"]
            if not batch["masks"][focal].all():
                raise ValueError("Forecast query is missing its target mask")
            field = f"{channel}_values"
            batch[field] = batch[field].masked_fill(focal[..., None], 0)
        _, output = model(batch)
        predictions = output[channel][:actual][mask].flatten().float()
        if not torch.isfinite(predictions).all() or not torch.isfinite(y).all():
            raise ValueError("Nonfinite prediction or target")
        if kind == "clf":
            y = (y > 0).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                predictions, y, reduction="sum"
            )
        else:
            loss = torch.nn.functional.huber_loss(predictions, y, reduction="sum")
        preds.extend(predictions.cpu().tolist())
        targets.extend(y.cpu().tolist())
        weighted_loss += float(loss)
    metric = None
    if len(set(targets)) > 1:
        metric = float(
            roc_auc_score(targets, preds) if kind == "clf" else r2_score(targets, preds)
        )
    return {
        "kind": kind,
        "metric": "auroc" if kind == "clf" else "r2",
        "score": metric,
        "loss": weighted_loss / len(targets),
        "examples": len(targets),
    }


def evaluate(args, root, out, device):
    forecast = args.eval != "synthetic"
    if not forecast:
        pre = root / "synthetic/pre/holdout/val/reference"
        corpus, splits = pre / "corpus.json", ("val",)
        if not args.checkpoint:
            train_manifest = read_json(
                root / "synthetic/pre" / args.suite / "train" / args.arm / "corpus.json"
            )
            tunable = {item["seed"] for item in train_manifest["databases"]}
            if tunable & {item["seed"] for item in read_json(corpus)["databases"]}:
                raise ValueError("Training and holdout seeds intersect")
    else:
        pre = forecast_cache(root)
        corpus, splits = pre / f"{args.eval}.json", ("val", "test")
        if not corpus.is_file():
            raise FileNotFoundError(
                f"Missing {corpus}. Run make preprocess-relbench for this DATA and EVAL."
            )
    manifest = read_json(corpus)
    checkpoint = args.checkpoint or out / "model.pt"
    model = build_model(device).eval()
    load_weights(model, torch.load(checkpoint, weights_only=True, map_location="cpu"))
    records = []
    for split in splits:
        for task in manifest["tasks"]:
            data = dataset(
                pre,
                [task],
                split,
                vars(args),
                shuffle_seed=0,
                forecast_policies=manifest["forecast_policies"] if forecast else None,
            )
            record = score(model, data, task[-1], args.eval_batches, forecast=forecast)
            record.update(database=task[0], table=task[1], target=task[2], split=split)
            records.append(record)
            print(
                f"{split} {'/'.join(task[:3])} {record['metric']}={record['score']}",
                flush=True,
            )
    write_json(
        out / f"eval-{evaluation_name(args.eval)}.json",
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "corpus_sha256": digest(corpus),
            "batch_limit": args.eval_batches,
            **({"forecast_protocol": FORECAST_PROTOCOL} if forecast else {}),
            "sampling": {
                key: getattr(args, key) for key in ("batch", "context", "width")
            },
            "results": records,
        },
    )


def evaluation_name(selection):
    return selection if selection == "synthetic" else f"{selection}-{FORECAST_PROTOCOL}"


def summarize(root, args):
    base = root / "runs" / args.run / args.suite
    name = evaluation_name(args.eval)
    paths = sorted(base.glob(f"*/seed-*/eval-{name}.json"))
    if not paths:
        raise FileNotFoundError(f"No evaluations for {args.eval} under {base}")
    details = {}
    protocols = set()
    for path in paths:
        result = read_json(path)
        if (
            args.eval != "synthetic"
            and result.get("forecast_protocol") != FORECAST_PROTOCOL
        ):
            raise ValueError(f"{path} does not use the {FORECAST_PROTOCOL} policy")
        protocols.add(
            (
                result["corpus_sha256"],
                result["batch_limit"],
                tuple(result["sampling"].values()),
            )
        )
        groups = {}
        for row in result["results"]:
            key = row["split"], row["metric"]
            group = groups.setdefault(key, {"scores": [], "losses": []})
            group["losses"].append(row["loss"])
            if row["score"] is not None:
                group["scores"].append(row["score"])
        details[(path.parents[1].name, path.parent.name)] = {
            key: (
                float(np.mean(group["scores"])) if group["scores"] else None,
                float(np.mean(group["losses"])),
                len(group["scores"]),
            )
            for key, group in groups.items()
        }
    if len(protocols) != 1:
        raise ValueError("Evaluations use different corpora or batch limits")
    seeds = sorted({seed for _, seed in details})
    missing = [
        (arm, seed)
        for seed in seeds
        for arm in ARMS[args.suite]
        if (arm, seed) not in details
    ]
    if missing:
        raise ValueError(f"Missing evaluation files for {missing}")
    target = base / f"summary-{name}.csv"
    with target.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "arm",
                "seed",
                "split",
                "metric",
                "mean_score",
                "mean_loss",
                "scored_tasks",
            ]
        )
        for (arm, seed), groups in details.items():
            for (split, metric), values in sorted(groups.items()):
                writer.writerow([arm, seed, split, metric, *values])
    if args.suite == "factorial":
        contrasts = {
            "global_dag": (-0.5, -0.5, 0.5, 0.5),
            "quantile": (-0.5, 0.5, -0.5, 0.5),
            "interaction": (1, -1, -1, 1),
        }
    else:
        contrasts = {"direct_minus_family": (-1, 1)}
    with (base / f"contrasts-{name}.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["seed", "split", "metric", "contrast", "score_delta", "loss_delta"]
        )
        for seed in seeds:
            groups_by_arm = [details.get((arm, seed), {}) for arm in ARMS[args.suite]]
            for split, metric in sorted(
                set.intersection(*(set(g) for g in groups_by_arm))
            ):
                pairs = [g[split, metric] for g in groups_by_arm]
                for label, weights in contrasts.items():
                    score_delta = (
                        sum(w * p[0] for w, p in zip(weights, pairs))
                        if all(p[0] is not None for p in pairs)
                        else None
                    )
                    writer.writerow(
                        [
                            seed,
                            split,
                            metric,
                            label,
                            score_delta,
                            sum(w * p[1] for w, p in zip(weights, pairs)),
                        ]
                    )
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("train", "evaluate", "summarize", "check-model")
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path.home() / "scratch/relrel-rt"
    )
    parser.add_argument("--suite", choices=ARMS, default="factorial")
    parser.add_argument("--arm", default="global-quantile")
    parser.add_argument("--run", default="pilot")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=7629)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--eval", choices=("amazon", "all", "synthetic"), default="amazon"
    )
    parser.add_argument("--eval-batches", type=int)
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    if min(args.steps, args.batch, args.context, args.width, args.checkpoint_every) < 1:
        parser.error("Training and sampling sizes must be positive")
    if args.eval_batches is not None and args.eval_batches < 1:
        parser.error("Evaluation batch limit must be positive")
    if not 0 <= args.data_seed < 2**32 or args.seed < 0:
        parser.error("Seeds must be nonnegative and data seed must fit in 32 bits")
    if args.command == "summarize":
        summarize(root, args)
        return
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(2)
    if args.command == "check-model":
        model = build_model(device)
        assert sum(p.numel() for p in model.parameters()) == 19_189_635
        if args.checkpoint:
            load_weights(
                model,
                torch.load(args.checkpoint, weights_only=True, map_location="cpu"),
            )
        print(
            "PluRel RT, 19,189,635 parameters"
            + ("; strict checkpoint loaded" if args.checkpoint else "")
        )
        return
    if args.arm not in ARMS[args.suite]:
        parser.error(f"Expected one of {', '.join(ARMS[args.suite])} for {args.suite}")
    out = run_dir(root, args.suite, args.arm, args.seed, args.run)
    if args.command == "train":
        if args.steps < 6:
            parser.error("Use at least six steps with the paper's warmup schedule")
        launch_training(args, root, out, device)
    else:
        evaluate(args, root, out, device)


if __name__ == "__main__":
    main()
