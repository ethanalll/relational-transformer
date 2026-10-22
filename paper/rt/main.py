"""Single-GPU paper-model continuation with explicit corpora and strict weights."""

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import r2_score, roc_auc_score

from .checkpoints import load_paper_checkpoint, save_checkpoint
from .corpus import digest
from .data import RelationalDataset
from .model import RelationalTransformer
from .tasks import load_tasks

PAPER_DIMS = dict(num_blocks=12, d_model=256, d_text=384, num_heads=8, d_ff=1024)


def build_optimizer_and_scheduler(model, config):
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("wd", 0.1),
        betas=(0.9, 0.999),
        eps=1e-8,
        fused=next(model.parameters()).device.type == "cuda",
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=config["lr"],
        total_steps=config["steps"],
        pct_start=0.2,
        anneal_strategy="linear",
    )
    return opt, scheduler


def move_batch(batch, device, dtype):
    return {
        k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        for k, v in batch.items()
    }


@torch.inference_mode()
def evaluate(model, pre_dir, config, *, diagnostic=False):
    model.eval()
    results = {}
    for task in load_tasks(pre_dir):
        data = RelationalDataset(
            [task],
            pre_dir,
            batch_size=config["batch_size"],
            ctx_len=config["ctx_len"],
            max_bfs_width=config["bfs_width"],
            seed=0,
            backend=config["backend"],
        )
        losses, preds, labels = [], [], []
        for index in range(min(len(data), config.get("eval_batches", 2))):
            batch = move_batch(
                data[index], next(model.parameters()).device, next(model.parameters()).dtype
            )
            loss, prediction = model(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite evaluation loss")
            key = "boolean" if task.task_type == "clf" else "number"
            mask = batch["is_targets"]
            losses.extend([float(loss)] * len(mask))
            preds.extend(prediction[key][mask].flatten().float().cpu().tolist())
            y = batch[key + "_values"][mask].flatten().float().cpu().numpy()
            labels.extend((y > 0).astype(int).tolist() if task.task_type == "clf" else y.tolist())
        metric_name = "auroc" if task.task_type == "clf" else "r2"
        reason = None
        if len(labels) < 2 or len(set(labels)) < 2:
            metric, reason = None, "insufficient_label_variation"
        else:
            metric = float(
                roc_auc_score(labels, preds) if task.task_type == "clf" else r2_score(labels, preds)
            )
            if not np.isfinite(metric):
                metric, reason = None, "undefined_metric"
        results["/".join(task.key())] = dict(
            task_type=task.task_type,
            loss=float(np.mean(losses)),
            metric_name=metric_name,
            metric=metric,
            count=len(labels),
            undefined_reason=reason,
        )
    model.train()
    return dict(tasks=results, mean_loss=float(np.mean([r["loss"] for r in results.values()])))


def validate_run_config(config):
    required = [
        "train_pre_dir",
        "output_dir",
        "steps",
        "batch_size",
        "ctx_len",
        "bfs_width",
        "lr",
        "backend",
    ]
    for key in required:
        if key not in config or config[key] is None:
            raise ValueError(f"Missing setting {key}")
    if config.get("initialization", "pretrained") not in ("pretrained", "scratch"):
        raise ValueError("initialization must be pretrained or scratch")
    if (
        config["steps"] < 6
        or min(config["batch_size"], config["bfs_width"]) < 1
        or config["ctx_len"] < 2
    ):
        raise ValueError("Need >=6 steps and positive batch/context/width")
    if config.get("initialization", "pretrained") == "pretrained" and not config.get("checkpoint"):
        raise ValueError("Pretrained continuation requires an explicit checkpoint")
    if not config.get("diagnostic", False):
        if config["backend"] != "native":
            raise ValueError("Experiments require the paper native sampler")
        if not config.get("val_pre_dir"):
            raise ValueError("Experiments require unseen-database validation")
    train_fingerprints = set()
    for group in ["train_pre_dir", "val_pre_dir"]:
        root = config.get(group)
        if not root:
            continue
        metas = list(Path(root).glob("*/meta.json"))
        if not metas:
            raise ValueError(f"No preprocessed databases at {root}")
        fingerprints = set()
        for path in metas:
            meta = json.loads(path.read_text())
            if not config.get("diagnostic", False) and meta.get("diagnostic_embeddings", True):
                raise ValueError("Diagnostic/missing embeddings are forbidden for experiments")
            if meta["backend"] != config["backend"]:
                raise ValueError("Backend mismatch")
            if not meta.get("embedding_model"):
                raise ValueError("Missing embedding provenance")
            fingerprints.add(meta["source_fingerprint"])
        if group == "train_pre_dir":
            train_fingerprints = fingerprints
        elif fingerprints & train_fingerprints:
            raise ValueError("Training/validation database overlap")


def run(config):
    config = dict(config)
    validate_run_config(config)
    out = Path(config["output_dir"]).resolve()
    resume_path = config.get("resume")
    if out.exists() and not resume_path:
        raise FileExistsError(f"Refusing to overwrite run {out}")
    out.mkdir(parents=True, exist_ok=True)
    seed = config.get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("torch_threads", 2))
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    dims = config.get("model", PAPER_DIMS)
    model = RelationalTransformer(**dims)
    if config.get("checkpoint"):
        load_paper_checkpoint(config["checkpoint"], model)
        config["checkpoint_sha256"] = digest(config["checkpoint"])
    model = model.to(device=device, dtype=dtype)
    if config.get("compile", False):
        model = torch.compile(model, dynamic=False)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    tasks = load_tasks(config["train_pre_dir"])
    data = RelationalDataset(
        tasks,
        config["train_pre_dir"],
        batch_size=config["batch_size"],
        ctx_len=config["ctx_len"],
        max_bfs_width=config["bfs_width"],
        seed=seed,
        backend=config["backend"],
    )
    steps = epoch = batch_index = tokens = targets = slots = 0
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=True)
        previous = state["config"]
        for key in [
            "train_pre_dir",
            "steps",
            "batch_size",
            "ctx_len",
            "bfs_width",
            "lr",
            "backend",
            "seed",
            "model",
        ]:
            if previous.get(key) != config.get(key):
                raise ValueError(f"Resume changed {key}")
        current_corpora = {
            key: [json.loads(p.read_text()) for p in sorted(Path(config[key]).glob("*/meta.json"))]
            for key in ["train_pre_dir", "val_pre_dir"]
            if config.get(key)
        }
        if previous.get("corpora") != current_corpora:
            raise ValueError("Resume changed corpus fingerprints or preprocessing provenance")
        load_paper_checkpoint(resume_path, getattr(model, "_orig_mod", model))
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        steps, epoch, batch_index, tokens, targets, slots = [
            state[k] for k in ["steps", "epoch", "batch_index", "tokens", "targets", "slots"]
        ]
        data.set_epoch(epoch)
    config["torch_version"] = str(torch.__version__)
    config["dtype"] = str(dtype)
    config["corpora"] = {
        key: [json.loads(p.read_text()) for p in sorted(Path(config[key]).glob("*/meta.json"))]
        for key in ["train_pre_dir", "val_pre_dir"]
        if config.get(key)
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    history = (
        json.loads((out / "history.json").read_text())
        if resume_path and (out / "history.json").exists()
        else []
    )
    if steps >= config["steps"]:
        if (out / "report.json").exists():
            return json.loads((out / "report.json").read_text())
        raise ValueError("Resume is already complete but final report is missing")
    validation = config.get("val_pre_dir")
    if validation:
        initial = evaluate(model, validation, config)
        history.append(dict(step=steps, validation=initial))
    model.train()
    start = time.monotonic()
    for step in range(steps, config["steps"]):
        if batch_index >= len(data):
            epoch += 1
            batch_index = 0
            data.set_epoch(epoch)
        batch = move_batch(data[batch_index], device, dtype)
        batch_index += 1
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite training loss at {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        tokens += int((~batch["is_padding"]).sum())
        targets += int(batch["is_targets"].sum())
        slots += batch["is_padding"].numel()
        entry = dict(
            step=step + 1,
            loss=float(loss.detach()),
            grad_norm=float(grad),
            tokens=tokens,
            targets=targets,
            slots=slots,
            lr=scheduler.get_last_lr()[0],
        )
        history.append(entry)
        if (step + 1) % config.get("eval_every", 128) == 0 or step + 1 == config["steps"]:
            if validation:
                entry["validation"] = evaluate(model, validation, config)
            save_checkpoint(
                out / "resume.pt",
                model,
                optimizer,
                scheduler,
                config=config,
                steps=step + 1,
                epoch=epoch,
                batch_index=batch_index,
                tokens=tokens,
                targets=targets,
                slots=slots,
            )
            (out / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
    final = getattr(model, "_orig_mod", model)
    torch.save(final.state_dict(), out / "model.pt")
    report = dict(
        parameters=sum(p.numel() for p in final.parameters()),
        steps=config["steps"],
        tokens=tokens,
        slots=slots,
        targets=targets,
        seconds=time.monotonic() - start,
        backend=config["backend"],
        diagnostic=config.get("diagnostic", False),
        checkpoint_tested=config.get("checkpoint"),
        last_loss=history[-1]["loss"],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-pre-dir")
    ap.add_argument("--val-pre-dir")
    ap.add_argument("--output-dir")
    ap.add_argument("--checkpoint")
    ap.add_argument("--resume")
    ap.add_argument("--device")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    for k, v in vars(args).items():
        if k != "config" and v is not None:
            config[k] = v
    print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
