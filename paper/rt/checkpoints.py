"""Strict paper-state loading; no architecture migration or partial loading."""

from pathlib import Path
import torch


def strip_wrapper_prefixes(state):
    state = dict(state)
    while state and any(all(k.startswith(p) for k in state) for p in ("module.", "_orig_mod.")):
        prefix = next(p for p in ("module.", "_orig_mod.") if all(k.startswith(p) for k in state))
        state = {k[len(prefix) :]: v for k, v in state.items()}
    return state


def load_paper_checkpoint(path, model):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = strip_wrapper_prefixes(state)
    model.load_state_dict(state, strict=True)
    return model


def save_checkpoint(path, model, optimizer=None, scheduler=None, **metadata):
    raw = getattr(model, "_orig_mod", model)
    payload = dict(model={k: v.detach().cpu() for k, v in raw.state_dict().items()}, **metadata)
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
