"""Portable training and checkpoint primitives; the renderer stays external."""
import copy
import hashlib
import json
import os
from pathlib import Path
import torch
from .losses import BCEPolicy, objective


def forward_batch(model, batch, config):
    device = next(model.parameters()).device
    fixed, positive, negative = [batch[k].to(device) for k in ("fixed", "positive", "negative")]
    if fixed.shape[-2:] != (config["resolution"], config["resolution"]):
        raise ValueError("Provider image resolution differs from configuration")
    dp, dn = model(fixed, positive), model(fixed, negative)
    policy = BCEPolicy(**config["bce"]) if config.get("bce") is not None else None
    losses = objective(dp, dn, batch["mtre_mm"], margin=config["margin"],
                       alpha=config.get("alpha", .001), cap_mm=config.get("cap_mm", 10.), bce=policy)
    return losses, dp, dn


def train_epoch(model, optimizer, batches, config):
    model.train()
    total, samples, updates = {}, 0, 0
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        values, dp, _ = forward_batch(model, batch, config)
        if not torch.isfinite(values["loss"]):
            raise RuntimeError("Nonfinite loss")
        values["loss"].backward()
        for parameter in model.parameters():
            if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
                raise RuntimeError("Missing or nonfinite trainable gradient")
        optimizer.step()
        for key, value in values.items():
            total[key] = total.get(key, 0.) + float(value.detach())*len(dp)
        samples += len(dp)
        updates += 1
    if not samples:
        raise ValueError("Empty training epoch")
    return {**{k: v/samples for k, v in total.items()}, "samples": samples, "updates": updates}


@torch.no_grad()
def validate(model, batches, config):
    model.eval()
    total, count, rows = {}, 0, []
    for batch in batches:
        values, dp, dn = forward_batch(model, batch, config)
        for key, value in values.items():
            total[key] = total.get(key, 0.) + float(value)*len(dp)
        count += len(dp)
        errors = torch.as_tensor(batch["mtre_mm"]).reshape(-1).tolist()
        rows.extend(dict(d_pos=float(p), d_neg=float(n), mtre_mm=float(e)) for p, n, e in zip(dp, dn, errors))
    if not count:
        raise ValueError("Empty validation set")
    return {k: v/count for k, v in total.items()}, rows


def _cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(v) for v in value)
    return copy.deepcopy(value)


def save_checkpoint(path, model, optimizer, epoch, config, identity):
    """Native safe-loadable format; never overwrite a previous checkpoint."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format_version=1, model=_cpu(model.state_dict()),
                   optimizer_by_name={n: _cpu(optimizer.state[p]) for n, p in model.named_parameters() if p in optimizer.state},
                   epoch=epoch, config=config, identity=identity, torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    temp = path.with_suffix(".tmp")
    with temp.open("xb") as f:
        torch.save(payload, f)
    os.replace(temp, path)


def load_checkpoint(path, model, optimizer=None):
    """Load only trusted FluoroQA native checkpoints with PyTorch safe loading.

    Name-based optimizer transfer retains trained head moments when a tail is
    newly unfrozen. New tail parameters have no moments until their first step.
    Configuration/data compatibility must be checked by the caller.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Not a native FluoroQA checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    if optimizer is not None:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and name in payload["optimizer_by_name"]:
                state = copy.deepcopy(payload["optimizer_by_name"][name])
                optimizer.state[parameter] = {k: (v.to(parameter.device) if torch.is_tensor(v) and k != "step" else v) for k, v in state.items()}
        torch.set_rng_state(payload["torch_rng"])
        if payload["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()
