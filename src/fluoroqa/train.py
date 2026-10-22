"""Explicit training entry point for a user-supplied online data provider."""
import argparse
import importlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from .engine import train_epoch, validate, save_checkpoint, load_checkpoint, file_sha256
from .model import FluoroQA


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", required=True, help="Trusted importable module:factory(config); see docs/data.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--offline-smoke", action="store_true", help="Random model; synthetic provider only, never research weights")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf8"))
    if args.output.exists():
        parser.error("Output already exists; choose a new directory to preserve history")
    if args.offline_smoke and args.provider != "fluoroqa.synthetic:build":
        parser.error("Offline smoke is restricted to the synthetic provider")
    if args.offline_smoke and (config["end_epoch"] != 1 or config["start_epoch"] != 0):
        parser.error("Use configs/smoke.json for offline smoke")
    if config["start_epoch"] and not args.init_checkpoint:
        parser.error("This recipe requires its previous-phase checkpoint")
    if not 0 <= config["start_epoch"] < config["end_epoch"]:
        parser.error("Invalid epoch interval")
    torch.set_num_threads(4)
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    module, factory = args.provider.split(":")
    provider = getattr(importlib.import_module(module), factory)(config)
    identity = provider.identity()
    if not isinstance(identity, str) or not identity:
        raise ValueError("Provider must return a stable nonempty dataset/split/preprocessing identity")
    model = FluoroQA(config["backbone"], pretrained=not(args.offline_smoke or args.init_checkpoint)).to(args.device)
    optimizer = torch.optim.Adam(model.parameter_groups(config["head_lr"], config["tail_lr"]), betas=(.9, .999), eps=1e-8)
    if args.init_checkpoint:
        cp = load_checkpoint(args.init_checkpoint, model, optimizer)
        if cp["epoch"] != config["start_epoch"] or cp["identity"] != identity:
            raise ValueError("Checkpoint epoch or data identity mismatch")
        for key in ("resolution", "seed", "batch_size", "margin", "alpha", "cap_mm"):
            if cp["config"][key] != config[key]:
                raise ValueError(f"Checkpoint protocol differs: {key}")
        del cp
    args.output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    record = dict(config=config, provider=args.provider, identity=identity, started_utc=started,
                  python=platform.python_version(), torch=str(torch.__version__), device=args.device,
                  offline_smoke=args.offline_smoke, config_sha256=file_sha256(args.config),
                  init_sha256=file_sha256(args.init_checkpoint) if args.init_checkpoint else None)
    (args.output/"configuration.json").write_text(json.dumps(record, indent=2), encoding="utf8")
    writer = SummaryWriter(str(args.output/"tensorboard"))
    start = time.perf_counter()
    try:
        with (args.output/"metrics.jsonl").open("x", encoding="utf8") as log:
            for epoch in range(config["start_epoch"]+1, config["end_epoch"]+1):
                train = train_epoch(model, optimizer, provider.train_batches(epoch), config)
                val, rows = validate(model, provider.val_batches(), config)
                result = dict(epoch=epoch, train=train, val=val, seconds=time.perf_counter()-start)
                for split in ("train", "val"):
                    for key, value in result[split].items():
                        writer.add_scalar(f"{split}/{key}", value, epoch)
                for index, group in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"learning_rate/group{index}", group["lr"], epoch)
                log.write(json.dumps(result)+"\n"); log.flush(); print(json.dumps(result), flush=True)
                (args.output/f"val_{epoch:03d}.json").write_text(json.dumps(rows), encoding="utf8")
                save_checkpoint(args.output/f"epoch_{epoch:03d}.pt", model, optimizer, epoch, config, identity)
        state = "completed"
    except Exception as exc:
        state = "failed"
        (args.output/"failure.txt").write_text(repr(exc), encoding="utf8")
        raise
    finally:
        writer.close()
        (args.output/"status.json").write_text(json.dumps(dict(state=locals().get("state", "failed"), ended_utc=datetime.now(timezone.utc).isoformat(), seconds=time.perf_counter()-start)), encoding="utf8")
        if hasattr(provider, "close"):
            provider.close()


if __name__ == "__main__":
    main()
