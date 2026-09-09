from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import pad_topology_targets
from .utils.distributed import reduce_mean


def move_targets(targets: list[dict[str, torch.Tensor]], device: torch.device) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    world_size: int,
    gradient_clip_norm: float,
    amp: bool,
    rank: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    progress = tqdm(loader, disable=rank != 0, desc=f"train {epoch:03d}")
    for batch in progress:
        images = batch["images"].to(device, non_blocking=True)
        frame_valid = batch["frame_valid"].to(device, non_blocking=True)
        targets = move_targets(batch["targets"], device)
        topology = pad_topology_targets(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            outputs = model(images, frame_valid, topology)
            losses = criterion(outputs, targets)
        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        steps += 1
        for name, value in losses.items():
            reduced = reduce_mean(value, world_size)
            totals[name] = totals.get(name, 0.0) + float(reduced)
        if rank == 0:
            progress.set_postfix(loss=f"{totals['loss'] / steps:.4f}")
    return {name: value / max(steps, 1) for name, value in totals.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    amp: bool,
    rank: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    steps = 0
    for batch in tqdm(loader, disable=rank != 0, desc="validate"):
        images = batch["images"].to(device, non_blocking=True)
        frame_valid = batch["frame_valid"].to(device, non_blocking=True)
        targets = move_targets(batch["targets"], device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            outputs = model(images, frame_valid)
            losses = criterion(outputs, targets)
        steps += 1
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(reduce_mean(value, world_size))
    return {name: value / max(steps, 1) for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: dict,
) -> None:
    module = model.module if hasattr(model, "module") else model
    trainable = {name: value.cpu() for name, value in module.state_dict().items() if not name.startswith("encoder.vggt.")}
    torch.save(
        {
            "model": trainable,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def append_metrics(path: Path, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
    record = {"epoch": epoch, "time": time.time(), "train": train_metrics, "validation": val_metrics}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

