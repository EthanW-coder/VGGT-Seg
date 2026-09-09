from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from vggt_seg.config import load_config
from vggt_seg.data import MultiViewInstanceDataset, collate_scenes
from vggt_seg.engine import append_metrics, save_checkpoint, train_one_epoch, validate
from vggt_seg.models import VGGTSeg
from vggt_seg.models.criterion import VGGTSegCriterion
from vggt_seg.utils.distributed import close_distributed, initialize_distributed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VGGT-Seg")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("vanilla", "full"), required=True)
    return parser.parse_args()


def make_dataset(config, split: str, training: bool) -> MultiViewInstanceDataset:
    manifest = Path(config.data.root) / getattr(config.data, f"{split}_manifest")
    return MultiViewInstanceDataset(
        manifest,
        config.model.num_classes,
        training,
        config.data.train_min_frames,
        config.data.train_max_frames,
        config.data.inference_max_frames,
        config.model.image_size,
        config.model.patch_size,
    )


def make_grad_scaler(device: torch.device, enabled: bool):
    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=enabled and device.type == "cuda")
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.variant)
    device, rank, world_size = initialize_distributed()
    seed = config.train.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_dataset = make_dataset(config, "train", True)
    val_dataset = make_dataset(config, "val", False)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    loader_options = {
        "batch_size": config.train.batch_size_per_gpu,
        "num_workers": config.data.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_scenes,
        "persistent_workers": config.data.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, sampler=train_sampler, shuffle=train_sampler is None, **loader_options)
    val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, **loader_options)

    model = VGGTSeg(config.model).to(device)
    criterion = VGGTSegCriterion(
        config.model.num_classes,
        config.train.lambda_cls,
        config.train.lambda_mask,
        0.0 if args.variant == "vanilla" else config.train.lambda_icr,
        config.train.lambda_bce,
        config.train.lambda_dice,
        config.train.no_object_weight,
        config.model.icr_temperature,
        config.model.icr_radius,
    ).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.train.epochs, eta_min=config.train.min_learning_rate
    )
    scaler = make_grad_scaler(device, config.train.amp)

    output_dir = Path(config.train.output_dir) / config.dataset / args.variant
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    if config.train.resume:
        checkpoint = torch.load(config.train.resume, map_location="cpu", weights_only=False)
        module = model.module if hasattr(model, "module") else model
        module.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1

    try:
        for epoch in range(start_epoch, config.train.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(
                model, criterion, train_loader, optimizer, scaler, device, epoch, world_size,
                config.train.gradient_clip_norm, config.train.amp, rank,
            )
            val_metrics = validate(model, criterion, val_loader, device, world_size, config.train.amp, rank)
            scheduler.step()
            if rank == 0:
                append_metrics(output_dir / "metrics.jsonl", epoch, train_metrics, val_metrics)
                save_checkpoint(
                    output_dir / "checkpoint_last.pt", model, optimizer, scheduler, scaler, epoch, config.to_dict()
                )
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
