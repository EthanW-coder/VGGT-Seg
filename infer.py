from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from vggt_seg.config import load_config
from vggt_seg.data.image import load_rgb
from vggt_seg.models import VGGTSeg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RGB-only VGGT-Seg inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", choices=("vanilla", "full"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    return parser.parse_args()


def adaptive_image_paths(directory: Path, maximum: int) -> list[Path]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    paths = sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.name),
    )
    if not paths:
        raise ValueError(f"no RGB images found in {directory}")
    count = min(len(paths), maximum)
    indices = np.linspace(0, len(paths) - 1, count, dtype=np.int64)
    return [paths[index] for index in indices]


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = adaptive_image_paths(Path(args.image_dir), config.data.inference_max_frames)
    images = torch.stack(
        [load_rgb(path, config.model.image_size, config.model.patch_size) for path in paths]
    ).unsqueeze(0).to(device)

    model = VGGTSeg(config.model).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    if isinstance(state, dict):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=False)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model(images)

    class_probability = outputs["pred_logits"][0].softmax(dim=-1)[:, :-1]
    scores, labels = class_probability.max(dim=-1)
    mask_probability = outputs["pred_masks"][0].sigmoid()
    masks = mask_probability > config.model.mask_threshold
    mask_quality = (mask_probability * masks).sum(dim=-1) / masks.sum(dim=-1).clamp_min(1)
    scores = scores * mask_quality
    keep = (scores >= args.score_threshold) & masks.any(dim=-1)
    valid = outputs["point_valid"][0]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        points=outputs["points"][0, valid].float().cpu().numpy(),
        colors=outputs["colors"][0, valid].float().cpu().numpy(),
        masks=masks[keep][:, valid].cpu().numpy(),
        mask_probabilities=mask_probability[keep][:, valid].float().cpu().numpy(),
        labels=labels[keep].cpu().numpy(),
        scores=scores[keep].float().cpu().numpy(),
        image_paths=np.asarray([str(path) for path in paths]),
    )
    print(f"saved {int(keep.sum())} instances to {destination}")


if __name__ == "__main__":
    main()
