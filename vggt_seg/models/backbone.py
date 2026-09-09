from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass
class EncoderOutput:
    points: Tensor
    colors: Tensor
    confidence: Tensor
    features: list[Tensor]
    shallow_attention: Tensor
    point_valid: Tensor


def _first_transformer_block(dino: nn.Module) -> nn.Module:
    for block in dino.blocks:
        if hasattr(block, "attn"):
            return block
        for nested in block:
            if hasattr(nested, "attn"):
                return nested
    raise RuntimeError("could not locate the first DINOv2 attention block")


class FrozenVGGTEncoder(nn.Module):
    def __init__(
        self,
        repository: str = "facebook/VGGT-1B",
        checkpoint: str | None = None,
        feature_layers: Sequence[int] = (4, 11, 17, 23),
        patch_size: int = 14,
    ) -> None:
        super().__init__()
        try:
            from vggt.models.vggt import VGGT
        except ImportError as error:
            # Allow running from a source checkout before the bundled package is installed.
            import sys

            source_root = Path(__file__).resolve().parents[2] / "vggt"
            if not (source_root / "vggt").is_dir():
                raise ImportError("install the bundled VGGT package with `pip install -e ./vggt`") from error
            sys.path.insert(0, str(source_root))
            import vggt as vggt_namespace

            if hasattr(vggt_namespace, "__path__"):
                vggt_namespace.__path__.append(str(source_root / "vggt"))
            try:
                from vggt.models.vggt import VGGT
            except ImportError as nested_error:
                raise ImportError("install the bundled VGGT package with `pip install -e ./vggt`") from nested_error

        local_checkpoint = Path(checkpoint) if checkpoint else None
        repository_path = Path(repository)
        if local_checkpoint is None and repository_path.is_file():
            local_checkpoint = repository_path
        if local_checkpoint is None and repository_path.is_dir():
            candidate = repository_path / "model.pt"
            if candidate.exists():
                local_checkpoint = candidate

        if local_checkpoint is not None:
            self.vggt = VGGT()
            state = torch.load(local_checkpoint, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            self.vggt.load_state_dict(state, strict=True)
        else:
            self.vggt = VGGT.from_pretrained(repository)
        self.feature_layers = tuple(feature_layers)
        self.patch_size = patch_size
        for parameter in self.vggt.parameters():
            parameter.requires_grad_(False)
        self.vggt.eval()

    def train(self, mode: bool = True) -> "FrozenVGGTEncoder":
        super().train(False)
        self.vggt.eval()
        return self

    @staticmethod
    def _attention_from_tokens(block: nn.Module, tokens: Tensor, register_count: int) -> Tensor:
        attention = block.attn
        value = block.norm1(tokens)
        batch, token_count, _ = value.shape
        qkv = attention.qkv(value).reshape(
            batch, token_count, 3, attention.num_heads, attention.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, _ = qkv.unbind(0)
        query, key = attention.q_norm(query), attention.k_norm(key)
        score = (query[:, :, :1] * attention.scale) @ key.transpose(-2, -1)
        return score[:, :, 0, 1 + register_count :]

    @torch.no_grad()
    def forward(self, images: Tensor, frame_valid: Tensor | None = None) -> EncoderOutput:
        if images.ndim != 5:
            raise ValueError("images must have shape [B, S, 3, H, W]")
        batch, frames, _, height, width = images.shape
        if height % self.patch_size or width % self.patch_size:
            raise ValueError("image dimensions must be divisible by the VGGT patch size")

        dino = self.vggt.aggregator.patch_embed
        first_block = _first_transformer_block(dino)
        captured: list[Tensor] = []

        def capture_input(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            if not captured:
                captured.append(inputs[0].detach())

        hook = first_block.register_forward_pre_hook(capture_input)
        try:
            tokens, patch_start = self.vggt.aggregator(images)
        finally:
            hook.remove()
        if not captured:
            raise RuntimeError("DINOv2 shallow tokens were not captured")

        world_points, confidence = self.vggt.point_head(tokens, images=images, patch_start_idx=patch_start)
        shallow = self._attention_from_tokens(first_block, captured[0], dino.num_register_tokens)
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        shallow = shallow.view(batch, frames, shallow.shape[1], patch_h * patch_w)
        shallow = shallow.permute(0, 2, 1, 3).flatten(2)

        y = torch.arange(patch_h, device=images.device) * self.patch_size + self.patch_size // 2
        x = torch.arange(patch_w, device=images.device) * self.patch_size + self.patch_size // 2
        point_grid = world_points[:, :, y[:, None], x[None, :]]
        color_grid = images.permute(0, 1, 3, 4, 2)[:, :, y[:, None], x[None, :]]
        confidence_grid = confidence[:, :, y[:, None], x[None, :]]

        features = []
        for layer in self.feature_layers:
            if layer >= len(tokens) or tokens[layer] is None:
                raise RuntimeError(f"VGGT layer {layer} was not cached")
            feature = tokens[layer][:, :, patch_start:].flatten(1, 2).detach()
            features.append(torch.nan_to_num(feature))

        points = point_grid.flatten(1, 3).detach()
        colors = color_grid.flatten(1, 3).detach()
        point_confidence = confidence_grid.flatten(1, 3).detach()
        if frame_valid is None:
            point_valid = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
        else:
            if frame_valid.shape != (batch, frames):
                raise ValueError("frame_valid must have shape [B, S]")
            point_valid = frame_valid[:, :, None].expand(batch, frames, patch_h * patch_w).flatten(1)
        point_valid &= torch.isfinite(points).all(dim=-1)
        points = torch.nan_to_num(points)
        colors = torch.nan_to_num(colors)
        point_confidence = torch.nan_to_num(point_confidence)
        return EncoderOutput(
            points=points,
            colors=colors,
            confidence=point_confidence,
            features=features,
            shallow_attention=torch.nan_to_num(shallow.detach()),
            point_valid=point_valid,
        )
