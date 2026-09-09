from __future__ import annotations

import torch
from torch import Tensor


def _validate_inputs(points: Tensor, count: int, valid: Tensor | None) -> Tensor:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, N, 3]")
    if count < 1:
        raise ValueError("count must be positive")
    if valid is None:
        valid = torch.ones(points.shape[:2], dtype=torch.bool, device=points.device)
    if valid.shape != points.shape[:2]:
        raise ValueError("valid must have shape [B, N]")
    # Never sample non-finite coordinates, even if validity came from an
    # external annotation source.
    valid = valid.to(dtype=torch.bool) & torch.isfinite(points).all(dim=-1)
    if (valid.sum(dim=1) == 0).any():
        raise ValueError("every batch item must contain at least one valid point")
    return valid


@torch.no_grad()
def farthest_point_sampling(points: Tensor, count: int, valid: Tensor | None = None) -> Tensor:
    valid = _validate_inputs(points, count, valid)
    batch, total, _ = points.shape
    count = min(count, total)
    selected = torch.empty((batch, count), dtype=torch.long, device=points.device)
    safe_points = points.masked_fill(~valid.unsqueeze(-1), 0.0)
    centroid = safe_points.sum(dim=1) / valid.sum(dim=1, keepdim=True)
    distance = torch.linalg.vector_norm(safe_points - centroid.unsqueeze(1), dim=-1)
    distance = distance.masked_fill(~valid, -torch.inf)
    current = distance.argmax(dim=1)
    minimum = torch.full((batch, total), torch.inf, dtype=points.dtype, device=points.device)
    rows = torch.arange(batch, device=points.device)
    selected_mask = torch.zeros((batch, total), dtype=torch.bool, device=points.device)
    first_valid = valid.to(torch.int64).argmax(dim=1)

    for step in range(count):
        selected[:, step] = current
        selected_mask[rows, current] = True
        delta = torch.linalg.vector_norm(safe_points - safe_points[rows, current].unsqueeze(1), dim=-1)
        minimum = torch.minimum(minimum, delta)
        available = valid & ~selected_mask
        score = minimum.masked_fill(~available, -torch.inf)
        if step + 1 < count:
            current = torch.where(available.any(dim=1), score.argmax(dim=1), first_valid)
    return selected


@torch.no_grad()
def attention_guided_fps(
    points: Tensor,
    attention: Tensor,
    count: int,
    alpha: float = 1.0,
    valid: Tensor | None = None,
) -> Tensor:
    valid = _validate_inputs(points, count, valid)
    if attention.ndim == 3:
        attention = attention.mean(dim=1)
    if attention.shape != points.shape[:2]:
        raise ValueError("attention must have shape [B, N] or [B, H, N]")

    batch, total, _ = points.shape
    count = min(count, total)
    logits = torch.nan_to_num(attention.float(), nan=0.0, posinf=0.0, neginf=0.0)
    logits = logits.masked_fill(~valid, -torch.inf)
    weights = logits.softmax(dim=-1).clamp_min(1e-12)
    selected = torch.empty((batch, count), dtype=torch.long, device=points.device)
    rows = torch.arange(batch, device=points.device)
    current = weights.argmax(dim=1)
    minimum = torch.full((batch, total), torch.inf, dtype=points.dtype, device=points.device)
    selected_mask = torch.zeros((batch, total), dtype=torch.bool, device=points.device)
    first_valid = valid.to(torch.int64).argmax(dim=1)
    safe_points = points.masked_fill(~valid.unsqueeze(-1), 0.0)

    for step in range(count):
        selected[:, step] = current
        selected_mask[rows, current] = True
        delta = torch.linalg.vector_norm(safe_points - safe_points[rows, current].unsqueeze(1), dim=-1)
        minimum = torch.minimum(minimum, delta)
        available = valid & ~selected_mask
        score = alpha * weights.log() + minimum
        score = score.masked_fill(~available, -torch.inf)
        if step + 1 < count:
            current = torch.where(available.any(dim=1), score.argmax(dim=1), first_valid)
    return selected


def gather_points(values: Tensor, indices: Tensor) -> Tensor:
    if values.shape[0] != indices.shape[0]:
        raise ValueError("batch dimensions do not match")
    expand = indices.view(indices.shape + (1,) * (values.ndim - 2)).expand(
        indices.shape + values.shape[2:]
    )
    return values.gather(1, expand)
