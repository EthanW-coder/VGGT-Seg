from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .position import PositionEmbedding3D


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, value: Tensor) -> Tensor:
        return value + self.layers(self.norm(value))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.ffn = FeedForward(hidden_dim)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_valid: Tensor,
        spatial_mask: Tensor | None = None,
        query_valid: Tensor | None = None,
    ) -> Tensor:
        attention_mask = None
        if spatial_mask is not None:
            attention_mask = spatial_mask.repeat_interleave(self.num_heads, dim=0)
        update = self.cross_attention(
            self.cross_norm(queries),
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_valid,
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        queries = queries + update
        update = self.self_attention(
            self.self_norm(queries),
            self.self_norm(queries),
            self.self_norm(queries),
            key_padding_mask=None if query_valid is None else ~query_valid,
            need_weights=False,
        )[0]
        queries = queries + update
        queries = self.ffn(queries)
        if query_valid is not None:
            queries = queries * query_valid.unsqueeze(-1)
        return queries


class PredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.scene_projection = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim))
        self.mask_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes + 1),
        )

    def forward(self, queries: Tensor, scene_features: Tensor) -> tuple[Tensor, Tensor]:
        scene = self.scene_projection(scene_features).sigmoid()
        masks = torch.einsum("bqd,bnd->bqn", self.mask_projection(queries), scene)
        return masks / math.sqrt(queries.shape[-1]), self.classifier(queries)


class TopologyQueryGenerator(nn.Module):
    def __init__(self, hidden_dim: int, gamma: float) -> None:
        super().__init__()
        self.gamma = gamma
        self.position = PositionEmbedding3D(hidden_dim)

    def forward(self, centers: Tensor, covariance: Tensor, valid: Tensor) -> Tensor:
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
        scale = eigenvalues.clamp_min(1e-8).sqrt().to(dtype=centers.dtype)
        transform = eigenvectors.to(dtype=centers.dtype) @ torch.diag_embed(scale)
        noise = torch.randn_like(centers).unsqueeze(-2) @ transform.transpose(-1, -2)
        anchors = centers + math.sqrt(self.gamma) * noise.squeeze(-2)
        return self.position(anchors) * valid.unsqueeze(-1)


def _spatial_attention_mask(mask_logits: Tensor, valid: Tensor, threshold: float) -> Tensor:
    blocked = mask_logits.sigmoid() <= threshold
    blocked = blocked | ~valid.unsqueeze(1)
    all_blocked = blocked.all(dim=-1)
    if all_blocked.any():
        best = mask_logits.masked_fill(~valid.unsqueeze(1), -torch.inf).argmax(dim=-1)
        batch, query = all_blocked.nonzero(as_tuple=True)
        blocked[batch, query, best[batch, query]] = False
    return blocked


def _query_centers(mask_logits: Tensor, points: Tensor, valid: Tensor) -> Tensor:
    weights = mask_logits.sigmoid() * valid.unsqueeze(1)
    safe_points = points.masked_fill(~valid.unsqueeze(-1), 0.0)
    return torch.einsum("bqn,bnd->bqd", weights, safe_points) / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)


@dataclass
class DecoderOutput:
    mask_logits: Tensor
    class_logits: Tensor
    query_features: Tensor
    query_centers: Tensor
    topology_features: Tensor | None
    auxiliary: list[dict[str, Tensor]]


class VanillaDecoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, layers: int, num_classes: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(hidden_dim, num_heads) for _ in range(layers)])
        self.prediction = PredictionHead(hidden_dim, num_classes)

    def forward(
        self,
        queries: Tensor,
        scene_features: Tensor,
        points: Tensor,
        point_valid: Tensor,
    ) -> DecoderOutput:
        auxiliary = []
        for layer in self.layers:
            queries = layer(queries, scene_features, point_valid)
            masks, classes = self.prediction(queries, scene_features)
            auxiliary.append({"mask_logits": masks, "class_logits": classes})
        return DecoderOutput(
            mask_logits=masks,
            class_logits=classes,
            query_features=queries,
            query_centers=_query_centers(masks, points, point_valid),
            topology_features=None,
            auxiliary=auxiliary[:-1],
        )


class TIDLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.geometry_attention = AttentionBlock(hidden_dim, num_heads)
        self.topology_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.semantic_attention = AttentionBlock(hidden_dim, num_heads)

    def forward(
        self,
        queries: Tensor,
        geometry: Tensor,
        semantics: Tensor,
        point_valid: Tensor,
        spatial_mask: Tensor,
        topology: Tensor | None,
        topology_valid: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        queries = self.geometry_attention(queries, geometry, point_valid, spatial_mask)
        query_count = queries.shape[1]
        if topology is not None:
            modulated = torch.cat((queries, topology), dim=1)
            query_valid = torch.cat(
                (
                    torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device),
                    topology_valid,
                ),
                dim=1,
            )
        else:
            modulated = queries
            query_valid = None
        modulated = modulated + self.topology_projection(modulated)
        modulated = self.semantic_attention(modulated, semantics, point_valid, query_valid=query_valid)
        return modulated[:, :query_count], None if topology is None else modulated[:, query_count:]


class TopologyRectifiedInterleavedDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        layers: int,
        num_classes: int,
        mask_threshold: float,
        topology_gamma: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TIDLayer(hidden_dim, num_heads) for _ in range(layers)])
        self.prediction = PredictionHead(hidden_dim, num_classes)
        self.topology = TopologyQueryGenerator(hidden_dim, topology_gamma)
        self.mask_threshold = mask_threshold

    def forward(
        self,
        queries: Tensor,
        geometry_features: Tensor,
        semantic_features: Tensor,
        points: Tensor,
        point_valid: Tensor,
        topology_centers: Tensor | None = None,
        topology_covariance: Tensor | None = None,
        topology_valid: Tensor | None = None,
    ) -> DecoderOutput:
        if not self.training:
            topology_centers = topology_covariance = topology_valid = None
        topology = None
        if topology_centers is not None:
            if topology_covariance is None or topology_valid is None:
                raise ValueError("topology centers, covariance and validity must be provided together")
            topology = self.topology(topology_centers, topology_covariance, topology_valid)

        auxiliary = []
        masks, classes = self.prediction(queries, semantic_features)
        decoded_topology = None
        for layer in self.layers:
            spatial_mask = _spatial_attention_mask(masks, point_valid, self.mask_threshold)
            queries, decoded_topology = layer(
                queries,
                geometry_features,
                semantic_features,
                point_valid,
                spatial_mask,
                topology,
                topology_valid,
            )
            masks, classes = self.prediction(queries, semantic_features)
            auxiliary.append({"mask_logits": masks, "class_logits": classes})
        return DecoderOutput(
            mask_logits=masks,
            class_logits=classes,
            query_features=queries,
            query_centers=_query_centers(masks, points, point_valid),
            topology_features=decoded_topology,
            auxiliary=auxiliary[:-1],
        )
