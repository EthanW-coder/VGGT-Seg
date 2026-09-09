from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from vggt_seg.config import ModelConfig

from .backbone import EncoderOutput, FrozenVGGTEncoder
from .decoder import TopologyRectifiedInterleavedDecoder, VanillaDecoder
from .position import PositionEmbedding3D
from .sampling import attention_guided_fps, farthest_point_sampling, gather_points


class MultiLevelFeatureProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, levels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim)) for _ in range(levels)]
        )
        self.semantic_fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim * levels),
            nn.Linear(hidden_dim * levels, hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: list[Tensor]) -> tuple[Tensor, Tensor]:
        if len(features) != len(self.projections):
            raise ValueError("the number of encoder feature levels does not match the model")
        projected = [layer(value) for layer, value in zip(self.projections, features)]
        return projected[0], self.semantic_fusion(torch.cat(projected, dim=-1))


class VGGTSeg(nn.Module):
    def __init__(self, config: ModelConfig, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.config = config
        self.encoder = (
            encoder
            if encoder is not None
            else FrozenVGGTEncoder(
                repository=config.backbone_repo,
                checkpoint=config.backbone_checkpoint,
                feature_layers=config.feature_layers,
                patch_size=config.patch_size,
            )
        )
        self.feature_projection = MultiLevelFeatureProjection(2048, config.hidden_dim, len(config.feature_layers))
        self.position = PositionEmbedding3D(config.hidden_dim)
        self.geometry_position = nn.Sequential(
            PositionEmbedding3D(config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

        if config.variant == "full":
            self.learnable_queries = nn.Parameter(torch.empty(config.learnable_queries, config.hidden_dim))
            nn.init.normal_(self.learnable_queries, std=0.02)
            self.decoder = TopologyRectifiedInterleavedDecoder(
                config.hidden_dim,
                config.num_heads,
                config.decoder_layers,
                config.num_classes,
                config.mask_threshold,
                config.topology_gamma,
            )
        else:
            self.register_parameter("learnable_queries", None)
            self.decoder = VanillaDecoder(
                config.hidden_dim, config.num_heads, config.decoder_layers, config.num_classes
            )

    def train(self, mode: bool = True) -> "VGGTSeg":
        super().train(mode)
        self.encoder.eval()
        return self

    def _initialize_queries(self, encoded: EncoderOutput) -> tuple[Tensor, Tensor]:
        if self.config.variant == "full":
            indices = attention_guided_fps(
                encoded.points,
                encoded.shallow_attention,
                self.config.semantic_queries,
                self.config.afps_alpha,
                encoded.point_valid,
            )
            anchors = gather_points(encoded.points, indices)
            semantic = self.position(anchors)
            learned = self.learnable_queries.unsqueeze(0).expand(encoded.points.shape[0], -1, -1)
            queries = torch.cat((semantic, learned), dim=1)
            zero_anchors = torch.zeros(
                encoded.points.shape[0], self.config.learnable_queries, 3,
                dtype=encoded.points.dtype, device=encoded.points.device,
            )
            return queries, torch.cat((anchors, zero_anchors), dim=1)

        count = self.config.semantic_queries + self.config.learnable_queries
        indices = farthest_point_sampling(encoded.points, count, encoded.point_valid)
        anchors = gather_points(encoded.points, indices)
        return self.position(anchors), anchors

    def forward(
        self,
        images: Tensor,
        frame_valid: Tensor | None = None,
        topology_targets: dict[str, Tensor] | None = None,
    ) -> dict[str, Any]:
        encoded = self.encoder(images, frame_valid)
        point_valid = encoded.point_valid.to(dtype=torch.bool) & torch.isfinite(encoded.points).all(dim=-1)
        encoded = EncoderOutput(
            points=torch.nan_to_num(encoded.points),
            colors=torch.nan_to_num(encoded.colors),
            confidence=torch.nan_to_num(encoded.confidence),
            features=[torch.nan_to_num(feature) for feature in encoded.features],
            shallow_attention=torch.nan_to_num(encoded.shallow_attention),
            point_valid=point_valid,
        )
        geometry, semantics = self.feature_projection(encoded.features)
        geometry = geometry + self.geometry_position(encoded.points)
        queries, query_anchors = self._initialize_queries(encoded)

        if self.config.variant == "full":
            topology_targets = topology_targets or {}
            decoded = self.decoder(
                queries,
                geometry,
                semantics,
                encoded.points,
                encoded.point_valid,
                topology_targets.get("centers"),
                topology_targets.get("covariance"),
                topology_targets.get("valid"),
            )
        else:
            decoded = self.decoder(queries, semantics, encoded.points, encoded.point_valid)

        return {
            "pred_masks": decoded.mask_logits,
            "pred_logits": decoded.class_logits,
            "query_features": decoded.query_features,
            "query_centers": decoded.query_centers,
            "query_anchors": query_anchors,
            "topology_features": decoded.topology_features,
            "aux_outputs": decoded.auxiliary,
            "points": encoded.points,
            "colors": encoded.colors,
            "point_confidence": encoded.confidence,
            "point_valid": encoded.point_valid,
        }
