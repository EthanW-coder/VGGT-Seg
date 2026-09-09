from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    variant: str = "full"
    num_classes: int = 18
    image_size: int = 518
    patch_size: int = 14
    hidden_dim: int = 256
    num_heads: int = 8
    decoder_layers: int = 6
    semantic_queries: int = 200
    learnable_queries: int = 100
    afps_alpha: float = 1.0
    mask_threshold: float = 0.5
    topology_gamma: float = 0.15
    icr_temperature: float = 0.1
    icr_radius: float = 0.2
    backbone_repo: str = "facebook/VGGT-1B"
    backbone_checkpoint: str | None = None
    feature_layers: list[int] = field(default_factory=lambda: [4, 11, 17, 23])

    def validate(self) -> None:
        if self.variant not in {"vanilla", "full"}:
            raise ValueError("model.variant must be 'vanilla' or 'full'")
        if self.image_size % self.patch_size:
            raise ValueError("model.image_size must be divisible by model.patch_size")
        if self.hidden_dim % self.num_heads:
            raise ValueError("model.hidden_dim must be divisible by model.num_heads")
        if self.semantic_queries < 1 or self.learnable_queries < 0:
            raise ValueError("query counts must be non-negative and include a sampled query")
        if self.decoder_layers < 1:
            raise ValueError("model.decoder_layers must be positive")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("model.mask_threshold must be in [0, 1]")
        if self.topology_gamma < 0.0:
            raise ValueError("model.topology_gamma must be non-negative")
        if self.afps_alpha < 0.0:
            raise ValueError("model.afps_alpha must be non-negative")
        if self.icr_temperature <= 0.0 or self.icr_radius <= 0.0:
            raise ValueError("contrastive temperature and radius must be positive")


@dataclass
class DataConfig:
    root: str = "data/scannet"
    train_manifest: str = "train.json"
    val_manifest: str = "val.json"
    train_min_frames: int = 12
    train_max_frames: int = 48
    inference_max_frames: int = 80
    num_workers: int = 4


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size_per_gpu: int = 2
    learning_rate: float = 3.0e-4
    min_learning_rate: float = 1.0e-4
    weight_decay: float = 0.05
    gradient_clip_norm: float = 0.1
    seed: int = 42
    amp: bool = True
    output_dir: str = "outputs"
    resume: str | None = None
    lambda_cls: float = 0.5
    lambda_mask: float = 1.0
    lambda_icr: float = 0.5
    lambda_bce: float = 1.0
    lambda_dice: float = 1.0
    no_object_weight: float = 0.1


@dataclass
class ExperimentConfig:
    dataset: str = "scannet"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        if not 1 <= self.data.train_min_frames <= self.data.train_max_frames:
            raise ValueError("invalid training frame range")
        if self.data.inference_max_frames < 1:
            raise ValueError("inference_max_frames must be positive")
        if self.train.epochs < 1 or self.train.batch_size_per_gpu < 1:
            raise ValueError("training epochs and batch size must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    known = set(instance.__dataclass_fields__)
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    for key, value in values.items():
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path, variant: str | None = None) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    known = {"dataset", "model", "data", "train"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {sorted(unknown)}")

    config = ExperimentConfig(dataset=raw.get("dataset", "scannet"))
    _merge_dataclass(config.model, raw.get("model", {}))
    _merge_dataclass(config.data, raw.get("data", {}))
    _merge_dataclass(config.train, raw.get("train", {}))
    if variant is not None:
        config.model.variant = variant
    config.validate()
    return config
