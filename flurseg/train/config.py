"""Validated configuration for local EoMT-DINOv3 training."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Model):
    """Describe the EoMT-DINOv3 checkpoint and image normalization."""

    pretrained: str = Field(min_length=1)
    tiny_random: bool = False
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


class DatasetSplitConfig(_Model):
    """Hold explicit, reproducible dataset-split overrides."""

    overrides: dict[str, Literal["train", "val", "test"]] = Field(default_factory=dict)


class DataConfig(_Model):
    """Configure paired image and mask augmentation."""

    crop_size: int = Field(default=768, gt=0)
    scale_min: float = Field(default=0.5, gt=0)
    scale_max: float = Field(default=2.0, gt=0)
    color_jitter: float = Field(default=0, ge=0, le=1)
    brightness: float = Field(default=0.3, ge=0, le=1)
    contrast: float = Field(default=0.3, ge=0, le=1)
    saturation: float = Field(default=0.2, ge=0, le=1)
    gamma_min: float = Field(default=0.7, gt=0)
    gamma_max: float = Field(default=1.4, gt=0)
    hue: float = Field(default=0.02, ge=0, le=0.5)
    gaussian_blur_prob: float = Field(default=0.1, ge=0, le=1)
    gaussian_blur_sigma_min: float = Field(default=0.5, ge=0)
    gaussian_blur_sigma_max: float = Field(default=1.2, ge=0)
    gaussian_noise_prob: float = Field(default=0.1, ge=0, le=1)
    gaussian_noise_std: float = Field(default=0.03, ge=0, le=1)
    phone_rotation_degrees: float = Field(default=8.0, ge=0, le=45)

    @model_validator(mode="after")
    def validate_scale_range(self) -> DataConfig:
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must not exceed scale_max")
        if self.gamma_min > self.gamma_max:
            raise ValueError("gamma_min must not exceed gamma_max")
        if self.gaussian_blur_sigma_min > self.gaussian_blur_sigma_max:
            raise ValueError("gaussian_blur_sigma_min must not exceed gaussian_blur_sigma_max")
        return self


class SamplingConfig(_Model):
    """Configure repeat-factor sampling and rare-class crop selection."""

    min_class_pixels: int = Field(default=200, ge=1)
    threshold: float = Field(default=0.3, gt=0, le=1)
    max_repeat: float = Field(default=5.0, ge=1)
    boost_classes: list[str] = Field(default_factory=lambda: ["curb", "road_marking", "utility_cover"])
    boost_factor: float = Field(default=1.5, ge=1)
    rare_crop_prob: float = Field(default=0.5, ge=0, le=1)


class TrainingConfig(_Model):
    """Configure optimization, evaluation, and early stopping."""

    seed: int = 42
    batch_size: int = Field(default=2, gt=0)
    gradient_accumulation: int = Field(default=8, gt=0)
    epochs_target: int = Field(default=60, gt=0)
    total_steps: int | Literal["auto"] = "auto"
    warmup_steps: int | Literal["auto"] = "auto"
    eval_every: int | Literal["auto"] = "auto"
    patience: int = Field(default=8, gt=0)
    head_lr: float = Field(default=1e-4, gt=0)
    backbone_lr: float = Field(default=1e-5, gt=0)
    layer_decay: float = Field(default=0.8, gt=0, le=1)
    weight_decay: float = Field(default=0.05, ge=0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    polynomial_power: float = Field(default=0.9, gt=0)


class LossConfig(_Model):
    """Configure the ignore-aware EoMT loss weights and sampling."""

    mask_weight: float = Field(default=5.0, gt=0)
    dice_weight: float = Field(default=5.0, gt=0)
    class_weight: float = Field(default=2.0, gt=0)
    no_object_weight: float = Field(default=0.1, gt=0)
    num_points: int = Field(default=12544, gt=0)
    oversample_ratio: float = Field(default=3.0, ge=1.0)
    importance_sample_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
    deep_supervision: bool = Field(default=True)


class InferenceConfig(_Model):
    """Configure sliding-window inference."""

    window_size: int = Field(default=768, gt=0)
    stride: int = Field(default=384, gt=0)


class TrainConfig(_Model):
    """Collect all validated training settings."""

    model: ModelConfig
    dataset_split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)

    @model_validator(mode="after")
    def validate_window_matches_crop(self) -> TrainConfig:
        if self.inference.window_size != self.data.crop_size:
            raise ValueError(
                "inference.window_size must equal data.crop_size because the EoMT patch grid is fixed to the training size"
            )
        return self


def load_train_config(path: Path | str) -> TrainConfig:
    """Load a strict training YAML file."""
    with Path(path).open("r", encoding="utf-8") as stream:
        return TrainConfig.model_validate(yaml.safe_load(stream) or {})
