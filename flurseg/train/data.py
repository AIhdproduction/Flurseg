"""Dataset loading, repeat-factor sampling, and paired augmentation."""

from __future__ import annotations

import json
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from flurseg.train.config import DataConfig, ModelConfig, SamplingConfig

IGNORE_INDEX = 255


def mask_to_targets(
    mask: np.ndarray, ignore_index: int = IGNORE_INDEX
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a semantic mask into one binary EoMT target per present class."""
    valid = mask != ignore_index
    classes = [int(value) for value in np.unique(mask[valid])]
    masks = (
        np.stack([mask == class_id for class_id in classes], axis=0)
        if classes
        else np.zeros((0, *mask.shape), dtype=bool)
    )
    return (
        torch.from_numpy(masks.astype(np.float32)),
        torch.tensor(classes, dtype=torch.long),
        torch.from_numpy(valid),
    )


def _crop_or_pad(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: int,
    rng: np.random.Generator,
    rare_class_ids: Collection[int] = (),
    rare_crop_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a paired sample, optionally keeping a rare pixel in the crop."""
    height, width = image.shape[:2]
    pad_height, pad_width = max(0, crop_size - height), max(0, crop_size - width)
    if pad_height or pad_width:
        image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
    height, width = image.shape[:2]
    top, left = int(rng.integers(0, height - crop_size + 1)), int(rng.integers(0, width - crop_size + 1))
    if rare_class_ids and rng.random() < rare_crop_prob:
        rare_pixels = np.argwhere(np.isin(mask, tuple(rare_class_ids)))
        if len(rare_pixels):
            row, column = rare_pixels[int(rng.integers(len(rare_pixels)))]
            top_low, top_high = max(0, int(row) - crop_size + 1), min(int(row), height - crop_size)
            left_low, left_high = max(0, int(column) - crop_size + 1), min(int(column), width - crop_size)
            top, left = int(rng.integers(top_low, top_high + 1)), int(rng.integers(left_low, left_high + 1))
    return image[top : top + crop_size, left : left + crop_size], mask[
        top : top + crop_size, left : left + crop_size
    ]


def _apply_photometric_augmentation(
    image: np.ndarray, config: DataConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply configured photometric changes to a BGR image only."""
    values = image.astype(np.float32) / 255.0
    brightness, saturation = (
        config.brightness or config.color_jitter,
        config.saturation or config.color_jitter,
    )
    if brightness:
        values += rng.uniform(-brightness, brightness)
    if config.contrast:
        values = (values - 0.5) * rng.uniform(1 - config.contrast, 1 + config.contrast) + 0.5
    if saturation or config.hue:
        hsv = cv2.cvtColor(np.clip(values, 0, 1), cv2.COLOR_BGR2HSV)
        if saturation:
            hsv[:, :, 1] *= rng.uniform(1 - saturation, 1 + saturation)
        if config.hue:
            hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-config.hue, config.hue) * 360) % 360
        values = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    values = np.power(np.clip(values, 0, 1), rng.uniform(config.gamma_min, config.gamma_max))
    if config.gaussian_blur_prob and rng.random() < config.gaussian_blur_prob:
        sigma = rng.uniform(config.gaussian_blur_sigma_min, config.gaussian_blur_sigma_max)
        values = cv2.GaussianBlur(values, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if config.gaussian_noise_prob and rng.random() < config.gaussian_noise_prob:
        values += rng.normal(0, config.gaussian_noise_std, values.shape).astype(np.float32)
    return (np.clip(values, 0, 1) * 255).astype(np.uint8)


def _rotate_phone_pair(
    image: np.ndarray, mask: np.ndarray, degrees: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a small upright-preserving phone rotation with ignored mask borders."""
    if not degrees:
        return image, mask
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), float(rng.uniform(-degrees, degrees)), 1.0)
    return (
        cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)),
        cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=IGNORE_INDEX),
    )


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    source: str,
    config: DataConfig,
    rng: np.random.Generator,
    rare_class_ids: Collection[int] = (),
    rare_crop_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a paired BGR image and mask while respecting source orientation."""
    scale = float(rng.uniform(config.scale_min, config.scale_max))
    height, width = image.shape[:2]
    resized_size = max(1, round(width * scale)), max(1, round(height * scale))
    image, mask = (
        cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR),
        cv2.resize(mask, resized_size, interpolation=cv2.INTER_NEAREST),
    )
    image, mask = _crop_or_pad(image, mask, config.crop_size, rng, rare_class_ids, rare_crop_prob)
    if source in {"drone", "ortho"}:
        rotations = int(rng.integers(0, 4))
        image, mask = (
            np.ascontiguousarray(np.rot90(image, rotations)),
            np.ascontiguousarray(np.rot90(mask, rotations)),
        )
        if rng.random() < 0.5:
            image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
    else:
        if rng.random() < 0.5:
            image, mask = np.ascontiguousarray(image[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
        image, mask = _rotate_phone_pair(image, mask, config.phone_rotation_degrees, rng)
    return _apply_photometric_augmentation(image, config, rng), mask


class SegmentationDataset(Dataset[dict[str, Any]]):
    """Load dataset tiles with deterministic repeat-factor training epochs."""

    def __init__(
        self,
        root: Path | str,
        split: str,
        data_config: DataConfig,
        model_config: ModelConfig,
        training: bool,
        tile_ids: Collection[str] | None = None,
        sampling_config: SamplingConfig | None = None,
        sampling_seed: int = 0,
    ) -> None:
        self.root, self.split = Path(root), split
        self.data_config, self.model_config, self.training = data_config, model_config, training
        self.sampling_config, self.sampling_seed = sampling_config or SamplingConfig(), sampling_seed
        with (self.root / "dataset.json").open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        self.ignore_index = int(self.manifest.get("ignore_index", IGNORE_INDEX))
        self.metadata = self.manifest.get("tile_metadata", {})
        self.images = sorted((self.root / split / "images").glob("*"))
        if tile_ids is not None:
            self.images = [path for path in self.images if path.stem in set(tile_ids)]
        self.class_names = {
            int(item["id"]): str(item["name"])
            for item in self.manifest.get("classes", [])
            if int(item["id"]) != self.ignore_index
        }
        self.epoch = 0
        self.tile_class_counts: np.ndarray | None = None
        self.tile_repeats = np.ones(len(self.images), dtype=np.float64)
        self.class_tile_frequency: dict[int, float] = {}
        self.class_repeat_factors: dict[int, float] = {}
        self.rare_class_ids: set[int] = set()
        self._epoch_image_indices = list(range(len(self.images)))
        if training:
            self._compute_sampling_statistics()
            self.set_epoch(0)

    def _compute_sampling_statistics(self) -> None:
        """Read every train mask once and cache per-tile class pixel counts."""
        class_ids = sorted(self.class_names)
        counts = np.zeros((len(self.images), len(class_ids)), dtype=np.int64)
        for row, image_path in enumerate(self.images):
            mask_path = self.root / self.split / "masks" / f"{image_path.stem}.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Unreadable training mask: {mask_path}")
            histogram = np.bincount(mask[mask != self.ignore_index].ravel(), minlength=256)
            for column, class_id in enumerate(class_ids):
                counts[row, column] = int(histogram[class_id])
        self.tile_class_counts = counts
        eligible = counts >= self.sampling_config.min_class_pixels
        frequencies = eligible.mean(axis=0) if len(self.images) else np.zeros(len(class_ids))
        boost_names = set(self.sampling_config.boost_classes)
        repeats = np.ones(len(class_ids), dtype=np.float64)
        for column, class_id in enumerate(class_ids):
            frequency = float(frequencies[column])
            self.class_tile_frequency[class_id] = frequency
            if frequency:
                repeat = max(1.0, math.sqrt(self.sampling_config.threshold / frequency))
                if self.class_names[class_id] in boost_names:
                    repeat *= self.sampling_config.boost_factor
                repeats[column] = repeat
            self.class_repeat_factors[class_id] = float(repeats[column])
        if len(self.images):
            self.tile_repeats = np.maximum(1.0, np.max(np.where(eligible, repeats, 1.0), axis=1))
            self.tile_repeats = np.minimum(self.tile_repeats, self.sampling_config.max_repeat)
        self.rare_class_ids = {
            class_id
            for class_id in class_ids
            if self.class_names[class_id] in boost_names
            or self.class_tile_frequency[class_id] < self.sampling_config.threshold
        }

    @property
    def expected_epoch_length(self) -> float:
        """Return the expected number of samples in one resampled epoch."""
        return float(self.tile_repeats.sum())

    def class_statistics(self) -> list[dict[str, float | int | str]]:
        """Return cached per-class coverage and expected resampling statistics."""
        if self.tile_class_counts is None:
            return []
        valid_pixels = int(self.tile_class_counts.sum())
        rows: list[dict[str, float | int | str]] = []
        for column, class_id in enumerate(sorted(self.class_names)):
            present = self.tile_class_counts[:, column] > 0
            rows.append(
                {
                    "class": self.class_names[class_id],
                    "tiles": int(present.sum()),
                    "pixel_share": float(self.tile_class_counts[:, column].sum() / max(valid_pixels, 1)),
                    "samples_before": int(present.sum()),
                    "samples_after": float(self.tile_repeats[present].sum()),
                }
            )
        return rows

    def sampling_summary_lines(self) -> list[str]:
        """Format cached training coverage and repeat-factor information for logging."""
        lines = ["class | tiles | pixel share | samples before | expected samples after"]
        for row in self.class_statistics():
            lines.append(
                f"{row['class']} | {row['tiles']} | {float(row['pixel_share']):.2%} | {row['samples_before']} | {float(row['samples_after']):.2f}"
            )
        return lines

    def set_epoch(self, epoch: int) -> None:
        """Select this epoch's deterministically stochastic-rounded tile sequence."""
        self.epoch = epoch
        if not self.training:
            return
        rng = np.random.default_rng(self.sampling_seed + epoch * 1_000_003)
        indices: list[int] = []
        for index, repeat in enumerate(self.tile_repeats):
            copies = math.floor(repeat)
            if rng.random() < repeat - copies:
                copies += 1
            indices.extend([index] * copies)
        rng.shuffle(indices)
        self._epoch_image_indices = indices

    def __len__(self) -> int:
        """Return the resampled training epoch length or validation tile count."""
        return len(self._epoch_image_indices) if self.training else len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one normalized tile and its ragged EoMT targets."""
        image_index = self._epoch_image_indices[index] if self.training else index
        image_path = self.images[image_index]
        mask_path = self.root / self.split / "masks" / f"{image_path.stem}.png"
        image, mask = (
            cv2.imread(str(image_path), cv2.IMREAD_COLOR),
            cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE),
        )
        if image is None or mask is None:
            raise RuntimeError(f"Unreadable training tile: {image_path}")
        source = str(self.metadata.get(image_path.stem, {}).get("source", ""))
        rng = np.random.default_rng(self.sampling_seed + self.epoch * 1_000_003 + index)
        if self.training:
            image, mask = augment_pair(
                image,
                mask,
                source,
                self.data_config,
                rng,
                self.rare_class_ids,
                self.sampling_config.rare_crop_prob,
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        values = (rgb - np.asarray(self.model_config.image_mean)) / np.asarray(self.model_config.image_std)
        masks, labels, valid = mask_to_targets(mask, self.ignore_index)
        return {
            "pixel_values": torch.from_numpy(values.transpose(2, 0, 1)).float(),
            "mask_labels": masks,
            "class_labels": labels,
            "valid_mask": valid,
            "path": image_path,
        }


def collate_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate equal-size crops while retaining ragged EoMT labels."""
    return {
        "pixel_values": torch.stack([sample["pixel_values"] for sample in samples]),
        "mask_labels": [sample["mask_labels"] for sample in samples],
        "class_labels": [sample["class_labels"] for sample in samples],
        "valid_masks": [sample["valid_mask"] for sample in samples],
        "paths": [sample["path"] for sample in samples],
    }
