"""Dataset loading, paired augmentation, and EoMT target conversion."""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from flurseg.train.config import DataConfig, ModelConfig

IGNORE_INDEX = 255


def mask_to_targets(
    mask: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a semantic mask into one binary EoMT target per present class."""
    valid = mask != ignore_index
    classes = [int(value) for value in np.unique(mask[valid])]
    if classes:
        masks = np.stack([mask == class_id for class_id in classes], axis=0)
    else:
        masks = np.zeros((0, *mask.shape), dtype=bool)
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
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a paired sample, padding image and ignored mask pixels when needed."""
    height, width = image.shape[:2]
    pad_height = max(0, crop_size - height)
    pad_width = max(0, crop_size - width)
    if pad_height or pad_width:
        image = cv2.copyMakeBorder(
            image,
            0,
            pad_height,
            0,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        mask = cv2.copyMakeBorder(
            mask,
            0,
            pad_height,
            0,
            pad_width,
            cv2.BORDER_CONSTANT,
            value=IGNORE_INDEX,
        )
    height, width = image.shape[:2]
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))
    return (
        image[top : top + crop_size, left : left + crop_size],
        mask[top : top + crop_size, left : left + crop_size],
    )


def _apply_colour_jitter(
    image: np.ndarray,
    jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply lightweight brightness and saturation jitter to a BGR image."""
    if not jitter:
        return image
    floating = image.astype(np.float32) / 255.0
    floating = np.clip(floating * rng.uniform(1 - jitter, 1 + jitter), 0, 1)
    mean = floating.mean(axis=2, keepdims=True)
    floating = np.clip(
        mean + (floating - mean) * rng.uniform(1 - jitter, 1 + jitter),
        0,
        1,
    )
    return (floating * 255).astype(np.uint8)


def augment_pair(
    image: np.ndarray,
    mask: np.ndarray,
    source: str,
    config: DataConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical geometry to a BGR image and label mask.

    Phone tiles retain their upright orientation. Nadir sources (``drone`` and
    ``ortho``) can additionally receive quarter rotations and vertical flips.
    """
    scale = float(rng.uniform(config.scale_min, config.scale_max))
    height, width = image.shape[:2]
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    image = cv2.resize(image, resized_size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, resized_size, interpolation=cv2.INTER_NEAREST)
    image, mask = _crop_or_pad(image, mask, config.crop_size, rng)
    if rng.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if source in {"drone", "ortho"}:
        rotations = int(rng.integers(0, 4))
        if rotations:
            image = np.ascontiguousarray(np.rot90(image, rotations))
            mask = np.ascontiguousarray(np.rot90(mask, rotations))
        if rng.random() < 0.5:
            image = np.ascontiguousarray(image[::-1])
            mask = np.ascontiguousarray(mask[::-1])
    return _apply_colour_jitter(image, config.color_jitter, rng), mask


class SegmentationDataset(Dataset[dict[str, Any]]):
    """Load ``train/dataset/<split>`` tiles with deterministic epochs."""

    def __init__(
        self,
        root: Path | str,
        split: str,
        data_config: DataConfig,
        model_config: ModelConfig,
        training: bool,
        tile_ids: Collection[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.data_config = data_config
        self.model_config = model_config
        self.training = training
        with (self.root / "dataset.json").open("r", encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        self.ignore_index = int(self.manifest.get("ignore_index", IGNORE_INDEX))
        self.metadata = self.manifest.get("tile_metadata", {})
        self.images = sorted((self.root / split / "images").glob("*"))
        if tile_ids is not None:
            allowed = set(tile_ids)
            self.images = [path for path in self.images if path.stem in allowed]
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic augmentation epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        """Return the number of tiles in this split."""
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load one normalized tile and its ragged EoMT targets."""
        image_path = self.images[index]
        mask_path = self.root / self.split / "masks" / f"{image_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Unreadable training tile: {image_path}")
        source = str(self.metadata.get(image_path.stem, {}).get("source", ""))
        rng = np.random.default_rng(self.epoch * 1_000_003 + index)
        if self.training:
            image, mask = augment_pair(image, mask, source, self.data_config, rng)
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
