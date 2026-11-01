"""Evenly spaced training tile cutting, padding, and ignore-filtering."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from flurseg.segment.tiling import tile_starts


@dataclass(slots=True)
class TrainingTile:
    """One extracted training tile containing an image crop and mask crop."""

    index: int
    x: int
    y: int
    width: int
    height: int
    image: np.ndarray
    mask: np.ndarray
    ignore_fraction: float
    pad: tuple[int, int, int, int]  # (top, bottom, left, right)
    original_size: tuple[int, int]  # (height, width)

    @property
    def pad_top(self) -> int:
        return self.pad[0]

    @property
    def pad_bottom(self) -> int:
        return self.pad[1]

    @property
    def pad_left(self) -> int:
        return self.pad[2]

    @property
    def pad_right(self) -> int:
        return self.pad[3]


def iter_training_tiles(
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int = 1024,
    max_ignore_fraction: float = 0.9,
) -> Iterator[TrainingTile]:
    """Yield valid training tiles in row-major order."""
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(f"image size {image.shape[:2]} and mask size {mask.shape[:2]} must match")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    orig_height, orig_width = image.shape[:2]

    pad_bottom = max(0, tile_size - orig_height)
    pad_right = max(0, tile_size - orig_width)
    pad = (0, pad_bottom, 0, pad_right)

    target_height = orig_height + pad_bottom
    target_width = orig_width + pad_right

    if pad_bottom > 0 or pad_right > 0:
        if image.ndim == 3:
            padded_image = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
            padded_image[:orig_height, :orig_width, :] = image
        else:
            padded_image = np.zeros((target_height, target_width), dtype=image.dtype)
            padded_image[:orig_height, :orig_width] = image

        padded_mask = np.full((target_height, target_width), 255, dtype=np.uint8)
        padded_mask[:orig_height, :orig_width] = mask
    else:
        padded_image = image
        padded_mask = mask.astype(np.uint8, copy=False)

    y_starts = tile_starts(target_height, tile_size, 0.0)
    x_starts = tile_starts(target_width, tile_size, 0.0)

    tile_index = 0
    for y in y_starts:
        for x in x_starts:
            crop_img = padded_image[y : y + tile_size, x : x + tile_size]
            crop_mask = padded_mask[y : y + tile_size, x : x + tile_size]

            ignore_fraction = float(np.count_nonzero(crop_mask == 255) / crop_mask.size)
            if ignore_fraction > max_ignore_fraction:
                continue

            yield TrainingTile(
                index=tile_index,
                x=x,
                y=y,
                width=tile_size,
                height=tile_size,
                image=crop_img,
                mask=crop_mask,
                ignore_fraction=ignore_fraction,
                pad=pad,
                original_size=(orig_height, orig_width),
            )
            tile_index += 1


def extract_tiles(
    image: np.ndarray,
    mask: np.ndarray,
    tile_size: int = 1024,
    max_ignore_fraction: float = 0.9,
) -> list[TrainingTile]:
    """Extract training tiles as a list, padding small images and dropping ignore-heavy tiles."""
    return list(iter_training_tiles(image, mask, tile_size, max_ignore_fraction))
