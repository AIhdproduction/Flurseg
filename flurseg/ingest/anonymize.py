"""Prompt-based selective blur for privacy-sensitive phone imagery."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from flurseg.segment.backend import SegmentationBackend
from flurseg.segment.tiling import TileAccumulator, iter_tiles, pad_to_tile


def anonymize(
    image: np.ndarray, backend: SegmentationBackend, prompts: Sequence[str], tiling: object, blur: object
) -> tuple[np.ndarray, float]:
    """Blur only pixels backed by privacy-prompt scores and return their fraction."""
    tile_size = int(tiling.tile_size)
    overlap = tiling.min_overlap
    minimum = float(blur.min_score)
    kernel = int(blur.blur_kernel)
    sigma = float(blur.blur_sigma)
    height, width = image.shape[:2]
    accumulator = TileAccumulator(len(prompts), width, height, reduction="max")
    for tile in iter_tiles(width, height, tile_size, overlap):
        crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        if crop.shape[:2] != (tile_size, tile_size):
            padded, slices = pad_to_tile(crop, tile_size)
            scores = backend.score_tile(padded, prompts)[:, slices[0], slices[1]]
        else:
            scores = backend.score_tile(crop, prompts)
        accumulator.add(tile, scores)
    selected = accumulator.scores().max(axis=0) >= minimum
    selected = cv2.dilate(selected.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    blurred = cv2.GaussianBlur(image, (kernel, kernel), sigma)
    result = image.copy()
    result[selected] = blurred[selected]
    return result, float(selected.mean())
