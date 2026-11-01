"""Evenly spaced overlapping tiles and score accumulation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np


def tile_starts(dim: int, tile: int, min_overlap: float) -> list[int]:
    """Return evenly spaced starts that cover a dimension without a remainder."""
    if dim <= 0 or tile <= 0:
        raise ValueError("dim and tile must be positive")
    if dim <= tile:
        return [0]
    overlap = ceil(tile * min_overlap) if isinstance(min_overlap, float) else min_overlap
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    extra = dim - tile
    count = ceil(extra / (tile - overlap)) + 1
    return [round(index * extra / (count - 1)) for index in range(count)]


@dataclass(frozen=True, slots=True)
class Tile:
    """One image tile bounded by pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def window(self) -> tuple[int, int, int, int]:
        """Return ``(x0, y0, x1, y1)`` coordinates."""
        return self.x, self.y, self.x + self.width, self.y + self.height


def iter_tiles(width: int, height: int, tile: int, min_overlap: float) -> Iterator[Tile]:
    """Iterate tiles in row-major order, with smaller edge tiles only for small images."""
    tile_width = min(width, tile)
    tile_height = min(height, tile)
    for y in tile_starts(height, tile, min_overlap):
        for x in tile_starts(width, tile, min_overlap):
            yield Tile(x=x, y=y, width=tile_width, height=tile_height)


def pad_to_tile(image: np.ndarray, tile_size: int) -> tuple[np.ndarray, tuple[slice, slice]]:
    """Pad a small image to a square tile and return slices that crop it back."""
    height, width = image.shape[:2]
    if height > tile_size or width > tile_size:
        raise ValueError("pad_to_tile only accepts images no larger than the tile")
    pad_height = tile_size - height
    pad_width = tile_size - width
    padding = ((0, pad_height), (0, pad_width)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(image, padding, mode="edge"), (slice(0, height), slice(0, width))


class TileAccumulator:
    """Merge tiled dense scores using maximum or feathered weighted mean reduction."""

    def __init__(self, classes: int, width: int, height: int, reduction: Literal["max", "mean"] = "max") -> None:
        if reduction not in {"max", "mean"}:
            raise ValueError("reduction must be 'max' or 'mean'")
        self.classes = classes
        self.width = width
        self.height = height
        self.reduction = reduction
        self._scores = (
            np.full((classes, height, width), -np.inf, dtype=np.float32)
            if reduction == "max"
            else np.zeros((classes, height, width), dtype=np.float32)
        )
        self._weights = np.zeros((height, width), dtype=np.float32)
        self._count = np.zeros((height, width), dtype=np.float32)
        self._moments = np.zeros((height, width), dtype=np.float32)
        self._squares = np.zeros((height, width), dtype=np.float32)

    def _feather_weights(self, tile: Tile) -> np.ndarray:
        y = np.ones(tile.height, dtype=np.float32)
        x = np.ones(tile.width, dtype=np.float32)
        if tile.x > 0:
            x = np.minimum(x, (np.arange(tile.width, dtype=np.float32) + 1) / max(1, tile.width // 4))
        if tile.x + tile.width < self.width:
            x = np.minimum(x, (np.arange(tile.width, 0, -1, dtype=np.float32)) / max(1, tile.width // 4))
        if tile.y > 0:
            y = np.minimum(y, (np.arange(tile.height, dtype=np.float32) + 1) / max(1, tile.height // 4))
        if tile.y + tile.height < self.height:
            y = np.minimum(y, (np.arange(tile.height, 0, -1, dtype=np.float32)) / max(1, tile.height // 4))
        return np.outer(y, x).clip(1e-6, 1.0)

    def add(self, tile: Tile, scores: np.ndarray) -> None:
        """Merge one ``(classes, tile.height, tile.width)`` score stack."""
        expected = (self.classes, tile.height, tile.width)
        if scores.shape != expected:
            raise ValueError(f"expected scores shaped {expected}, got {scores.shape}")
        destination = np.s_[..., tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        weights = self._feather_weights(tile) if self.reduction == "mean" else np.ones((tile.height, tile.width))
        if self.reduction == "max":
            self._scores[destination] = np.maximum(self._scores[destination], scores)
        else:
            self._scores[destination] += scores * weights
            self._weights[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += weights
        scalar = scores.mean(axis=0)
        self._moments[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += scalar
        self._squares[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += scalar * scalar
        self._count[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] += 1

    def scores(self) -> np.ndarray:
        """Return the merged score stack."""
        if self.reduction == "max":
            return np.where(np.isfinite(self._scores), self._scores, 0).astype(np.float32)
        return (self._scores / np.maximum(self._weights, 1e-6)).astype(np.float32)

    def variance(self) -> np.ndarray:
        """Return per-pixel variance across accumulated tile/TTA scores."""
        counts = np.maximum(self._count, 1)
        mean = self._moments / counts
        return np.maximum(0, self._squares / counts - mean * mean).astype(np.float32)
