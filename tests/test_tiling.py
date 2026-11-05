"""Tile placement, accumulation, and padding coverage."""

from __future__ import annotations

from itertools import pairwise
from math import ceil

import numpy as np
import pytest

from flurseg.segment.tiling import TileAccumulator, iter_tiles, pad_to_tile, tile_starts


@pytest.mark.parametrize(
    ("dimension", "tile", "overlap"),
    [
        (1, 8, 0.25), (7, 8, 0.25), (8, 8, 0.25), (9, 8, 0.25),
        (15, 8, 0.25), (16, 8, 0.25), (17, 8, 0.25), (31, 16, 0.25),
        (32, 16, 0.25), (33, 16, 0.25), (100, 32, 0.25), (101, 32, 0.25),
        (255, 64, 0.25), (256, 64, 0.25), (257, 64, 0.25), (999, 100, 0.25),
        (1000, 100, 0.25), (1001, 100, 0.25), (4096, 1008, 0.25), (9999, 2048, 0.25),
    ],
)
def test_tile_starts_cover_dimension_with_even_required_overlap(
    dimension: int, tile: int, overlap: float
) -> None:
    starts = tile_starts(dimension, tile, overlap)
    assert starts[0] == 0
    assert starts[-1] + min(dimension, tile) == dimension
    coverage = np.zeros(dimension, dtype=bool)
    for start in starts:
        coverage[start : start + min(dimension, tile)] = True
    assert coverage.all()
    steps = [right - left for left, right in pairwise(starts)]
    if steps:
        assert max(steps) - min(steps) <= 1
        assert all(tile - step >= ceil(tile * overlap) for step in steps)


@pytest.mark.parametrize("reduction", ["max", "mean"])
def test_accumulator_keeps_constant_input_constant_without_seams(reduction: str) -> None:
    width, height, tile = 150, 120, 100
    accumulator = TileAccumulator(2, width, height, reduction)  # type: ignore[arg-type]
    for item in iter_tiles(width, height, tile, 0.25):
        accumulator.add(item, np.full((2, item.height, item.width), 0.37, dtype=np.float32))
    assert np.allclose(accumulator.scores(), 0.37)


def test_pad_to_tile_and_crop_back_is_identity() -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    padded, slices = pad_to_tile(image, 8)
    assert padded.shape == (8, 8, 3)
    assert np.array_equal(padded[slices[0], slices[1]], image)


@pytest.mark.parametrize("width,height", [(1, 1), (1008, 1008), (1009, 1008), (4896, 3672)])
def test_tiles_cover_images_with_ten_percent_overlap(width: int, height: int) -> None:
    tile = 1008
    overlap = 0.10
    tiles = list(iter_tiles(width, height, tile, overlap))
    assert tiles
    assert max(item.x + item.width for item in tiles) == width
    assert max(item.y + item.height for item in tiles) == height
    for starts in (tile_starts(width, tile, overlap), tile_starts(height, tile, overlap)):
        assert all(right - left <= tile - ceil(tile * overlap) for left, right in pairwise(starts))
