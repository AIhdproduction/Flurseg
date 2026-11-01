"""Tests for training tile extraction, padding, and ignore-filtering."""

from __future__ import annotations

import math
from inspect import isgenerator

import numpy as np
import pytest

from flurseg.dataset.tiles import (
    TrainingTile,
    extract_tiles,
    iter_training_tiles,
)
from flurseg.segment.tiling import tile_starts


def test_tile_grid_coverage_large_image() -> None:
    height, width = 1500, 2200
    tile_size = 512
    image = np.full((height, width, 3), 100, dtype=np.uint8)
    mask = np.ones((height, width), dtype=np.uint8)

    tiles = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.9)

    expected_y_starts = tile_starts(height, tile_size, 0.0)
    expected_x_starts = tile_starts(width, tile_size, 0.0)
    expected_count = len(expected_y_starts) * len(expected_x_starts)
    assert len(tiles) == expected_count

    # Check that the union of tiles covers the entire original image
    covered = np.zeros((height, width), dtype=bool)
    for tile in tiles:
        assert isinstance(tile, TrainingTile)
        assert tile.width == tile_size
        assert tile.height == tile_size
        assert tile.image.shape == (tile_size, tile_size, 3)
        assert tile.mask.shape == (tile_size, tile_size)
        assert tile.pad == (0, 0, 0, 0)
        assert tile.original_size == (height, width)
        assert tile.ignore_fraction == 0.0

        covered[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width] = True

        expected_crop_img = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        expected_crop_mask = mask[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        np.testing.assert_array_equal(tile.image, expected_crop_img)
        np.testing.assert_array_equal(tile.mask, expected_crop_mask)

    assert covered.all()


def test_exact_tile_size_image_requires_no_padding() -> None:
    tile_size = 1024
    image = np.full((tile_size, tile_size, 3), 42, dtype=np.uint8)
    mask = np.full((tile_size, tile_size), 5, dtype=np.uint8)

    tiles = extract_tiles(image, mask, tile_size=tile_size)

    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.index == 0
    assert tile.x == 0
    assert tile.y == 0
    assert tile.width == tile_size
    assert tile.height == tile_size
    assert tile.pad == (0, 0, 0, 0)
    assert tile.pad_top == 0
    assert tile.pad_bottom == 0
    assert tile.pad_left == 0
    assert tile.pad_right == 0
    assert tile.original_size == (tile_size, tile_size)
    assert tile.ignore_fraction == 0.0
    np.testing.assert_array_equal(tile.image, image)
    np.testing.assert_array_equal(tile.mask, mask)


def test_small_image_padding_with_255_mask_and_black_image() -> None:
    orig_h, orig_w = 400, 600
    tile_size = 1024
    image = np.full((orig_h, orig_w, 3), 128, dtype=np.uint8)
    mask = np.full((orig_h, orig_w), 2, dtype=np.uint8)

    # Padding adds: bottom = 1024 - 400 = 624, right = 1024 - 600 = 424
    # Ignore fraction is 1 - (400*600)/(1024*1024) = 1 - 240000/1048576 ≈ 0.7711
    tiles = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.90)

    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.index == 0
    assert tile.x == 0
    assert tile.y == 0
    assert tile.width == tile_size
    assert tile.height == tile_size
    assert tile.pad == (0, 624, 0, 424)
    assert tile.pad_top == 0
    assert tile.pad_bottom == 624
    assert tile.pad_left == 0
    assert tile.pad_right == 424
    assert tile.original_size == (orig_h, orig_w)

    expected_ignore_fraction = (tile_size * tile_size - orig_h * orig_w) / (tile_size * tile_size)
    assert math.isclose(tile.ignore_fraction, expected_ignore_fraction, rel_tol=1e-5)

    # Check original image and mask content preserved
    np.testing.assert_array_equal(tile.image[:orig_h, :orig_w], image)
    np.testing.assert_array_equal(tile.mask[:orig_h, :orig_w], mask)

    # Check padded regions: image is black (0), mask is ignore (255)
    np.testing.assert_array_equal(tile.image[orig_h:, :], 0)
    np.testing.assert_array_equal(tile.image[:, orig_w:], 0)
    np.testing.assert_array_equal(tile.mask[orig_h:, :], 255)
    np.testing.assert_array_equal(tile.mask[:, orig_w:], 255)


def test_small_grayscale_image_padding() -> None:
    orig_h, orig_w = 300, 350
    tile_size = 512
    image = np.full((orig_h, orig_w), 80, dtype=np.uint8)
    mask = np.full((orig_h, orig_w), 1, dtype=np.uint8)

    tiles = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.90)

    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.image.ndim == 2
    assert tile.image.shape == (tile_size, tile_size)
    assert tile.mask.shape == (tile_size, tile_size)

    np.testing.assert_array_equal(tile.image[:orig_h, :orig_w], image)
    np.testing.assert_array_equal(tile.mask[:orig_h, :orig_w], mask)
    np.testing.assert_array_equal(tile.image[orig_h:, :], 0)
    np.testing.assert_array_equal(tile.image[:, orig_w:], 0)
    np.testing.assert_array_equal(tile.mask[orig_h:, :], 255)
    np.testing.assert_array_equal(tile.mask[:, orig_w:], 255)


def test_one_dimension_smaller_and_one_dimension_larger() -> None:
    height, width = 1200, 800
    tile_size = 1024
    image = np.ones((height, width, 3), dtype=np.uint8)
    mask = np.ones((height, width), dtype=np.uint8)

    # Height > 1024 -> pad_bottom = 0; Width < 1024 -> pad_right = 224
    tiles = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.90)

    y_starts = tile_starts(height, tile_size, 0.0)
    assert len(y_starts) == 2
    assert len(tiles) == 2

    for tile in tiles:
        assert tile.pad == (0, 0, 0, 224)
        assert tile.original_size == (height, width)
        assert tile.width == tile_size
        assert tile.height == tile_size
        # The right 224 pixels are padded
        np.testing.assert_array_equal(tile.mask[:, width:], 255)
        np.testing.assert_array_equal(tile.image[:, width:], 0)


def test_ignore_heavy_tiles_dropped() -> None:
    tile_size = 512
    # Two tiles side by side: left (0, 0) and right (0, 512)
    image = np.zeros((512, 1024, 3), dtype=np.uint8)
    mask = np.zeros((512, 1024), dtype=np.uint8)

    # Left tile has 95% ignore pixels (> 0.90)
    left_rows = int(512 * 0.95)
    mask[:left_rows, :512] = 255

    # Right tile has 10% ignore pixels (<= 0.90)
    right_rows = int(512 * 0.10)
    mask[:right_rows, 512:] = 255

    tiles = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.90)

    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.index == 0
    assert tile.x == 512
    assert tile.y == 0
    assert tile.ignore_fraction <= 0.90


def test_all_tiles_dropped_when_completely_ignore() -> None:
    image = np.zeros((1024, 1024, 3), dtype=np.uint8)
    mask = np.full((1024, 1024), 255, dtype=np.uint8)

    tiles = extract_tiles(image, mask, tile_size=1024, max_ignore_fraction=0.90)
    assert tiles == []


def test_small_image_dropped_when_padding_exceeds_max_ignore_fraction() -> None:
    # 100 x 100 in 1024 x 1024 -> ignore fraction is ~99.05%
    image = np.ones((100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)

    tiles = extract_tiles(image, mask, tile_size=1024, max_ignore_fraction=0.90)
    assert tiles == []


def test_boundary_ignore_fraction_threshold() -> None:
    tile_size = 100
    image = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    mask = np.zeros((tile_size, tile_size), dtype=np.uint8)

    # Set exactly 50% to ignore
    mask[:50, :] = 255

    # Exactly 0.50 is NOT dropped with max_ignore_fraction=0.50 (strict > check)
    tiles_kept = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.50)
    assert len(tiles_kept) == 1
    assert math.isclose(tiles_kept[0].ignore_fraction, 0.50)

    # But it IS dropped if max_ignore_fraction=0.49
    tiles_dropped = extract_tiles(image, mask, tile_size=tile_size, max_ignore_fraction=0.49)
    assert len(tiles_dropped) == 0


def test_iter_training_tiles_returns_generator() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)

    gen = iter_training_tiles(image, mask, tile_size=200)
    assert isgenerator(gen)
    tile_list = list(gen)
    assert len(tile_list) == 1


def test_shape_mismatch_raises_value_error() -> None:
    image = np.zeros((400, 500, 3), dtype=np.uint8)
    mask = np.zeros((400, 501), dtype=np.uint8)

    with pytest.raises(ValueError, match="must match"):
        extract_tiles(image, mask, tile_size=256)


def test_invalid_tile_size_raises_value_error() -> None:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    mask = np.zeros((200, 200), dtype=np.uint8)

    with pytest.raises(ValueError, match="tile_size must be positive"):
        extract_tiles(image, mask, tile_size=0)

    with pytest.raises(ValueError, match="tile_size must be positive"):
        extract_tiles(image, mask, tile_size=-100)
