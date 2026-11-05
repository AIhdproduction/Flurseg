"""Tests for deterministic equirectangular perspective crops."""

from __future__ import annotations

import numpy as np

from flurseg.samples.extra.equirect import equirect_to_perspective


def test_equirect_yaw_wrap_shape_and_fov() -> None:
    """Yaw follows longitude, wraps cleanly, and field of view changes the sampled extent."""
    width, height = 360, 180
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)
    centre = equirect_to_perspective(image, 0, 0, 70, (61, 41))
    right = equirect_to_perspective(image, 90, 0, 70, (61, 41))
    seam = equirect_to_perspective(image, 180, 0, 120, (61, 41))
    narrow = equirect_to_perspective(image, 0, 0, 45, (61, 41))
    assert centre.shape == (41, 61, 3)
    assert abs(int(centre[20, 30, 0]) - 180) < 3
    assert abs(int(right[20, 30, 0]) - (270 % 256)) < 3
    assert seam[20, 30, 0] < 3 and seam[20, 0, 0] != seam[20, -1, 0]
    assert int(narrow[20, 0, 0]) > int(centre[20, 0, 0])


def _longitude_image(width: int = 200, height: int = 100) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)
    image[..., 1] = np.arange(height, dtype=np.uint8)[:, None]
    return image


def test_yaw_zero_pitch_zero_samples_the_equirectangular_centre() -> None:
    image = _longitude_image()
    view = equirect_to_perspective(image, 0, 0, 75, (51, 31))
    assert abs(int(view[15, 25, 0]) - image.shape[1] // 2) < 2


def test_yaw_turns_shift_the_centre_by_a_quarter_turn() -> None:
    image = _longitude_image()
    east = equirect_to_perspective(image, 90, 0, 75, (51, 31))
    west = equirect_to_perspective(image, -90, 0, 75, (51, 31))
    assert abs(int(east[15, 25, 0]) - 150) < 2
    assert abs(int(west[15, 25, 0]) - 50) < 2


def test_yaw_180_wraps_at_the_horizontal_seam_without_black_pixels() -> None:
    image = _longitude_image()
    view = equirect_to_perspective(image, 180, 0, 100, (81, 41))
    assert int(view[20, 40, 0]) < 2
    assert int(view[20, 0, 0]) > 5 and int(view[20, -1, 0]) > 5


def test_output_size_dtype_and_values_are_deterministic() -> None:
    image = _longitude_image(180, 90)
    first = equirect_to_perspective(image, 37, -4, 80, (73, 29))
    second = equirect_to_perspective(image, 37, -4, 80, (73, 29))
    small = equirect_to_perspective(image, 37, -4, 80, (17, 11))
    assert first.shape == (29, 73, 3) and first.dtype == np.uint8
    assert small.shape == (11, 17, 3) and small.dtype == np.uint8
    assert np.array_equal(first, second)


def test_larger_fov_covers_a_wider_longitude_range() -> None:
    image = _longitude_image()
    narrow = equirect_to_perspective(image, 0, 0, 40, (101, 41))
    wide = equirect_to_perspective(image, 0, 0, 120, (101, 41))
    narrow_span = abs(int(narrow[20, -1, 0]) - int(narrow[20, 0, 0]))
    wide_span = abs(int(wide[20, -1, 0]) - int(wide[20, 0, 0]))
    assert wide_span > narrow_span + 20


def test_negative_pitch_samples_rows_below_the_horizon() -> None:
    image = _longitude_image()
    horizon = equirect_to_perspective(image, 0, 0, 70, (61, 41))
    downward = equirect_to_perspective(image, 0, -10, 70, (61, 41))
    assert int(downward[20, 30, 1]) > int(horizon[20, 30, 1]) + 3
