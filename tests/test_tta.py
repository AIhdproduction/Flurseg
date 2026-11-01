"""Pixel-exact TTA transform coverage."""

from __future__ import annotations

import numpy as np
import pytest

from flurseg.segment.tta import apply, invert


@pytest.mark.parametrize("variant", ["identity", "rot90", "hflip", "vflip"])
def test_tta_image_inverse_is_bit_exact(variant: str) -> None:
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    assert np.array_equal(invert(variant, apply(variant, image)), image)


@pytest.mark.parametrize("variant", ["identity", "rot90", "hflip", "vflip"])
def test_tta_score_stack_inverse_is_bit_exact(variant: str) -> None:
    stack = np.arange(3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7)
    assert np.array_equal(invert(variant, apply(variant, stack)), stack)


@pytest.mark.parametrize("variant", ["identity", "rot90", "hflip", "vflip"])
def test_tta_inverse_is_bit_exact_for_non_square_input(variant: str) -> None:
    image = np.arange(2 * 9 * 3, dtype=np.uint8).reshape(2, 9, 3)
    assert np.array_equal(invert(variant, apply(variant, image)), image)


def test_rot90_moves_top_left_corner_to_expected_corner() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 0] = (255, 0, 0)
    rotated = apply("rot90", image)
    assert np.array_equal(rotated[2, 0], (255, 0, 0))
