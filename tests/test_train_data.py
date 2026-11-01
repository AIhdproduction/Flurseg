"""Regression tests for paired training augmentation."""

from __future__ import annotations

import numpy as np

from flurseg.train.config import DataConfig
from flurseg.train.data import _crop_or_pad, augment_pair


def test_augmentation_keeps_image_and_mask_aligned_for_multiple_seeds() -> None:
    """Every paired geometric augmentation preserves a synthetic pixel pattern."""
    pattern = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image = np.dstack([pattern, pattern, pattern])
    config = DataConfig(crop_size=8, scale_min=1, scale_max=1, color_jitter=0)
    for seed in range(20):
        augmented_image, augmented_mask = augment_pair(
            image,
            pattern,
            "drone",
            config,
            np.random.default_rng(seed),
        )
        assert np.array_equal(augmented_image[:, :, 0], augmented_mask)


def test_crop_or_pad_marks_mask_padding_as_ignore() -> None:
    """Mask padding is exactly 255 while image padding remains black."""
    image = np.full((3, 4, 3), 17, dtype=np.uint8)
    mask = np.full((3, 4), 4, dtype=np.uint8)
    padded_image, padded_mask = _crop_or_pad(
        image,
        mask,
        crop_size=8,
        rng=np.random.default_rng(1),
    )
    assert padded_image.shape == (8, 8, 3)
    assert padded_mask.shape == (8, 8)
    assert np.all(padded_mask[3:, :] == 255)
    assert np.all(padded_mask[:, 4:] == 255)
    assert np.all(padded_image[3:, :] == 0)
    assert np.all(padded_image[:, 4:] == 0)


def test_nadir_tiles_can_rotate_or_flip_vertically() -> None:
    """At least one deterministic nadir seed applies a non-phone orientation change."""
    pattern = np.arange(16, dtype=np.uint8).reshape(4, 4)
    image = np.dstack([pattern, pattern, pattern])
    config = DataConfig(crop_size=4, scale_min=1, scale_max=1, color_jitter=0)
    phone_orientations = {pattern.tobytes(), pattern[:, ::-1].tobytes()}
    transformed = {
        augment_pair(image, pattern, "ortho", config, np.random.default_rng(seed))[1].tobytes()
        for seed in range(30)
    }
    assert any(value not in phone_orientations for value in transformed)
