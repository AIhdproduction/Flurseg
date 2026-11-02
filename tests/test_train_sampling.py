"""CPU regression tests for rare-class sampling and augmentation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from flurseg.train.config import DataConfig, ModelConfig, SamplingConfig, TrainConfig
from flurseg.train.data import SegmentationDataset, _crop_or_pad, augment_pair
from flurseg.train.engine import resolve_total_steps


def _write_dataset(root: Path, rare_tiles: set[int], count: int = 20) -> Path:
    """Create a deterministic dataset with optional curb tiles."""
    dataset = root / "dataset"
    metadata: dict[str, dict[str, str]] = {}
    for index in range(count):
        name = f"tile-{index:02d}"
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        if index in rare_tiles:
            mask[:16, :16] = 8
        for folder in ("images", "masks"):
            (dataset / "train" / folder).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dataset / "train" / "images" / f"{name}.png"), image)
        cv2.imwrite(str(dataset / "train" / "masks" / f"{name}.png"), mask)
        metadata[name] = {"source": "drone", "split": "train", "session": "synthetic"}
    manifest = {
        "ignore_index": 255,
        "classes": [{"id": 0, "name": "other"}, {"id": 8, "name": "curb"}],
        "tile_metadata": metadata,
    }
    (dataset / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


def _dataset(root: Path, rare_tiles: set[int] | None = None) -> SegmentationDataset:
    """Return a training dataset with repeat sampling but no image augmentation effects."""
    rare_tiles = {0} if rare_tiles is None else rare_tiles
    return SegmentationDataset(
        _write_dataset(root, rare_tiles),
        "train",
        DataConfig(
            crop_size=32,
            scale_min=1,
            scale_max=1,
            brightness=0,
            contrast=0,
            saturation=0,
            gamma_min=1,
            gamma_max=1,
            hue=0,
            gaussian_blur_prob=0,
            gaussian_noise_prob=0,
        ),
        ModelConfig(pretrained="offline"),
        training=True,
        sampling_config=SamplingConfig(min_class_pixels=200, threshold=0.3, boost_factor=1.5),
        sampling_seed=41,
    )


def test_repeat_factors_boost_rare_tiles_and_leave_common_tiles_at_one(tmp_path: Path) -> None:
    """A one-in-twenty curb tile receives its boosted LVIS repeat factor."""
    dataset = _dataset(tmp_path)
    expected = math.sqrt(0.3 / 0.05) * 1.5
    assert dataset.tile_repeats[0] == np.minimum(5.0, expected)
    assert dataset.tile_repeats[1] == 1.0


def test_epoch_sampling_is_deterministic_and_matches_expected_repeat(tmp_path: Path) -> None:
    """Stochastic rounding is stable per epoch and unbiased over many epochs."""
    dataset = _dataset(tmp_path)
    dataset.set_epoch(7)
    first = list(dataset._epoch_image_indices)
    dataset.set_epoch(7)
    assert dataset._epoch_image_indices == first
    dataset.set_epoch(8)
    assert dataset._epoch_image_indices != first
    counts = []
    for epoch in range(500):
        dataset.set_epoch(epoch)
        counts.append(dataset._epoch_image_indices.count(0))
    np.testing.assert_allclose(np.mean(counts), dataset.tile_repeats[0], rtol=0.1)


def test_rare_crop_always_contains_a_rare_pixel_when_required() -> None:
    """Rare crop anchoring retains a tiny curb through randomized crop placement."""
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[91, 87] = 8
    for seed in range(100):
        _image, crop = _crop_or_pad(image, mask, 32, np.random.default_rng(seed), {8}, 1.0)
        assert np.any(crop == 8)


def test_photometric_augmentation_preserves_masks_and_varies_both_directions() -> None:
    """Photometric changes preserve masks while producing darker and brighter outputs."""
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    config = DataConfig(
        crop_size=32,
        scale_min=1,
        scale_max=1,
        brightness=0.3,
        contrast=0,
        saturation=0,
        gamma_min=0.7,
        gamma_max=1.4,
        hue=0,
        gaussian_blur_prob=0,
        gaussian_noise_prob=0,
        phone_rotation_degrees=0,
    )
    baseline = DataConfig.model_validate(
        {**config.model_dump(), "brightness": 0, "gamma_min": 1, "gamma_max": 1}
    )
    means = []
    for seed in range(100):
        augmented, returned_mask = augment_pair(image, mask, "phone", config, np.random.default_rng(seed))
        _baseline_image, baseline_mask = augment_pair(
            image, mask, "phone", baseline, np.random.default_rng(seed)
        )
        assert np.array_equal(returned_mask, baseline_mask)
        assert augmented.dtype == np.uint8 and augmented.min() >= 0 and augmented.max() <= 255
        means.append(float(augmented.mean()))
    assert min(means) < image.mean() < max(means)


def test_source_geometry_preserves_orientation_rules_and_pair_alignment() -> None:
    """Phone stays upright while nadir sees all dihedral variants with aligned labels."""
    pattern = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image = np.dstack([pattern, pattern, pattern])
    stable = DataConfig(
        crop_size=8,
        scale_min=1,
        scale_max=1,
        brightness=0,
        contrast=0,
        saturation=0,
        gamma_min=1,
        gamma_max=1,
        hue=0,
        gaussian_blur_prob=0,
        gaussian_noise_prob=0,
        phone_rotation_degrees=0,
    )
    phone = {
        augment_pair(image, pattern, "phone", stable, np.random.default_rng(seed))[1].tobytes()
        for seed in range(40)
    }
    assert phone <= {pattern.tobytes(), pattern[:, ::-1].tobytes()}
    nadir = {
        augment_pair(image, pattern, "drone", stable, np.random.default_rng(seed))[1].tobytes()
        for seed in range(100)
    }
    assert len(nadir) == 8
    rotated = DataConfig.model_validate({**stable.model_dump(), "crop_size": 32, "phone_rotation_degrees": 8})
    large_pattern = np.zeros((32, 32), dtype=np.uint8)
    large_pattern[:, 16:] = 1
    large_image = np.dstack([large_pattern * 200] * 3)
    phone_image, phone_mask = augment_pair(
        large_image, large_pattern, "phone", rotated, np.random.default_rng(3)
    )
    valid = phone_mask != 255
    image_foreground = phone_image[:, :, 0] > 175
    image_background = phone_image[:, :, 0] < 25
    assert np.all(phone_mask[image_foreground & valid] == 1)
    assert np.all(phone_mask[image_background & valid] == 0)
    assert np.any(phone_mask == 255)


def test_auto_total_steps_uses_resampled_epoch_length(tmp_path: Path) -> None:
    """Automatic step count receives the stochastic-rounded resampled epoch length."""
    dataset = _dataset(tmp_path)
    config = TrainConfig.model_validate(
        {
            "model": {"pretrained": "offline"},
            "data": {"crop_size": 32},
            "inference": {"window_size": 32, "stride": 16},
            "training": {"batch_size": 1, "gradient_accumulation": 1, "epochs_target": 100},
        }
    )
    assert resolve_total_steps(config, len(dataset)) == round(100 * len(dataset))
