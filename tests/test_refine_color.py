from __future__ import annotations

import numpy as np

from flurseg.refine.color_rules import apply

IDS = {"ground": 1, "paved_surface": 2, "low_vegetation": 3, "road_marking": 9, "vehicle": 6}


def _config(**overrides: float) -> object:
    values: dict[str, float | int] = {
        "exg_threshold": 0.15,
        "exg_low_threshold": -0.02,
        "min_brightness": 70,
        "min_target_score": 0.25,
        "min_region_px": 3000,
    }
    values.update(overrides)
    return type("ColorConfig", (), values)()


def test_color_does_not_convert_shadowed_slightly_green_asphalt_with_low_vegetation_score() -> None:
    mask = np.full((60, 60), 2, np.uint8)
    image = np.full((60, 60, 3), (29, 38, 29), np.uint8)
    scores = {"low_vegetation": np.full(mask.shape, 0.1, np.float32)}

    output = apply(mask, image, _config(), IDS, np.ones(mask.shape, np.int32), class_scores=scores)

    assert np.array_equal(output, mask)


def test_color_converts_large_bright_grass_superpixel_with_sam_evidence() -> None:
    mask = np.full((60, 60), 2, np.uint8)
    image = np.full((60, 60, 3), (0, 255, 0), np.uint8)
    scores = {"low_vegetation": np.full(mask.shape, 0.6, np.float32)}

    output = apply(mask, image, _config(), IDS, np.ones(mask.shape, np.int32), class_scores=scores)

    assert np.all(output == IDS["low_vegetation"])


def test_color_reverts_small_converted_group() -> None:
    mask = np.full((60, 60), 2, np.uint8)
    image = np.full((60, 60, 3), 120, np.uint8)
    image[:40, :40] = (0, 255, 0)
    labels = np.full(mask.shape, 2, np.int32)
    labels[:40, :40] = 1
    scores = {"low_vegetation": np.full(mask.shape, 0.6, np.float32)}

    output = apply(mask, image, _config(), IDS, labels, class_scores=scores)

    assert np.array_equal(output, mask)


def test_color_without_scores_still_converts_and_preserves_other_classes() -> None:
    mask = np.array([[2, 2, 6], [2, 2, 6]], np.uint8)
    image = np.full((2, 3, 3), (0, 255, 0), np.uint8)
    labels = np.array([[1, 1, 2], [1, 1, 2]], np.int32)

    output = apply(mask, image, _config(min_region_px=1), IDS, labels)

    assert output[:, :2].tolist() == [[3, 3], [3, 3]]
    assert np.all(output[:, 2] == IDS["vehicle"])
