from __future__ import annotations

import numpy as np

from flurseg.refine.color_rules import apply


def test_color_relabels_green_surface() -> None:
    config = type("C", (), {"exg_threshold": 0.08, "marking_brightness_percentile": 90})()
    output = apply(np.array([[1]], np.uint8), np.array([[[0, 255, 0]]], np.uint8), config, {"ground": 1, "paved_surface": 2, "low_vegetation": 3, "road_marking": 9})
    assert output[0, 0] == 3


def test_color_relabels_neutral_vegetation_to_ground_and_preserves_other_classes() -> None:
    config = type("C", (), {"exg_threshold": 0.08, "marking_brightness_percentile": 90})()
    ids = {"ground": 1, "paved_surface": 2, "low_vegetation": 3, "road_marking": 9, "vehicle": 6}
    mask = np.array([[3, 6]], np.uint8)
    image = np.array([[[128, 128, 128], [0, 255, 0]]], np.uint8)
    output = apply(mask, image, config, ids, np.array([[1, 2]], np.int32))
    assert output.tolist() == [[1, 6]]


def test_color_does_not_grow_markings_into_sunlit_paved_surface_by_default() -> None:
    config = type("C", (), {"exg_threshold": 0.08, "marking_brightness_percentile": 50})()
    ids = {"ground": 1, "paved_surface": 2, "low_vegetation": 3, "road_marking": 9}
    mask = np.array([[2, 9, 2], [2, 2, 2]], np.uint8)
    image = np.array([[[240, 240, 240], [5, 5, 5], [240, 240, 240]], [[20, 20, 20]] * 3], np.uint8)
    output = apply(mask, image, config, ids, np.array([[1, 2, 3], [1, 1, 1]], np.int32))
    assert output[0, 0] == 2 and output[0, 1] == 9


def test_color_grows_only_a_high_contrast_gap_superpixel_inside_a_marking() -> None:
    config = type(
        "C",
        (),
        {"exg_threshold": 0.08, "marking_brightness_percentile": 90, "marking_growth": True, "marking_min_contrast": 40, "marking_max_saturation": 40},
    )()
    ids = {"ground": 1, "paved_surface": 2, "low_vegetation": 3, "road_marking": 9}
    mask = np.full((15, 13), 2, np.uint8)
    mask[7, 3:5] = 9
    mask[7, 8:10] = 9
    image = np.full((15, 13, 3), 80, np.uint8)
    image[7, 5:8] = 200
    image[:7, 5:8] = 200
    superpixels = np.ones((15, 13), np.int32)
    superpixels[7, 5:8] = 2
    superpixels[:7, 5:8] = 3
    output = apply(mask, image, config, ids, superpixels)
    assert np.all(output[7, 5:8] == 9)
    assert np.all(output[:7, 5:8] == 2)


def test_color_never_changes_an_isolated_pixel_and_changes_a_whole_green_superpixel() -> None:
    config = type("C", (), {"exg_threshold": 0.08, "marking_brightness_percentile": 90})()
    ids = {"ground": 1, "paved_surface": 2, "low_vegetation": 3, "road_marking": 9}
    mask = np.full((5, 12), 2, np.uint8)
    image = np.full((5, 12, 3), 100, np.uint8)
    image[0, 0] = (0, 255, 0)
    image[:, 6:] = (0, 255, 0)
    superpixels = np.ones((5, 12), np.int32)
    superpixels[:, 6:] = 2
    output = apply(mask, image, config, ids, superpixels)
    assert output[0, 0] == 2
    assert np.all(output[:, 6:] == 3)
