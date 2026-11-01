from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flurseg.refine.shape_rules import apply


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        utility_cover_diameter_m=(2.0, 4.0),
        utility_cover_min_compactness=0.65,
        vehicle_area_m2=(3.0, 8.0),
    )


def test_cover_diameter_below_range_becomes_ignore() -> None:
    mask = np.array([[10]], dtype=np.uint8)
    assert apply(mask, 1.0, _config(), {"utility_cover": 10})[0, 0] == 255


def test_compact_cover_inside_diameter_range_is_kept() -> None:
    mask = np.full((3, 3), 10, dtype=np.uint8)
    assert np.array_equal(apply(mask, 1.0, _config(), {"utility_cover": 10}), mask)


def test_cover_diameter_above_range_becomes_ignore() -> None:
    mask = np.full((5, 5), 10, dtype=np.uint8)
    assert np.all(apply(mask, 1.0, _config(), {"utility_cover": 10}) == 255)


def test_non_compact_cover_with_valid_area_becomes_ignore() -> None:
    mask = np.zeros((7, 7), dtype=np.uint8)
    mask[1:6, 1] = 10
    mask[5, 1:6] = 10
    assert np.all(apply(mask, 1.0, _config(), {"utility_cover": 10})[mask == 10] == 255)


def test_vehicle_area_below_range_becomes_ignore() -> None:
    mask = np.full((1, 2), 6, dtype=np.uint8)
    assert np.all(apply(mask, 1.0, _config(), {"vehicle": 6}) == 255)


def test_vehicle_area_inside_range_is_kept() -> None:
    mask = np.full((2, 2), 6, dtype=np.uint8)
    assert np.array_equal(apply(mask, 1.0, _config(), {"vehicle": 6}), mask)


def test_vehicle_area_above_range_becomes_ignore() -> None:
    mask = np.full((3, 3), 6, dtype=np.uint8)
    assert np.all(apply(mask, 1.0, _config(), {"vehicle": 6}) == 255)


def test_missing_gsd_leaves_mask_unchanged() -> None:
    mask = np.array([[10, 6], [6, 10]], dtype=np.uint8)
    output = apply(mask, None, _config(), {"utility_cover": 10, "vehicle": 6})
    assert np.array_equal(output, mask) and output is not mask
