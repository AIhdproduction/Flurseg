from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flurseg.config import load_config
from flurseg.paths import RepoPaths
from flurseg.uncertainty import calculate


def _config(weights: tuple[float, float, float, float, float], threshold: float = 1.0, min_region_px: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        weights=SimpleNamespace(
            top_score=weights[0], margin=weights[1], priority_conflict=weights[2], tta=weights[3], rule_change=weights[4]
        ),
        threshold=threshold,
        min_region_px=min_region_px,
    )


def _calculate(weights: tuple[float, float, float, float, float], **values: object):
    return calculate(
        np.array([[1, 255]], np.uint8),
        np.asarray(values.get("top", [[0.2, 0.9]]), np.float32),
        np.asarray(values.get("margin", [[0.3, 0.8]]), np.float32),
        np.asarray(values.get("conflict", [[0.0, 0.0]]), np.float32),
        np.asarray(values.get("tta", [[0.1, 0.2]]), values.get("tta_dtype", np.float32)),
        np.asarray(values.get("change", [[0, 1]]), np.uint8),
        _config(weights, float(values.get("threshold", 1.0))),
        float(values.get("priority_margin", 1.0)),
    )


def test_top_score_only_is_one_minus_top_score() -> None:
    assert np.allclose(_calculate((1, 0, 0, 0, 0)).uncertainty, [[0.8, 0.1]])


def test_margin_only_is_one_minus_margin() -> None:
    assert np.allclose(_calculate((0, 1, 0, 0, 0)).uncertainty, [[0.7, 0.2]])


def test_priority_conflict_only_is_below_threshold_within_margin() -> None:
    result = _calculate((0, 0, 1, 0, 0), conflict=[[0.1, 0.0]], threshold=0.5, priority_margin=0.15)
    assert result.mask[0, 0] == 1


def test_conflict_above_priority_margin_needs_a_large_enough_region(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = calculate(
        np.array([[1]], np.uint8),
        np.array([[1.0]], np.float32),
        np.array([[1.0]], np.float32),
        np.array([[0.3]], np.float32),
        np.array([[0.0]], np.float32),
        np.array([[0]], np.uint8),
        cfg.pipeline.uncertainty,
        cfg.pipeline.fusion.priority_margin,
    )
    assert result.mask[0, 0] == 1


def test_uint8_tta_only_is_scaled_to_unit_interval() -> None:
    result = _calculate((0, 0, 0, 1, 0), tta=[[64, 255]], tta_dtype=np.uint8)
    assert np.allclose(result.uncertainty, [[64 / 255, 1.0]])


def test_rule_change_only_is_binary() -> None:
    assert np.array_equal(_calculate((0, 0, 0, 0, 1)).uncertainty, np.array([[0.0, 1.0]], np.float32))


def test_weights_are_normalized() -> None:
    assert np.allclose(_calculate((2, 2, 0, 0, 0)).uncertainty, _calculate((1, 1, 0, 0, 0)).uncertainty)


def test_threshold_marks_uncertain_and_preserves_existing_ignore() -> None:
    assert np.array_equal(_calculate((1, 0, 0, 0, 0), threshold=0.5).mask, np.array([[255, 255]], np.uint8))


def test_zero_weights_raise_value_error() -> None:
    with pytest.raises(ValueError, match="positive"):
        _calculate((0, 0, 0, 0, 0))


def test_uncertainty_ignores_only_large_connected_regions() -> None:
    config = _config((1, 0, 0, 0, 0), threshold=0.5, min_region_px=64)
    mask = np.full((30, 30), 2, np.uint8)
    top = np.ones((30, 30), np.float32)
    top[1:3, 1:6] = 0.0
    top[10:30, 5:30] = 0.0
    result = calculate(mask, top, np.ones_like(top), np.zeros_like(top), np.zeros_like(top), np.zeros_like(mask), config, 1.0)
    assert np.all(result.mask[1:3, 1:6] == 2)
    assert np.all(result.mask[10:30, 5:30] == 255)
