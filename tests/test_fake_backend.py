"""Deterministic fake segmentation backend coverage."""

from __future__ import annotations

import time

import numpy as np
import pytest

from flurseg.segment.fake_backend import FakeBackend
from flurseg.segment.tta import apply, invert


def _image() -> np.ndarray:
    image = np.full((9, 13, 3), 80, dtype=np.uint8)
    image[2:5, 3:8] = (220, 20, 20)
    return image


def test_fake_backend_is_bit_deterministic() -> None:
    backend = FakeBackend()
    assert np.array_equal(backend.score_tile(_image(), ["person", "tree"]), backend.score_tile(_image(), ["person", "tree"]))


def test_fake_backend_returns_expected_shape_dtype_and_range() -> None:
    scores = FakeBackend().score_tile(_image(), ["person", "tree"])
    assert scores.shape == (2, 9, 13)
    assert scores.dtype == np.float32
    assert np.all((scores >= 0) & (scores <= 1))


@pytest.mark.parametrize("variant", ["identity", "rot90", "hflip", "vflip"])
def test_fake_backend_is_equivariant_for_each_tta_variant(variant: str) -> None:
    backend = FakeBackend(equivariant=True)
    expected = backend.score_tile(_image(), ["tree"])
    transformed = backend.score_tile(apply(variant, _image()), ["tree"])
    assert np.array_equal(invert(variant, transformed), expected)


def test_fake_backend_disagreement_creates_orientation_dependent_difference() -> None:
    backend = FakeBackend(equivariant=False, disagreement=1.0)
    direct = backend.score_tile(_image(), ["tree"])
    rotated = invert("rot90", backend.score_tile(apply("rot90", _image()), ["tree"]))
    assert np.any(direct != rotated)


def test_vegetation_prompt_scores_green_high_and_grey_low() -> None:
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, :4] = (15, 220, 15)
    image[:, 4:] = (128, 128, 128)
    scores = FakeBackend().score_tile(image, ["vegetation"])[0]
    assert scores[:, :4].mean() > 0.8
    assert scores[:, 4:].mean() < 0.3


def test_speed_budget() -> None:
    backend = FakeBackend()
    tile = np.full((1008, 1008, 3), 120, dtype=np.uint8)
    tile[200:400, 300:600] = (30, 180, 40)
    tile[500:700, 100:400] = (200, 50, 50)
    prompts = [
        "grass", "tree", "forest", "vegetation", "hedge", "shrub", "crop", "lawn",
        "asphalt", "road", "sidewalk", "pavement", "parking", "concrete",
        "marking", "line", "crosswalk",
        "roof", "building", "house", "facade",
        "person", "vehicle", "car", "truck", "license",
        "bare soil", "dirt", "gravel", "sand", "mud",
        "excavator", "crane", "scaffolding", "curb", "manhole cover", "storm drain", "drain grate",
    ]
    assert len(prompts) == 38
    durations: list[float] = []
    for _ in range(3):
        start = time.perf_counter()
        scores = backend.score_tile(tile, prompts)
        durations.append(time.perf_counter() - start)
    best = min(durations)
    assert scores.shape == (38, 1008, 1008)
    assert scores.dtype == np.float32
    assert best < 0.25, f"Execution took {best:.4f}s, expected < 0.25s"
