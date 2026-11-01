"""Selective privacy blur coverage."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flurseg.ingest.anonymize import anonymize
from flurseg.segment.fake_backend import FakeBackend


def _settings() -> tuple[SimpleNamespace, SimpleNamespace]:
    return (
        SimpleNamespace(tile_size=64, min_overlap=0.25),
        SimpleNamespace(min_score=0.5, blur_kernel=11, blur_sigma=0.0),
    )


def test_anonymize_changes_only_detected_person_region() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[25:55, 25:55] = (240, 10, 10)
    tiling, blur = _settings()
    result, _ = anonymize(image, FakeBackend(), ["person"], tiling, blur)
    changed = np.any(result != image, axis=2)
    assert not changed[:19].any()
    assert not changed[61:].any()
    assert not changed[:, :19].any()
    assert not changed[:, 61:].any()


def test_anonymize_reports_nonzero_blurred_fraction() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[25:55, 25:55] = (240, 10, 10)
    tiling, blur = _settings()
    _, fraction = anonymize(image, FakeBackend(), ["person"], tiling, blur)
    assert fraction > 0


def test_non_privacy_content_remains_unchanged() -> None:
    image = np.full((32, 32, 3), (10, 220, 10), dtype=np.uint8)
    tiling, blur = _settings()
    result, fraction = anonymize(image, FakeBackend(), ["person"], tiling, blur)
    assert fraction == 0
    assert np.array_equal(result, image)
