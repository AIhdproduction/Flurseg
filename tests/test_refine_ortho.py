from __future__ import annotations

import numpy as np

from flurseg.refine.ortho_rules import apply


def test_ortho_removes_small_cover() -> None:
    config = type("C", (), {"min_small_object_m": 0.4, "min_marking_width_m": 0.3})()
    assert apply(np.array([[10]], np.uint8), "ortho", 0.1, config, {"utility_cover": 10})[0, 0] == 255


def test_ortho_keeps_large_cover_and_skips_non_ortho_source() -> None:
    config = type("C", (), {"min_small_object_m": 0.4, "min_marking_width_m": 0.3})()
    ids = {"utility_cover": 10}
    assert np.all(apply(np.full((5, 5), 10, np.uint8), "ortho", 0.1, config, ids) == 10)
    assert apply(np.array([[10]], np.uint8), "drone", 0.1, config, ids)[0, 0] == 10


def test_ortho_removes_thin_marking_but_keeps_every_pixel_of_wide_marking() -> None:
    config = type("C", (), {"min_small_object_m": 0.4, "min_marking_width_m": 0.3})()
    ids = {"road_marking": 9}
    thin = np.full((5, 5), 255, np.uint8)
    thin[2, :] = 9
    assert not np.any(apply(thin, "ortho", 0.1, config, ids) == 9)
    wide = np.full((5, 5), 255, np.uint8)
    wide[1:4, :] = 9
    assert np.all(apply(wide, "ortho", 0.1, config, ids)[1:4, :] == 9)
