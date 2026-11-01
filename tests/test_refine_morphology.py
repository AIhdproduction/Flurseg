from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from flurseg.refine.morphology import apply
from flurseg.refine.registry import RefinementContext, refine


def test_morphology_uses_pixel_minimums_without_gsd() -> None:
    mask = np.array([[2, 2, 2], [2, 6, 2], [2, 2, 2]], np.uint8)
    config = type("C", (), {"min_island_m2": {"vehicle": 2.0}, "min_component_px": {"vehicle": 2}, "max_hole_m2": 0.0})()
    assert apply(mask, None, config, {"paved_surface": 2, "vehicle": 6})[1, 1] == 2


def test_morphology_removes_small_island_and_keeps_large_one() -> None:
    config = type("C", (), {"min_island_m2": {"vehicle": 2.0}, "min_component_px": {"vehicle": 2}, "max_hole_m2": 0.0})()
    ids = {"vehicle": 6}
    assert apply(np.array([[6]], np.uint8), 1.0, config, ids)[0, 0] == 255
    large = np.full((2, 2), 6, np.uint8)
    assert np.array_equal(apply(large, 1.0, config, ids), large)


def test_morphology_fills_only_enclosed_ignore_holes() -> None:
    config = type("C", (), {"min_island_m2": {"paved_surface": 2.0}, "min_component_px": {"paved_surface": 2}, "max_hole_m2": 2.0})()
    ids = {"paved_surface": 2, "utility_cover": 10}
    enclosed_ignore = np.full((3, 3), 2, np.uint8)
    enclosed_ignore[1, 1] = 255
    assert apply(enclosed_ignore, 1.0, config, ids)[1, 1] == 2
    cover = np.full((3, 3), 2, np.uint8)
    cover[1, 1] = 10
    assert apply(cover, 1.0, config, ids)[1, 1] == 10


def test_morphology_does_not_fill_border_touching_ignore() -> None:
    config = type("C", (), {"min_island_m2": {"paved_surface": 2.0}, "min_component_px": {"paved_surface": 2}, "max_hole_m2": 2.0})()
    mask = np.array([[255, 2], [2, 2]], np.uint8)
    assert apply(mask, 1.0, config, {"paved_surface": 2})[0, 0] == 255


def test_morphology_merges_many_surface_speckles_into_asphalt() -> None:
    rng = np.random.default_rng(4)
    mask = np.full((120, 120), 2, np.uint8)
    for _ in range(200):
        row, column = rng.integers(2, 115, size=2)
        size = int(rng.integers(1, 4))
        mask[row : row + size, column : column + size] = rng.choice([1, 3])
    config = type("C", (), {"min_island_m2": {}, "min_component_px": {"ground": 50, "low_vegetation": 50}, "max_hole_m2": 0.0})()
    refinement = SimpleNamespace(
        edge_snap=SimpleNamespace(enabled=False),
        color=SimpleNamespace(enabled=False),
        context=SimpleNamespace(enabled=False),
        shape=SimpleNamespace(enabled=False),
        ortho=SimpleNamespace(enabled=False),
        morphology=SimpleNamespace(
            enabled=True,
            min_island_m2=config.min_island_m2,
            min_component_px=config.min_component_px,
            max_hole_m2=config.max_hole_m2,
        ),
    )
    output = refine(
        RefinementContext(
            np.zeros((*mask.shape, 3), np.uint8),
            mask,
            "phone",
            None,
            SimpleNamespace(refinement=refinement),
            {"ground": 1, "paved_surface": 2, "low_vegetation": 3},
        )
    ).mask
    assert np.all(output == 2)


def test_morphology_fills_small_marking_holes_but_keeps_large_utility_cover() -> None:
    ids = {"paved_surface": 2, "road_marking": 9, "utility_cover": 10}
    config = type("C", (), {"min_island_m2": {}, "min_component_px": {"road_marking": 100, "paved_surface": 2_000, "utility_cover": 60}, "max_hole_m2": 0.0})()
    marking = np.full((30, 30), 9, np.uint8)
    marking[10:14, 10:15] = 255
    assert np.all(apply(marking, None, config, ids)[10:14, 10:15] == 9)
    asphalt = np.full((50, 50), 2, np.uint8)
    asphalt[5:30, 5:45] = 10
    assert np.all(apply(asphalt, None, config, ids)[5:30, 5:45] == 10)
