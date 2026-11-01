from __future__ import annotations

import numpy as np

from flurseg.refine.context_rules import apply


def test_context_marks_isolated_road_marking_ignore() -> None:
    assert apply(np.array([[9]], np.uint8), object(), {"road_marking": 9, "paved_surface": 2})[0, 0] == 255


def test_context_keeps_marking_on_paved_and_curb_at_paved_border() -> None:
    config = type("C", (), {"curb_max_distance_px": 1, "vehicle_ring_width_px": 1, "vehicle_on_building_ratio": 0.6})()
    ids = {"road_marking": 9, "paved_surface": 2, "curb": 8}
    mask = np.array([[8, 2, 2], [255, 9, 2], [255, 255, 255]], np.uint8)
    output = apply(mask, config, ids)
    assert output[1, 1] == 9 and output[0, 0] == 8


def test_context_keeps_a_complete_marking_stripe_inside_asphalt() -> None:
    ids = {"road_marking": 9, "paved_surface": 2}
    mask = np.full((9, 210), 2, np.uint8)
    mask[3:6, 5:205] = 9
    output = apply(mask, object(), ids)
    assert np.array_equal(output[3:6, 5:205], mask[3:6, 5:205])


def test_context_invalidates_a_complete_marking_component_inside_a_building() -> None:
    ids = {"road_marking": 9, "paved_surface": 2, "building": 5}
    mask = np.full((20, 20), 5, np.uint8)
    mask[5:15, 5:15] = 9
    assert np.all(apply(mask, object(), ids)[5:15, 5:15] == 255)


def test_context_keeps_nearby_curb_and_invalidates_isolated_curb() -> None:
    ids = {"curb": 8, "paved_surface": 2}
    config = type("C", (), {"curb_max_distance_px": 5})()
    mask = np.full((80, 100), 255, np.uint8)
    mask[:, :40] = 2
    mask[10:70, 40:48] = 8
    mask[10:20, 90:95] = 8
    output = apply(mask, config, ids)
    assert np.all(output[10:70, 40:48] == 8)
    assert np.all(output[10:20, 90:95] == 255)


def test_context_invalidates_vehicle_surrounded_by_building_but_not_paved() -> None:
    config = type("C", (), {"curb_max_distance_px": 1, "vehicle_ring_width_px": 1, "vehicle_on_building_ratio": 0.6})()
    ids = {"vehicle": 6, "building": 5, "paved_surface": 2}
    building = np.full((5, 5), 5, np.uint8)
    building[2, 2] = 6
    assert apply(building, config, ids)[2, 2] == 255
    paved = np.full((5, 5), 2, np.uint8)
    paved[2, 2] = 6
    assert apply(paved, config, ids)[2, 2] == 6


def test_context_keeps_utility_cover_inside_asphalt() -> None:
    config = type("C", (), {"utility_cover_ring_width_px": 1, "utility_cover_ground_ratio": 0.5})()
    ids = {"utility_cover": 10, "paved_surface": 2}
    mask = np.full((5, 5), 2, np.uint8)
    mask[2, 2] = 10
    assert apply(mask, config, ids)[2, 2] == 10


def test_context_invalidates_utility_cover_without_ground_plane_context() -> None:
    config = type("C", (), {"utility_cover_ring_width_px": 1, "utility_cover_ground_ratio": 0.5})()
    ids = {"utility_cover": 10, "building": 5, "other": 7, "paved_surface": 2}
    mask = np.full((5, 5), 5, np.uint8)
    mask[0, :] = 255
    mask[2, 2] = 10
    assert apply(mask, config, ids)[2, 2] == 255


def test_context_keeps_utility_cover_with_sixty_percent_ground_ring() -> None:
    config = type("C", (), {"utility_cover_ring_width_px": 1, "utility_cover_ground_ratio": 0.5})()
    ids = {"utility_cover": 10, "paved_surface": 2, "building": 5}
    mask = np.full((5, 5), 5, np.uint8)
    mask[2, 2] = 10
    for row, column in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 3)):
        mask[row, column] = 2
    assert apply(mask, config, ids)[2, 2] == 10
