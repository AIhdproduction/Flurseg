from __future__ import annotations

import numpy as np

from flurseg.refine import edge_snap
from flurseg.refine.edge_snap import apply


def test_edge_snap_returns_slic_labels() -> None:
    mask = np.array([[1, 1], [1, 2]], np.uint8)
    output, labels = apply(mask, np.zeros((2, 2, 3), np.uint8), type("C", (), {"slic_n_segments_per_mp": 1, "compactness": 1, "protected_classes": []})(), {})
    assert output.shape == mask.shape and labels.min() == 1


def test_edge_snap_majority_and_protected_pixels_share_a_superpixel() -> None:
    config = type("C", (), {"slic_n_segments_per_mp": 1, "compactness": 1, "protected_classes": ["road_marking"], "max_edge_fraction": 1.0})()
    mask = np.array([[1, 1], [2, 9]], np.uint8)
    labels = np.ones((2, 2), np.int32)
    output, _ = apply(mask, np.zeros((2, 2, 3), np.uint8), config, {"road_marking": 9}, labels)
    assert output.tolist() == [[1, 1], [1, 9]]


def test_edge_snap_reuses_stored_labels_without_slic(monkeypatch) -> None:
    config = type("C", (), {"slic_n_segments_per_mp": 1, "compactness": 1, "protected_classes": [], "max_edge_fraction": 1.0})()
    monkeypatch.setattr(edge_snap, "slic", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SLIC called")))
    output, used = apply(np.array([[1, 2]], np.uint8), np.zeros((1, 2, 3), np.uint8), config, {}, np.ones((1, 2), np.int32))
    assert output.tolist() == [[1, 1]] and used.shape == (1, 2)


def test_edge_snap_leaves_edge_heavy_superpixel_unchanged() -> None:
    config = type("C", (), {"slic_n_segments_per_mp": 1, "compactness": 1, "protected_classes": [], "max_edge_fraction": 0.0})()
    image = np.zeros((4, 4, 3), np.uint8)
    image[:, 2:] = 255
    mask = np.array([[1, 1, 2, 2]] * 4, np.uint8)
    output, _ = apply(mask, image, config, {}, np.ones((4, 4), np.int32))
    assert np.array_equal(output, mask)
