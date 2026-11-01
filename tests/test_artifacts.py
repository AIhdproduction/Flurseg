"""Independent round-trip coverage for every work artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from flurseg.artifacts import ArtifactStore, upsample_scores
from flurseg.config import load_config
from flurseg.paths import RepoPaths


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "work" / "item")


def _read_png(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


def test_image_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    image = np.full((6, 8, 3), (20, 100, 220), dtype=np.uint8)
    store.write_image(image)
    restored = np.asarray(Image.open(store.directory / "image.jpg").convert("RGB"))
    assert restored.shape == image.shape
    assert np.max(np.abs(restored.astype(int) - image.astype(int))) <= 3


def test_scores_artifact_round_trip_at_half_scale(tmp_path: Path) -> None:
    store = _store(tmp_path)
    scores = {"tree": np.full((3, 4), 0.5, dtype=np.float32)}
    store.write_scores(scores, {"tree": 0.5})
    restored, scales = store.read_scores()
    assert scales == {"tree": 0.5}
    assert np.allclose(restored["tree"], 128 / 255)


def test_scores_artifact_upsamples_back_to_full_size(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_scores({"tree": np.full((3, 4), 0.5, dtype=np.float32)}, {"tree": 0.5})
    restored, _ = store.read_scores()
    assert upsample_scores(restored, (6, 8))["tree"].shape == (6, 8)


def test_scores_artifact_keeps_class_local_diagnostics(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_scores(
        {"ground": np.ones((2, 3), np.float32)},
        {"ground": 0.5},
        {"ground": np.full((2, 3), 2, np.uint8)},
        {"ground": np.full((2, 3), 17, np.uint8)},
    )
    winners, std = store.read_score_diagnostics()
    assert winners["ground"].dtype == np.uint8 and winners["ground"][0, 0] == 2
    assert std["ground"][0, 0] == 17


def test_prompt_winner_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    array = np.array([[0, 255]], dtype=np.uint8)
    store.write_prompt_winner(array)
    assert np.array_equal(_read_png(store.directory / "prompt_winner.png"), array)


def test_tta_disagreement_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    array = np.array([[1, 2]], dtype=np.uint8)
    store.write_tta_disagreement(array)
    assert np.array_equal(_read_png(store.directory / "tta_disagreement.png"), array)


def test_superpixels_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    labels = np.array([[1, 2]], dtype=np.int32)
    store.write_superpixels(labels)
    assert np.array_equal(store.read_superpixels(), labels)


def test_superpixels_coarse_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.read_superpixels_coarse() is None
    labels = np.array([[1, 2], [3, 4]], dtype=np.int32)
    store.write_superpixels_coarse(labels)
    restored = store.read_superpixels_coarse()
    assert restored is not None
    assert restored.dtype == np.int32
    assert np.array_equal(restored, labels)


def test_raw_mask_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mask = np.array([[1, 2]], dtype=np.uint8)
    store.write_raw_mask(mask)
    assert np.array_equal(_read_png(store.directory / "raw_mask.png"), mask)


def test_proposed_mask_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mask = np.array([[3, 4]], dtype=np.uint8)
    store.write_proposed_mask(mask)
    assert np.array_equal(_read_png(store.directory / "proposed_mask.png"), mask)


def test_uncertainty_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    array = np.array([[5, 6]], dtype=np.uint8)
    store.write_uncertainty(array)
    assert np.array_equal(_read_png(store.directory / "uncertainty.png"), array)


def test_auto_change_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    array = np.array([[1, 6]], dtype=np.uint8)
    store.write_auto_change(array)
    assert np.array_equal(_read_png(store.directory / "auto_change.png"), array)


def test_overlay_artifact_uses_class_colors_and_renders_ignore_distinctly(tmp_root: Path) -> None:
    store = _store(tmp_root)
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 4
    mask[5:7, 5:7] = 255
    config = load_config(RepoPaths.from_root(tmp_root))
    colors = {item.id: item.display_color for item in config.classes.classes}
    store.write_overlay(image, mask, colors)
    overlay = np.asarray(Image.open(store.directory / "overlay.jpg").convert("RGB"))
    expected_tree = np.array([27, 94, 32]) * 0.55 + 200 * 0.45
    assert np.max(np.abs(overlay[1, 1].astype(float) - expected_tree)) <= 8
    assert overlay[5:7, 5:7].mean() < overlay[1:3, 1:3].mean()
    assert len(np.unique(overlay[5:7, 5:7].reshape(-1, 3), axis=0)) > 1


def test_meta_artifact_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.write_meta({"source": "phone", "blurred_fraction": 0.25})
    assert json.loads((store.directory / "meta.json").read_text(encoding="utf-8")) == {
        "blurred_fraction": 0.25,
        "source": "phone",
    }
