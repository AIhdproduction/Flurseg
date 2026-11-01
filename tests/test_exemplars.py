from __future__ import annotations

import json

import cv2
import numpy as np

from flurseg.segment.exemplars import find_similar, mosaic_canvas, score_mosaic


class RecordingBackend:
    def __init__(self) -> None:
        self.images: list[np.ndarray] = []
        self.boxes: list[list[tuple[int, int, int, int]]] = []

    def score_boxes(self, image, boxes, prompt):
        self.images.append(image.copy())
        self.boxes.append(list(boxes))
        return np.arange(image.shape[0] * image.shape[1], dtype=np.float32).reshape(image.shape[:2])


def _exemplar(directory, box=(1, 1, 5, 4)) -> None:
    directory.mkdir()
    cv2.imwrite(str(directory / "crop.png"), np.full((6, 8, 3), 80, np.uint8))
    (directory / "crop.json").write_text(json.dumps({"boxes": [box]}), encoding="utf-8")


def test_mosaic_canvas_uses_fixed_canvas_and_requested_strip_height(tmp_path) -> None:
    directory = tmp_path / "examples"
    _exemplar(directory)
    canvas, _, content = mosaic_canvas(np.full((40, 40, 3), 50, np.uint8), directory, tile_size=32, strip_fraction=0.25, max_per_class=2)
    assert canvas.shape == (32, 32, 3) and content == (24, 32)


def test_mosaic_boxes_are_translated_into_the_strip(tmp_path) -> None:
    directory = tmp_path / "examples"
    _exemplar(directory)
    _, boxes, content = mosaic_canvas(np.full((30, 30, 3), 50, np.uint8), directory, tile_size=32, strip_fraction=0.25, max_per_class=1)
    assert boxes and boxes[0][1] >= content[0] and boxes[0][3] <= 32


def test_score_mosaic_discards_the_exemplar_strip(tmp_path) -> None:
    directory = tmp_path / "examples"
    _exemplar(directory)
    backend = RecordingBackend()
    output = score_mosaic(np.full((20, 30, 3), 50, np.uint8), backend, "tree", directory, tile_size=32, strip_fraction=0.25, max_per_class=1)
    assert output.shape == (20, 30) and backend.images[0].shape == (32, 32, 3)


def test_find_similar_clamps_window_at_left_top_border() -> None:
    backend = RecordingBackend()
    output = find_similar(np.zeros((30, 40, 3), np.uint8), [(0, 0, 4, 4)], "tree", backend, tile_size=16)
    assert backend.images[0].shape[:2] == (16, 16) and np.any(output[:16, :16]) and not output[16:, :].any()


def test_find_similar_clamps_window_at_right_bottom_border() -> None:
    backend = RecordingBackend()
    output = find_similar(np.zeros((30, 40, 3), np.uint8), [(36, 26, 40, 30)], "tree", backend, tile_size=16)
    assert backend.images[0].shape[:2] == (16, 16) and np.any(output[-16:, -16:]) and not output[:-16, :].any()


def test_find_similar_pastes_window_scores_at_its_full_image_offset() -> None:
    backend = RecordingBackend()
    output = find_similar(np.zeros((40, 50, 3), np.uint8), [(24, 19, 26, 21)], "tree", backend, tile_size=10)
    assert output[15, 20] == 0 and output[16, 21] == 11 and not output[:15, :].any()
