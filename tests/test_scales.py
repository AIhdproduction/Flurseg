"""Scale resize and configured class assignment coverage."""

from __future__ import annotations

import numpy as np

from flurseg.config import load_config
from flurseg.paths import RepoPaths
from flurseg.segment.scales import class_scale_map, resize_image, resize_scores


def test_down_and_up_resize_round_trip_shapes_for_odd_sizes() -> None:
    image = np.zeros((11, 13, 3), dtype=np.uint8)
    down = resize_image(image, 0.5)
    up = resize_image(down, 2.0)
    scores = np.ones((2, down.shape[0], down.shape[1]), dtype=np.float32)
    restored_scores = resize_scores(scores, image.shape[:2])
    assert down.shape == (6, 6, 3)
    assert up.shape == (12, 12, 3)
    assert restored_scores.shape == (2, 11, 13)


def test_configured_class_scale_mapping_covers_each_trainable_class_once(tmp_root) -> None:
    config = load_config(RepoPaths.from_root(tmp_root))
    mapping = class_scale_map(config.pipeline.scales)
    trainable = {item.name for item in config.classes.classes if item.trainable}
    assert set(mapping) == trainable
    assert len(mapping) == len(trainable)
