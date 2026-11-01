from __future__ import annotations

import numpy as np
import pytest

from flurseg.config import load_config
from flurseg.fusion import fuse
from flurseg.paths import RepoPaths


def test_fusion_priority_override_and_no_prompt(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    scores = {"vehicle": np.array([[0.9, 0.1]], np.float32), "tree": np.array([[0.85, 0.1]], np.float32)}
    result = fuse(scores, (1, 2), cfg)
    assert result.mask.tolist() == [[4, 255]]


def test_fusion_priority_outside_margin_uses_most_confident_class(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"tree": np.array([[0.6]], np.float32), "low_vegetation": np.array([[0.96]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 3
    assert result.conflict[0, 0] == pytest.approx(0)


def test_fusion_priority_within_margin_sets_conflict(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"vehicle": np.array([[0.85]], np.float32), "ground": np.array([[0.95]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 6
    assert result.conflict[0, 0] == pytest.approx(0.10)


def test_fusion_priority_outside_margin_uses_ground(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"vehicle": np.array([[0.70]], np.float32), "ground": np.array([[0.95]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 1


def test_fusion_margin_extremes_match_strict_priority_and_pure_score(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    scores = {"vehicle": np.array([[0.6]], np.float32), "ground": np.array([[0.9]], np.float32)}
    cfg.pipeline.fusion.priority_margin = 1.0
    assert fuse(scores, (1, 1), cfg).mask[0, 0] == 6
    cfg.pipeline.fusion.priority_margin = 0.0
    assert fuse(scores, (1, 1), cfg).mask[0, 0] == 1


def test_fusion_presence_threshold_excludes_priority_class(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"vehicle": np.array([[0.3]], np.float32), "ground": np.array([[0.9]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 1


def test_fusion_uses_argmax_when_no_class_is_present(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"vehicle": np.array([[0.4]], np.float32), "ground": np.array([[0.45]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 1


def test_fusion_returns_ignore_when_all_scores_below_no_prompt(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"vehicle": np.array([[0.1]], np.float32), "ground": np.array([[0.1]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 255


def test_fusion_override_requires_its_minimum_score(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    outside_margin = fuse({"vehicle": np.array([[0.9]], np.float32), "tree": np.array([[0.6]], np.float32)}, (1, 1), cfg)
    within_margin = fuse({"vehicle": np.array([[0.9]], np.float32), "tree": np.array([[0.85]], np.float32)}, (1, 1), cfg)
    assert outside_margin.mask[0, 0] == 6
    assert within_margin.mask[0, 0] == 4


def test_fusion_supports_per_class_presence_threshold(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    cfg.pipeline.fusion.presence_threshold_per_class = {"vehicle": 0.7}
    result = fuse({"vehicle": np.array([[0.6]], np.float32), "ground": np.array([[0.9]], np.float32)}, (1, 1), cfg)
    assert result.mask[0, 0] == 1


def test_fusion_top_score_and_margin_belong_to_assigned_class(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = fuse({"vehicle": np.array([[0.85]], np.float32), "ground": np.array([[0.95]], np.float32)}, (1, 1), cfg)
    assert result.top_score[0, 0] == pytest.approx(0.85)
    assert result.margin[0, 0] == 0
