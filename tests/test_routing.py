from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flurseg.paths import RepoPaths
from flurseg.routing import RoutingDecision, decide, materialize


def _config(**values: object) -> SimpleNamespace:
    return SimpleNamespace(
        threshold=values.get("threshold", 0.5),
        auto_accept_max_uncertain_fraction=values.get("uncertain", 0.2),
        auto_accept_max_ignore_fraction=values.get("ignore", 0.3),
        spot_check_fraction=values.get("spot", 0.0),
        seed=values.get("seed", 42),
    )


def test_uncertainty_below_threshold_is_auto_accepted() -> None:
    assert decide(np.ones((4, 4), np.uint8), np.full((4, 4), 0.4), "a", False, _config()).status == "auto_accepted"


def test_uncertainty_above_guard_needs_review() -> None:
    decision = decide(np.ones((4, 4), np.uint8), np.full((4, 4), 0.9), "a", False, _config())
    assert decision.status == "needs_review" and decision.reason == "uncertain"


def test_ignore_fraction_above_guard_needs_review() -> None:
    decision = decide(np.array([[255, 255, 1, 1]], np.uint8), np.zeros((1, 4)), "a", False, _config(ignore=0.4))
    assert decision.status == "needs_review" and decision.reason == "uncertain"


def test_test_session_always_needs_review() -> None:
    assert decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), "a", True, _config()).reason == "test_session"


def test_spot_check_is_stable_for_seed_and_changes_with_another_seed() -> None:
    config = _config(spot=0.5, seed=17)
    image_hash = next(
        f"stable-hash-{index}"
        for index in range(100)
        if decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), f"stable-hash-{index}", False, config).spot_check
        != decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), f"stable-hash-{index}", False, _config(spot=0.5, seed=18)).spot_check
    )
    first = decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), image_hash, False, config)
    other = decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), image_hash, False, _config(spot=0.5, seed=18))
    assert first == decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), image_hash, False, config)
    assert first.spot_check != other.spot_check


def test_spot_check_rate_is_within_two_percent_of_ten_percent() -> None:
    config = _config(spot=0.10)
    fraction = np.mean([decide(np.ones((1, 1), np.uint8), np.zeros((1, 1)), f"hash-{index}", False, config).spot_check for index in range(2000)])
    assert 0.08 <= fraction <= 0.12


def test_materialize_switches_between_pool_and_review_and_removes_stale_copy(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    image = np.full((3, 4, 3), 90, np.uint8)
    mask, uncertainty = np.ones((3, 4), np.uint8), np.zeros((3, 4), np.uint8)
    accepted = RoutingDecision("auto_accepted", None, 0.0, 0.0, {1}, False)
    review = RoutingDecision("needs_review", "uncertain", 0.5, 0.0, {1}, False)
    materialize(paths, "item", image, mask, uncertainty, accepted)
    materialize(paths, "item", image, mask, uncertainty, review)
    assert (paths.review_dir / "item" / "image.jpg").is_file() and not (paths.pool_images_dir / "item.jpg").exists()
    materialize(paths, "item", image, mask, uncertainty, accepted)
    assert (paths.pool_masks_dir / "item.png").is_file() and not (paths.review_dir / "item" / "proposed_mask.png").exists()


def test_materialize_rejects_a_target_outside_the_root(tmp_root) -> None:
    paths = SimpleNamespace(root=tmp_root, pool_images_dir=tmp_root.parent / "outside", pool_masks_dir=tmp_root.parent / "outside", review_dir=tmp_root / "review")
    decision = RoutingDecision("auto_accepted", None, 0.0, 0.0, {1}, False)
    with pytest.raises(ValueError, match="inside"):
        materialize(paths, "item", np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8), np.zeros((1, 1), np.uint8), decision)
