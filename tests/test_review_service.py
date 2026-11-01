from __future__ import annotations

import numpy as np
import pytest

from flurseg.artifacts import ArtifactStore
from flurseg.config import load_config
from flurseg.io.atomic import atomic_save_png
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths
from flurseg.review.service import ReviewError, ReviewService


def _record(paths: RepoPaths, image_id: str, *, status: str = "needs_review", uncertainty: float = 0.2) -> None:
    directory = paths.work_dir / image_id
    artifact = ArtifactStore(directory)
    image = np.full((300, 300, 3), 110, dtype=np.uint8)
    proposed = np.zeros((300, 300), dtype=np.uint8)
    proposed[:10, :10] = 255
    artifact.write_image(image)
    artifact.write_proposed_mask(proposed)
    artifact.write_raw_mask(np.ones((300, 300), dtype=np.uint8))
    artifact.write_uncertainty(np.full((300, 300), 120, dtype=np.uint8))
    artifact.write_auto_change(np.ones((300, 300), dtype=np.uint8))
    artifact.write_superpixels(np.zeros((300, 300), dtype=np.int32))
    with Database(paths.database) as connection:
        ManifestRepository(connection).insert(
            ImageRecord(
                id=image_id, hash=f"hash-{image_id}", source="drone", session="s1", gsd_method="session",
                original_path="input/drone/s1/a.jpg", image_path=f"work/{image_id}/image.jpg",
                work_dir=f"work/{image_id}", width=300, height=300, status=status,
                uncertain_fraction=uncertainty, routing_reason="uncertain", class_pixel_counts={"0": 89900},
            )
        )


@pytest.fixture
def review(tmp_root) -> tuple[ReviewService, RepoPaths]:
    paths = RepoPaths.from_root(tmp_root)
    _record(paths, "first", uncertainty=0.1)
    _record(paths, "second", uncertainty=0.8)
    _record(paths, "pool", status="auto_accepted")
    return ReviewService(paths, load_config(paths)), paths


def test_queue_sorting_filters_and_pool_to_review(review) -> None:
    service, _ = review
    assert [row["id"] for row in service.queue()] == ["second", "first"]
    assert [row["id"] for row in service.queue(source="drone", reason="uncertain")] == ["second", "first"]
    assert service.to_review("pool")["status"] == "needs_review"
    assert {row["id"] for row in service.queue()} == {"first", "second", "pool"}


def test_draft_validation_accept_reject_and_skip(review) -> None:
    service, paths = review
    with pytest.raises(ReviewError, match="length"):
        service.save_draft("first", b"short")
    with pytest.raises(ReviewError, match="values"):
        service.save_draft("first", bytes([99]) * (300 * 300))
    draft = np.zeros((300, 300), dtype=np.uint8)
    draft[:10, :10] = 255
    draft[0, :4] = 3
    service.save_draft("first", draft.tobytes())
    accepted = service.accept("first", 12.5)
    assert accepted["changed_pixels"] == {"3": 4}
    assert (paths.pool_masks_dir / "first.png").is_file()
    assert service.reject("second")["status"] == "rejected"
    service.to_review("pool")
    assert service.skip("pool")["status"] == "needs_review"


def test_failed_atomic_draft_save_keeps_existing_mask(review, monkeypatch) -> None:
    service, paths = review
    initial = np.full((300, 300), 1, dtype=np.uint8)
    target = paths.review_dir / "first" / "draft_mask.png"
    atomic_save_png(target, initial)

    def fail(*_args, **_kwargs) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("flurseg.review.service.atomic_save_png", fail)
    with pytest.raises(OSError):
        service.save_draft("first", np.zeros(300 * 300, dtype=np.uint8).tobytes())
    assert np.array_equal(service._working_mask(service._record("first")), initial)


def test_queue_filters_each_field(review) -> None:
    """Each supported queue filter keeps only matching rows."""
    service, _ = review
    assert [row["id"] for row in service.queue(source="drone")] == ["second", "first"]
    assert [row["id"] for row in service.queue(session="s1")] == ["second", "first"]
    assert [row["id"] for row in service.queue(reason="uncertain")] == ["second", "first"]
    assert service.queue(reason="not-a-reason") == []


def test_accept_stores_review_seconds_and_pool_files(review) -> None:
    """Accept stores a non-negative duration and creates train-pool assets."""
    service, paths = review
    result = service.accept("first", 7.25)
    assert result["status"] == "reviewed"
    assert (paths.pool_images_dir / "first.jpg").is_file()
    assert service._record("first")["review_seconds"] == 7.25


def test_pool_image_can_be_returned_then_accepted(review) -> None:
    """An auto-accepted image can be recalled and accepted after review."""
    service, paths = review
    assert service.to_review("pool")["status"] == "needs_review"
    assert service.accept("pool")["status"] == "reviewed"
    assert (paths.pool_masks_dir / "pool.png").is_file()


def test_stats_error_rate_uses_spot_checks(review) -> None:
    """Stats expose zero error rate when no reviewed spot checks exist."""
    service, _ = review
    assert service.stats()["auto_accept_error_rate"]["total"] == 0.0


def test_reject_requires_a_queued_image(review) -> None:
    """Reject refuses images outside the review queue."""
    service, _ = review
    with pytest.raises(ReviewError, match="cannot be rejected"):
        service.reject("pool")
