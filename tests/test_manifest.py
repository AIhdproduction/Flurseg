"""Manifest schema, repository, and transition coverage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from flurseg.manifest.migrations import migrate
from flurseg.manifest.repo import ImageRecord, ManifestRepository


def _record(index: int, *, status: str = "ingested", source: str = "drone", session: str = "a") -> ImageRecord:
    return ImageRecord(
        id=f"{source}_{session}_{index:04d}",
        hash=f"{index:064x}",
        source=source,
        session=session,
        gsd_method="unknown",
        original_path=f"input/{source}/{session}/{index}.jpg",
        image_path=f"work/{index}/image.jpg",
        work_dir=f"work/{index}",
        width=10,
        height=11,
        status=status,
    )


def test_migrate_is_idempotent_and_records_schema_version(tmp_path: Path) -> None:
    connection = sqlite3.connect(tmp_path / "manifest.db")
    assert migrate(connection) == 1
    assert migrate(connection) == 1
    assert connection.execute("SELECT version FROM schema_version").fetchone()[0] == 1


def test_insert_and_query_by_status_source_and_session(tmp_path: Path) -> None:
    repository = ManifestRepository(sqlite3.connect(tmp_path / "manifest.db"))
    first = _record(1, source="drone", session="a")
    second = _record(2, source="phone", session="b")
    repository.insert(first)
    repository.insert(second)
    assert [item["id"] for item in repository.query(status="ingested")] == [first.id, second.id]
    assert [item["id"] for item in repository.query(source="drone")] == [first.id]
    assert [item["id"] for item in repository.query(session="b")] == [second.id]


@pytest.mark.parametrize(
    ("start", "target"),
    [
        ("ingested", "prelabeled"),
        ("ingested", "rejected"),
        ("prelabeled", "auto_accepted"),
        ("prelabeled", "needs_review"),
        ("prelabeled", "rejected"),
        ("auto_accepted", "needs_review"),
        ("auto_accepted", "reviewed"),
        ("auto_accepted", "rejected"),
        ("needs_review", "reviewed"),
        ("needs_review", "rejected"),
    ],
)
def test_every_allowed_status_transition_passes(tmp_path: Path, start: str, target: str) -> None:
    repository = ManifestRepository(sqlite3.connect(tmp_path / "manifest.db"))
    record = _record(1, status=start)
    repository.insert(record)
    repository.transition(record.id, target)
    assert repository.get(record.id)["status"] == target


@pytest.mark.parametrize(("start", "target"), [("ingested", "reviewed"), ("reviewed", "rejected")])
def test_forbidden_status_transition_raises(tmp_path: Path, start: str, target: str) -> None:
    repository = ManifestRepository(sqlite3.connect(tmp_path / "manifest.db"))
    record = _record(1, status=start)
    repository.insert(record)
    with pytest.raises(ValueError, match="invalid status transition"):
        repository.transition(record.id, target)


def test_json_columns_round_trip(tmp_path: Path) -> None:
    repository = ManifestRepository(sqlite3.connect(tmp_path / "manifest.db"))
    record = _record(1)
    record.class_pixel_counts = {"tree": 4}
    record.review_changed_pixels = {"road": 2}
    repository.insert(record)
    restored = repository.get(record.id)
    assert restored["class_pixel_counts"] == {"tree": 4}
    assert restored["review_changed_pixels"] == {"road": 2}


def test_processing_error_and_review_rows_are_recorded(tmp_path: Path) -> None:
    repository = ManifestRepository(sqlite3.connect(tmp_path / "manifest.db"))
    record = _record(1)
    repository.insert(record)
    repository.record_error(record.original_path, "ingest", "bad image", image_id=record.id)
    repository.add_review(record.id, "2026-01-01T00:00:00Z", "2026-01-01T00:01:00Z", "accept", {"tree": 2}, 60)
    error = repository.connection.execute("SELECT image_id, stage, message FROM processing_errors").fetchone()
    review = repository.connection.execute("SELECT image_id, action, changed_pixels FROM reviews").fetchone()
    assert tuple(error) == (record.id, "ingest", "bad image")
    assert (review[0], review[1]) == (record.id, "accept")
    assert review[2] == '{"tree": 2}'
