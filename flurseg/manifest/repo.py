"""Typed access to the manifest and review tables."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from flurseg.manifest.migrations import migrate

_STATUSES = {"ingested", "prelabeled", "auto_accepted", "needs_review", "reviewed", "rejected"}
_TRANSITIONS = {
    "ingested": {"prelabeled", "rejected"},
    "prelabeled": {"auto_accepted", "needs_review", "rejected"},
    "auto_accepted": {"needs_review", "reviewed", "rejected"},
    "needs_review": {"reviewed", "rejected"},
    "reviewed": set(),
    "rejected": set(),
}
_JSON_COLUMNS = {"class_pixel_counts", "review_changed_pixels", "changed_pixels"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ImageRecord:
    """One manifest image row with repository-relative paths as strings."""

    id: str
    hash: str
    source: str
    session: str
    gsd_method: str
    original_path: str
    image_path: str
    work_dir: str
    width: int
    height: int
    parent_hash: str | None = None
    tile_index: int | None = None
    camera: str | None = None
    gsd_m: float | None = None
    status: str = "ingested"
    routing_reason: str | None = None
    uncertain_fraction: float | None = None
    ignore_fraction: float | None = None
    class_pixel_counts: dict[str, int] | None = None
    processing_seconds: float | None = None
    review_seconds: float | None = None
    review_changed_pixels: dict[str, int] | None = None
    is_test_session: bool = False
    spot_check: bool = False
    error: str | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class ManifestRepository:
    """Repository that validates status changes and serializes JSON columns."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        connection.row_factory = sqlite3.Row
        self.connection = connection
        migrate(connection)

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        for name in _JSON_COLUMNS & item.keys():
            if item[name] is not None:
                item[name] = json.loads(item[name])
        item["is_test_session"] = bool(item["is_test_session"])
        item["spot_check"] = bool(item["spot_check"])
        return item

    def insert(self, record: ImageRecord | None = None, **values: Any) -> ImageRecord:
        """Insert one ingested image and return its normalized record."""
        if record is None:
            record = ImageRecord(**values)
        if record.status not in _STATUSES:
            raise ValueError(f"unknown image status: {record.status}")
        data = asdict(record)
        for name in _JSON_COLUMNS & data.keys():
            if data[name] is not None:
                data[name] = json.dumps(data[name], sort_keys=True)
        columns = list(data)
        self.connection.execute(
            f"INSERT INTO images ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
            [data[name] for name in columns],
        )
        self.connection.commit()
        return record

    insert_image = insert

    def get(self, image_id: str) -> dict[str, Any] | None:
        """Fetch an image by stable image identifier."""
        return self._decode(self.connection.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone())

    def find_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        """Fetch an image by original or tile content hash."""
        return self._decode(self.connection.execute("SELECT * FROM images WHERE hash = ?", (content_hash,)).fetchone())

    def transition(self, image_id: str, status: str, **updates: Any) -> None:
        """Transition an image only through an allowed state edge."""
        current = self.get(image_id)
        if current is None:
            raise KeyError(f"unknown image ID: {image_id}")
        if status not in _TRANSITIONS[current["status"]]:
            raise ValueError(f"invalid status transition: {current['status']} -> {status}")
        updates["status"] = status
        updates["updated_at"] = _now()
        for name in _JSON_COLUMNS & updates.keys():
            if updates[name] is not None:
                updates[name] = json.dumps(updates[name], sort_keys=True)
        assignments = ", ".join(f"{name} = ?" for name in updates)
        self.connection.execute(f"UPDATE images SET {assignments} WHERE id = ?", [*updates.values(), image_id])
        self.connection.commit()

    def reroute(self, image_id: str, status: str, **updates: Any) -> None:
        """Update an unreviewed image route during score-only reprocessing."""
        current = self.get(image_id)
        if current is None:
            raise KeyError(f"unknown image ID: {image_id}")
        if current["status"] in {"reviewed", "rejected"}:
            return
        if status not in {"auto_accepted", "needs_review"}:
            raise ValueError(f"invalid routed status: {status}")
        updates.update(status=status, updated_at=_now())
        for name in _JSON_COLUMNS & updates.keys():
            if updates[name] is not None:
                updates[name] = json.dumps(updates[name], sort_keys=True)
        assignments = ", ".join(f"{name} = ?" for name in updates)
        self.connection.execute(f"UPDATE images SET {assignments} WHERE id = ?", [*updates.values(), image_id])
        self.connection.commit()

    def query(
        self, *, status: str | None = None, source: str | None = None, session: str | None = None
    ) -> list[dict[str, Any]]:
        """List images filtered by status, source and/or session."""
        clauses: list[str] = []
        values: list[str] = []
        for column, value in (("status", status), ("source", source), ("session", session)):
            if value is not None:
                clauses.append(f"{column} = ?")
                values.append(value)
        statement = "SELECT * FROM images" + (" WHERE " + " AND ".join(clauses) if clauses else "")
        return [self._decode(row) for row in self.connection.execute(statement, values).fetchall()]

    def count(self, **filters: str) -> int:
        """Count image rows using the same supported filters as :meth:`query`."""
        return len(self.query(**filters))

    def next_sequence(self, source: str, session: str) -> int:
        """Return the next never-reused per-source, per-session sequence number."""
        rows = self.connection.execute(
            "SELECT id FROM images WHERE source = ? AND session = ?", (source, session)
        ).fetchall()
        numbers = [int(str(row[0]).rsplit("_", 1)[-1]) for row in rows if str(row[0]).rsplit("_", 1)[-1].isdigit()]
        return max(numbers, default=0) + 1

    def record_error(self, original_path: str, stage: str, message: str, traceback: str | None = None, image_id: str | None = None) -> None:
        """Record a processing failure without interrupting other candidates."""
        self.connection.execute(
            "INSERT INTO processing_errors(image_id, original_path, stage, message, traceback, time) VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, original_path, stage, message, traceback, _now()),
        )
        self.connection.commit()

    def add_review(
        self, image_id: str, started: str, finished: str | None, action: str | None,
        changed_pixels: dict[str, int] | None, seconds: float | None,
    ) -> None:
        """Store a review event."""
        if action not in {"accept", "reject", "skip", None}:
            raise ValueError(f"unknown review action: {action}")
        self.connection.execute(
            "INSERT INTO reviews(image_id, started, finished, action, changed_pixels, seconds) VALUES (?, ?, ?, ?, ?, ?)",
            (image_id, started, finished, action, json.dumps(changed_pixels) if changed_pixels else None, seconds),
        )
        self.connection.commit()
