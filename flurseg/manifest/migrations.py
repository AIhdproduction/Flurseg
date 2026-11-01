"""Versioned, idempotent schema migrations."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable


def _initial_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS images (
            id TEXT PRIMARY KEY,
            hash TEXT UNIQUE NOT NULL,
            parent_hash TEXT,
            tile_index INTEGER,
            source TEXT NOT NULL,
            session TEXT NOT NULL,
            camera TEXT,
            gsd_m REAL,
            gsd_method TEXT NOT NULL,
            original_path TEXT NOT NULL,
            image_path TEXT NOT NULL,
            work_dir TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('ingested','prelabeled','auto_accepted','needs_review','reviewed','rejected')),
            routing_reason TEXT,
            uncertain_fraction REAL,
            ignore_fraction REAL,
            class_pixel_counts TEXT,
            processing_seconds REAL,
            review_seconds REAL,
            review_changed_pixels TEXT,
            is_test_session INTEGER NOT NULL DEFAULT 0,
            spot_check INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
        CREATE INDEX IF NOT EXISTS idx_images_source_session ON images(source, session);
        CREATE TABLE IF NOT EXISTS processing_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT,
            original_path TEXT NOT NULL,
            stage TEXT NOT NULL,
            message TEXT NOT NULL,
            traceback TEXT,
            time TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            started TEXT NOT NULL,
            finished TEXT,
            action TEXT CHECK(action IN ('accept','reject','skip')),
            changed_pixels TEXT,
            seconds REAL,
            FOREIGN KEY(image_id) REFERENCES images(id)
        );
        """
    )


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_initial_schema]


def migrate(connection: sqlite3.Connection) -> int:
    """Apply each ordered migration exactly once and return the schema version."""
    connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    version = int(row[0]) if row else 0
    if row is None:
        connection.execute("INSERT INTO schema_version(version) VALUES (0)")
    for index, migration in enumerate(MIGRATIONS, start=1):
        if index > version:
            migration(connection)
            connection.execute("UPDATE schema_version SET version = ?", (index,))
            version = index
    connection.commit()
    return version
