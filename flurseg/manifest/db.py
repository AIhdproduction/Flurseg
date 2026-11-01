"""SQLite connection lifecycle with the required safety pragmas."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    """Context-managed SQLite database configured for local pipeline use."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.connection: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        """Open the database once and enable WAL and foreign keys."""
        if self.connection is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
        return self.connection

    def __enter__(self) -> sqlite3.Connection:
        return self.connect()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.connection is not None:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
            self.connection.close()
            self.connection = None
