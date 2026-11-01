"""Validation of optional per-session metadata."""

from __future__ import annotations

from datetime import date as Date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class SessionMetadata(BaseModel):
    """The optional ``session.yaml`` contract shared by all sources."""

    model_config = ConfigDict(extra="forbid")

    camera: str | None = None
    date: Date | None = None
    area_type: Literal["construction_site", "street", "residential", "urban", "agriculture", "forest_water"] | None = None
    gsd_m: float | None = Field(default=None, gt=0)
    test: bool = False
    notes: str = ""


def load_session(path: Path) -> SessionMetadata:
    """Load a session file, or defaults when it does not exist."""
    if not path.exists():
        return SessionMetadata()
    try:
        with path.open("r", encoding="utf-8") as stream:
            content = yaml.safe_load(stream) or {}
        return SessionMetadata.model_validate(content)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"invalid session metadata in {path}: {error}") from error
