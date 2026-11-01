"""Shared test fixtures for a disposable repository root."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture
def tmp_path(tmp_path_factory) -> Path:
    """Provide a standard temp path or a pre-created sandbox-safe directory."""
    root = os.environ.get("FLURSEG_TEST_TMP")
    if root is None:
        return tmp_path_factory.mktemp("flurseg")
    case_path = Path(root) / f"case-{uuid4().hex}"
    case_path.mkdir(parents=True, exist_ok=False)
    return case_path


@pytest.fixture
def tmp_root(tmp_path: Path) -> Path:
    """Create a temporary root containing copies of the checked-in YAML files."""
    source_config = Path(__file__).resolve().parents[1] / "config"
    target_config = tmp_path / "config"
    target_config.mkdir()
    shutil.copy2(source_config / "classes.yaml", target_config / "classes.yaml")
    shutil.copy2(source_config / "pipeline.yaml", target_config / "pipeline.yaml")
    return tmp_path
