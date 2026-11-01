"""Session metadata validation cases."""

from __future__ import annotations

from pathlib import Path

import pytest

from flurseg.ingest.session import load_session


def test_missing_session_file_returns_defaults(tmp_path: Path) -> None:
    session = load_session(tmp_path / "session.yaml")
    assert session.camera is None
    assert session.gsd_m is None
    assert session.test is False
    assert session.notes == ""


def test_valid_session_file_is_loaded(tmp_path: Path) -> None:
    path = tmp_path / "session.yaml"
    path.write_text("camera: Test Camera\narea_type: urban\ngsd_m: 0.02\ntest: true\n", encoding="utf-8")
    session = load_session(path)
    assert (session.camera, session.area_type, session.gsd_m, session.test) == ("Test Camera", "urban", 0.02, True)


def test_invalid_area_type_reports_an_error(tmp_path: Path) -> None:
    path = tmp_path / "session.yaml"
    path.write_text("area_type: moon_base\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid session metadata"):
        load_session(path)


def test_malformed_yaml_reports_its_file_name(tmp_path: Path) -> None:
    path = tmp_path / "session.yaml"
    path.write_text("camera: [unterminated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="session.yaml"):
        load_session(path)
