"""Discovery rules for each supported input source."""

from __future__ import annotations

from pathlib import Path

from flurseg.ingest.discover import discover
from flurseg.paths import RepoPaths


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


def test_discover_applies_extensions_per_source(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "drone" / "a", "drone.jpg", "drone.tif", "phone.heic")
    _touch(tmp_path / "input" / "phone" / "a", "phone.jpg", "phone.heic", "ortho.tif")
    _touch(tmp_path / "input" / "ortho" / "a", "ortho.tif", "drone.jpg", "phone.heic")
    candidates = discover(RepoPaths.from_root(tmp_path))
    assert [(item.source, item.path.name) for item in candidates] == [
        ("drone", "drone.jpg"),
        ("drone", "drone.tif"),
        ("phone", "phone.heic"),
        ("phone", "phone.jpg"),
        ("ortho", "ortho.tif"),
    ]


def test_discover_ignores_session_yaml(tmp_path: Path) -> None:
    session = tmp_path / "input" / "phone" / "a"
    _touch(session, "session.yaml", "image.jpg")
    candidates = discover(RepoPaths.from_root(tmp_path))
    assert [item.path.name for item in candidates] == ["image.jpg"]


def test_discover_output_is_sorted(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "phone" / "z", "b.jpg", "A.jpg")
    _touch(tmp_path / "input" / "phone" / "a", "c.jpg")
    candidates = discover(RepoPaths.from_root(tmp_path))
    assert [(item.session, item.path.name) for item in candidates] == [("a", "c.jpg"), ("z", "A.jpg"), ("z", "b.jpg")]


def test_discover_ignores_files_directly_in_source_directory_without_session(tmp_path: Path) -> None:
    _touch(tmp_path / "input" / "phone", "root.jpg", "session.yaml")
    assert discover(RepoPaths.from_root(tmp_path)) == []
