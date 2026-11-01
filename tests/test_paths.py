"""Tests for repository-root resolution and contained runtime directories."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from flurseg.cli import build_parser
from flurseg.paths import RepoPaths, ensure_inside, is_inside


def test_default_root_is_parent_of_package_directory() -> None:
    paths = RepoPaths.from_root()
    assert paths.root == Path(__file__).resolve().parents[1]


def test_explicit_root_from_cli_option(tmp_root: Path) -> None:
    arguments = build_parser().parse_args(["--root", str(tmp_root), "check-env"])
    paths = RepoPaths.from_root(arguments.root)
    assert paths.root == tmp_root.resolve()


def test_environment_root_overrides_default(tmp_root: Path, monkeypatch) -> None:
    monkeypatch.setenv("FLURSEG_ROOT", str(tmp_root))
    assert RepoPaths.from_root().root == tmp_root.resolve()


def test_ensure_runtime_dirs_creates_only_directories_below_root(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    paths.ensure_runtime_dirs()
    for directory in paths.runtime_dirs:
        assert directory.is_dir()
        assert directory.resolve().is_relative_to(tmp_root.resolve())


def test_inside_path_accepted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    child_file = root / "data" / "sample.txt"
    child_file.parent.mkdir()
    child_file.write_text("hello", encoding="utf-8")

    assert is_inside(child_file, root) is True
    normalized = ensure_inside(child_file, root)
    assert normalized.name == "sample.txt"
    assert is_inside(root, root) is True
    assert ensure_inside(root, root).name == root.name


def test_parent_escape_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    escape_nested = root / "subdir" / ".." / ".." / "outside.txt"
    assert is_inside(escape_nested, root) is False
    with pytest.raises(ValueError):
        ensure_inside(escape_nested, root)

    escape_direct = root / ".." / "outside.txt"
    assert is_inside(escape_direct, root) is False
    with pytest.raises(ValueError):
        ensure_inside(escape_direct, root)


def test_sibling_directory_with_common_prefix_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root2 = tmp_path / "root2"
    root.mkdir()
    root2.mkdir()
    target = root2 / "file.txt"

    assert is_inside(target, root) is False
    with pytest.raises(ValueError):
        ensure_inside(target, root)


def test_non_existent_nested_target_accepted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "deeply" / "nested" / "future_tile.png"

    assert not target.exists()
    assert is_inside(target, root) is True
    normalized = ensure_inside(target, root)
    assert normalized.name == "future_tile.png"
    assert not normalized.exists()


def test_extended_length_path_prefix_accepted(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    target = root / "review" / "tile.jpg"

    resolved_str = str(target.resolve())
    if resolved_str.startswith("\\\\?\\"):
        extended_path = Path(resolved_str)
    else:
        extended_path = Path(f"\\\\?\\{resolved_str}")

    assert is_inside(extended_path, root) is True
    normalized = ensure_inside(extended_path, root)
    assert not str(normalized).startswith("\\\\?\\")
    assert normalized.name == "tile.jpg"


def test_mixed_case_path_accepted_on_windows(tmp_path: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("Mixed case comparison is Windows-specific")

    root = tmp_path / "CaseRoot"
    root.mkdir()
    target = root / "SubDir" / "TestFile.txt"

    target_str = str(target)
    if ":" in target_str:
        drive, rest = target_str.split(":", 1)
        mixed_str = drive.lower() + ":" + rest.upper()
    else:
        mixed_str = target_str.swapcase()

    mixed_target = Path(mixed_str)
    assert is_inside(mixed_target, root) is True
    normalized = ensure_inside(mixed_target, root)
    assert normalized.name.lower() == "testfile.txt"


def test_short_8_3_name_accepted(tmp_path: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("8.3 short names are Windows-specific")

    import ctypes

    root = tmp_path / "very_long_root_name_for_testing"
    root.mkdir()
    target = root / "nested_dir" / "future_file.txt"

    buffer = ctypes.create_unicode_buffer(1024)
    result = ctypes.windll.kernel32.GetShortPathNameW(str(root), buffer, 1024)
    if result == 0 or result > 1024:
        pytest.skip("GetShortPathNameW failed or unavailable on this drive")

    short_root_str = buffer.value
    if short_root_str.lower() == str(root).lower():
        pytest.skip("Filesystem returned identical string (8.3 short names disabled)")

    short_root = Path(short_root_str)
    assert is_inside(target, short_root) is True
    normalized = ensure_inside(target, short_root)
    assert normalized.name == "future_file.txt"

    short_target = short_root / "nested_dir" / "future_file.txt"
    assert is_inside(short_target, root) is True
    assert ensure_inside(short_target, root).name == "future_file.txt"
