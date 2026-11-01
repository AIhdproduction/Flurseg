"""Hygiene tests ensuring source files are tracked and runtime data is ignored."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _skip_if_git_unavailable(repo_root: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git binary is not available on PATH")
    if not (repo_root / ".git").exists():
        pytest.skip("repository root has no .git directory")


def test_no_source_file_is_git_ignored() -> None:
    _skip_if_git_unavailable(REPO_ROOT)

    source_files: list[Path] = []

    flurseg_dir = REPO_ROOT / "flurseg"
    extensions = {".py", ".js", ".css", ".html", ".j2"}
    if flurseg_dir.is_dir():
        for path in flurseg_dir.rglob("*"):
            if path.is_file() and path.suffix in extensions and "__pycache__" not in path.parts:
                source_files.append(path)

    tests_dir = REPO_ROOT / "tests"
    if tests_dir.is_dir():
        for path in tests_dir.glob("*.py"):
            if path.is_file() and "__pycache__" not in path.parts:
                source_files.append(path)

    config_dir = REPO_ROOT / "config"
    if config_dir.is_dir():
        for path in config_dir.glob("*.yaml"):
            if path.is_file():
                source_files.append(path)

    for path in REPO_ROOT.glob("*.bat"):
        if path.is_file():
            source_files.append(path)

    root_files = [
        "README.md",
        "LICENSE",
        "NOTICE.md",
        "pyproject.toml",
        "requirements.txt",
    ]
    for filename in root_files:
        path = REPO_ROOT / filename
        if path.is_file():
            source_files.append(path)

    rel_posix_paths = sorted({p.relative_to(REPO_ROOT).as_posix() for p in source_files})
    assert rel_posix_paths, "No source files were collected for git ignore check"

    try:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=REPO_ROOT,
            input="\n".join(rel_posix_paths).encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git executable could not be executed")

    assert result.returncode in (0, 1), f"git check-ignore failed with code {result.returncode}: {result.stderr}"
    ignored_source_files = [
        line.strip().replace("\\", "/")
        for line in result.stdout.decode("utf-8").splitlines()
        if line.strip()
    ]
    assert not ignored_source_files, (
        f"The following source files are ignored by git:\n{ignored_source_files}"
    )


def test_runtime_folders_are_ignored() -> None:
    _skip_if_git_unavailable(REPO_ROOT)

    expected_ignored = [
        "train/dataset/x.png",
        "review/x/image.jpg",
        "work/x/scores.npz",
        "input/drone/s/a.jpg",
        "models/sam3.pt",
        "flurseg.db",
    ]
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin"],
            cwd=REPO_ROOT,
            input="\n".join(expected_ignored).encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git executable could not be executed")

    assert result.returncode in (0, 1), f"git check-ignore failed with code {result.returncode}: {result.stderr}"
    ignored_paths = [
        line.strip().replace("\\", "/")
        for line in result.stdout.decode("utf-8").splitlines()
        if line.strip()
    ]
    assert set(ignored_paths) == set(expected_ignored)
