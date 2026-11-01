"""Static assertions verifying contents of selftest.bat."""

from __future__ import annotations

from pathlib import Path


def test_selftest_bat_structure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bat_path = repo_root / "selftest.bat"
    assert bat_path.is_file(), f"File {bat_path} does not exist"

    text = bat_path.read_text(encoding="utf-8")
    expected_tokens = [
        "check-env",
        "pytest",
        "ruff",
        "SELFTEST PASS",
        "SELFTEST FAIL",
        "pause",
        "exit /b 1",
    ]
    for token in expected_tokens:
        assert token in text, f"Expected {token!r} in selftest.bat"
