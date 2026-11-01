"""Tests for the lightweight CLI and environment checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from flurseg.cli import _print_samples_table, build_parser, check_environment, main
from flurseg.paths import RepoPaths


def _fake_importer(name: str) -> ModuleType:
    cuda = SimpleNamespace(
        is_available=lambda: False,
        get_device_name=lambda index: "Test GPU",
        get_device_properties=lambda index: SimpleNamespace(total_memory=0),
    )
    modules = {
        "torch": SimpleNamespace(__version__="test", cuda=cuda),
        "ultralytics": SimpleNamespace(__version__="test"),
        "ultralytics.models.sam": SimpleNamespace(SAM3SemanticPredictor=object),
        "timm": SimpleNamespace(),
        "clip": SimpleNamespace(),
        "pillow_heif": SimpleNamespace(),
        "rasterio": SimpleNamespace(),
    }
    try:
        return modules[name]
    except KeyError as error:
        raise ModuleNotFoundError(name) from error


def test_module_help_works() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "flurseg", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    subcommands = [
        "check-env",
        "prelabel",
        "reprocess",
        "gpu-smoke",
        "review",
        "build-dataset",
        "report",
        "fetch-samples",
    ]
    for subcommand in subcommands:
        assert subcommand in result.stdout


def test_parser_help_lists_all_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    subcommands = [
        "check-env",
        "prelabel",
        "reprocess",
        "gpu-smoke",
        "review",
        "build-dataset",
        "report",
        "fetch-samples",
    ]
    for subcommand in subcommands:
        assert subcommand in help_text


def test_check_environment_returns_one_when_required_import_is_missing(tmp_root: Path) -> None:
    def missing_timm(name: str) -> ModuleType:
        if name == "timm":
            raise ModuleNotFoundError("timm")
        return _fake_importer(name)

    assert check_environment(RepoPaths.from_root(tmp_root), importer=missing_timm) == 1


def test_check_environment_returns_zero_when_only_warnings_remain(tmp_root: Path) -> None:
    assert check_environment(RepoPaths.from_root(tmp_root), importer=_fake_importer) == 0


def test_build_dataset_cli_calls_target(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[RepoPaths, Any]] = []

    def fake_build_dataset(paths: RepoPaths, cfg: Any) -> dict[str, Any]:
        calls.append((paths, cfg))
        return {"tile_counts": {"train": 1, "val": 0, "test": 0}}

    monkeypatch.setattr("flurseg.dataset.build.build_dataset", fake_build_dataset)
    exit_code = main(["build-dataset", "--root", str(tmp_root)])
    assert exit_code == 0
    assert len(calls) == 1
    paths, cfg = calls[0]
    assert paths.root == tmp_root
    assert hasattr(cfg, "classes")


def test_report_cli_calls_target(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[RepoPaths, Any]] = []

    def fake_generate_report(paths: RepoPaths, cfg: Any) -> Path:
        calls.append((paths, cfg))
        return tmp_root / "report.html"

    monkeypatch.setattr("flurseg.report.render.generate_report", fake_generate_report)
    exit_code = main(["report", "--root", str(tmp_root)])
    assert exit_code == 0
    assert len(calls) == 1
    paths, cfg = calls[0]
    assert paths.root == tmp_root
    assert hasattr(cfg, "classes")


def test_fetch_samples_dry_run_cli_calls_target(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_fetch_samples(
        paths: RepoPaths | Path | str | None = None,
        *,
        dry_run: bool = False,
        only: str | None = None,
    ) -> list[dict[str, Any]]:
        calls.append({"paths": paths, "dry_run": dry_run, "only": only})
        return [
            {
                "source": "drone",
                "session": "session_1",
                "filename": "image.jpg",
                "size": 1024,
                "licence": "CC0-1.0",
                "status": "dry_run",
            }
        ]

    monkeypatch.setattr("flurseg.samples.fetch.fetch_samples", fake_fetch_samples)
    exit_code = main(["fetch-samples", "--dry-run", "--only", "drone", "--root", str(tmp_root)])
    assert exit_code == 0
    assert len(calls) == 1
    call = calls[0]
    assert call["paths"].root == tmp_root
    assert call["dry_run"] is True
    assert call["only"] == "drone"


def test_fetch_samples_without_yes_and_input_n_downloads_nothing(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    def fake_fetch_samples(
        paths: RepoPaths | Path | str | None = None,
        *,
        dry_run: bool = False,
        only: str | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(dry_run)
        return [
            {
                "source": "drone",
                "session": "session_1",
                "filename": "image.jpg",
                "size": 1024,
                "licence": "CC0-1.0",
                "status": "dry_run" if dry_run else "downloaded",
            }
        ]

    monkeypatch.setattr("flurseg.samples.fetch.fetch_samples", fake_fetch_samples)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    exit_code = main(["fetch-samples", "--root", str(tmp_root)])
    assert exit_code == 0
    assert False not in calls
    assert calls == [True]


def test_fetch_samples_total_and_prompt_with_ortho_estimate(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_records = [
        {
            "source": "ortho",
            "session": "swissimage_test",
            "filename": "ortho.tif",
            "size": int(16.8 * 1024 * 1024),
            "remote_size": 60 * 1024 * 1024,
            "estimate_size": int(16.8 * 1024 * 1024),
            "is_estimate": True,
            "status": "dry_run",
            "licence": "swisstopo",
        },
        {
            "source": "drone",
            "session": "odm_test",
            "filename": "drone.jpg",
            "size": int(7.5 * 1024 * 1024),
            "is_estimate": False,
            "status": "dry_run",
            "licence": "CC0",
        },
        {
            "source": "phone",
            "session": "commons_test",
            "filename": "phone.jpg",
            "size": int(2.0 * 1024 * 1024),
            "is_estimate": False,
            "status": "dry_run",
            "licence": "CC-BY",
        },
    ]

    monkeypatch.setattr("flurseg.samples.fetch.fetch_samples", lambda *args, **kwargs: fake_records)

    prompt_messages: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompt_messages.append(prompt)
        return "n"

    monkeypatch.setattr("builtins.input", fake_input)

    exit_code = main(["fetch-samples", "--root", str(tmp_root)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Total: 3 files, ~26 MB" in output
    assert (
        "~ = estimate: ortho files are read as a window of a Cloud-Optimized GeoTIFF; "
        "only the window is transferred."
    ) in output
    assert len(prompt_messages) == 1
    assert "Download 3 files (~26 MB)? [y/N]" in prompt_messages[0]


def test_fetch_samples_total_ignores_existing_and_skipped_rows(
    tmp_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    existing_dest = tmp_root / "existing.tif"
    existing_dest.touch()

    fake_records = [
        {
            "source": "ortho",
            "session": "session_1",
            "filename": "existing.tif",
            "path": existing_dest,
            "size": int(16.8 * 1024 * 1024),
            "is_estimate": True,
            "status": "dry_run",
            "licence": "swisstopo",
        },
        {
            "source": "drone",
            "session": "session_2",
            "filename": "skipped.jpg",
            "size": 10 * 1024 * 1024,
            "status": "skipped",
            "licence": "CC0",
        },
        {
            "source": "phone",
            "session": "session_3",
            "filename": "new.jpg",
            "size": int(4.0 * 1024 * 1024),
            "status": "dry_run",
            "licence": "CC-BY",
        },
    ]

    monkeypatch.setattr("flurseg.samples.fetch.fetch_samples", lambda *args, **kwargs: fake_records)
    prompt_messages: list[str] = []
    monkeypatch.setattr("builtins.input", lambda prompt="": (prompt_messages.append(prompt), "n")[1])

    exit_code = main(["fetch-samples", "--root", str(tmp_root)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Total: 1 files, ~4 MB" in output
    assert "Download 1 files (~4 MB)? [y/N]" in prompt_messages[0]


def test_print_samples_table_estimate_footnote_and_total(capsys: pytest.CaptureFixture[str]) -> None:
    records = [
        {
            "source": "ortho",
            "session": "test_session",
            "filename": "ortho.tif",
            "size": int(16.8 * 1024 * 1024),
            "is_estimate": True,
            "status": "dry_run",
            "licence": "swisstopo",
        }
    ]
    _print_samples_table(records)
    out = capsys.readouterr().out
    assert "Total: 1 files, ~17 MB" in out
    assert (
        "~ = estimate: ortho files are read as a window of a Cloud-Optimized GeoTIFF; "
        "only the window is transferred."
    ) in out
