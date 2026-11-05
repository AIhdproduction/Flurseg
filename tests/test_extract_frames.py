"""Tests for local own-video frame extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from flurseg.cli import build_parser, main
from flurseg.extract_frames import extract_frames
from flurseg.ingest.session import SessionMetadata
from flurseg.paths import RepoPaths


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 2, (320, 240))
    assert writer.isOpened()
    for index in range(8):
        frame = _textured_frame(20 if index < 4 else 180)
        cv2.circle(frame, (40 + index * 24, 120), 26, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _textured_frame(value: int, *, blur: bool = False) -> np.ndarray:
    rows, columns = np.indices((240, 320))
    texture = ((rows * (value % 17 + 3) + columns * (value % 23 + 5) + value * 13) % 180 + 45).astype(np.uint8)
    frame = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    color = ((value * 17) % 255, (value * 47) % 255, (value * 79) % 255)
    cv2.rectangle(frame, (20 + (value * 7) % 180, 35), (150 + (value * 11) % 140, 190), color, -1)
    cv2.putText(frame, f"frame-{value}", (24, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    return cv2.GaussianBlur(frame, (31, 31), 8) if blur else frame


def _frames_video(path: Path, frames: list[np.ndarray], fps: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _paths(tmp_path: Path) -> RepoPaths:
    return RepoPaths.from_root(tmp_path)


def test_extract_frames_dry_run_writes_nothing_and_persists_metadata(tmp_path: Path) -> None:
    """Dry runs are read-only; real runs keep frames and valid session metadata under input."""
    source = tmp_path / "nested" / "drive.avi"
    source.parent.mkdir(parents=True)
    _video(source)
    paths = RepoPaths.from_root(tmp_path)
    dry = extract_frames(paths, [source.parent], session="own_drive", dry_run=True)
    assert dry[0]["planned_frames"] > 0 and not (tmp_path / "input").exists()
    result = extract_frames(paths, [source], session="own_drive", interval_s=1.0, max_frames_per_video=2)
    session = tmp_path / "input" / "phone" / "own_drive"
    assert result[0]["written"] == 2 and len(list(session.glob("*.jpg"))) == 2
    assert SessionMetadata.model_validate(yaml.safe_load((session / "session.yaml").read_text(encoding="utf-8"))).notes == "own footage"


def test_interval_uses_video_timestamps_in_output_names(tmp_path: Path) -> None:
    source = tmp_path / "timestamps.avi"
    _frames_video(source, [_textured_frame(value) for value in (20, 50, 80, 110, 140, 170)])
    extract_frames(_paths(tmp_path), [source], session="timed", interval_s=1.0)
    names = sorted(path.name for path in (tmp_path / "input" / "phone" / "timed").glob("*.jpg"))
    assert names == ["timestamps_0.jpg", "timestamps_1.jpg", "timestamps_2.jpg"]


def test_static_segments_are_deduplicated_by_phash(tmp_path: Path) -> None:
    source = tmp_path / "static.avi"
    still, changed = _textured_frame(40), _textured_frame(180)
    _frames_video(source, [still, still, still, still, changed, changed])
    records = extract_frames(_paths(tmp_path), [source], session="dedup", interval_s=0.5)
    assert records[0]["written"] == 2


def test_dark_frames_are_not_written(tmp_path: Path) -> None:
    source = tmp_path / "dark.avi"
    dark = np.zeros((240, 320, 3), dtype=np.uint8)
    _frames_video(source, [dark, _textured_frame(80), dark, _textured_frame(170)])
    records = extract_frames(_paths(tmp_path), [source], session="bright", interval_s=0.5)
    assert records[0]["written"] == 2


def test_frames_blurred_relative_to_the_video_median_are_dropped(tmp_path: Path) -> None:
    source = tmp_path / "blur.avi"
    sharp = [_textured_frame(value) for value in (40, 90, 140, 190)]
    _frames_video(source, [sharp[0], sharp[1], _textured_frame(160, blur=True), sharp[2], sharp[3]])
    records = extract_frames(_paths(tmp_path), [source], session="sharp", interval_s=0.5)
    assert 2 <= int(records[0]["written"]) < 5


def test_folder_recursion_accepts_case_insensitive_video_suffixes(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    _frames_video(root / "one.AVI", [_textured_frame(20), _textured_frame(80)])
    _frames_video(root / "nested" / "two.avi", [_textured_frame(40), _textured_frame(120)])
    (root / "nested" / "ignore.txt").write_text("not a video", encoding="utf-8")
    records = extract_frames(_paths(tmp_path), [root], session="recursive", interval_s=0.5)
    assert len(records) == 2 and sum(int(record["written"]) for record in records) == 4


def test_existing_frame_is_never_overwritten_and_gets_a_suffix(tmp_path: Path) -> None:
    source = tmp_path / "repeat.avi"
    _frames_video(source, [_textured_frame(20), _textured_frame(100)])
    paths = _paths(tmp_path)
    extract_frames(paths, [source], session="repeat", interval_s=0.5)
    extract_frames(paths, [source], session="repeat", interval_s=0.5)
    names = {path.name for path in (tmp_path / "input" / "phone" / "repeat").glob("*.jpg")}
    assert "repeat_0.jpg" in names and "repeat_0_1.jpg" in names


def test_session_metadata_preserves_cli_fields(tmp_path: Path) -> None:
    source = tmp_path / "metadata.avi"
    _frames_video(source, [_textured_frame(30), _textured_frame(120)])
    extract_frames(_paths(tmp_path), [source], session="metadata", camera="Tesla front", date="2026-11-04", area_type="urban", interval_s=0.5)
    raw = yaml.safe_load((tmp_path / "input" / "phone" / "metadata" / "session.yaml").read_text(encoding="utf-8"))
    metadata = SessionMetadata.model_validate(raw)
    assert metadata.camera == "Tesla front" and str(metadata.date) == "2026-11-04" and metadata.area_type == "urban"


def test_cli_dry_run_prints_duration_and_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "dry.avi"
    _frames_video(source, [_textured_frame(30), _textured_frame(120)])
    assert main(["extract-frames", str(source), "--session", "dry", "--dry-run", "--root", str(tmp_path)]) == 0
    assert "duration=" in capsys.readouterr().out and not (tmp_path / "input").exists()


def test_max_frames_per_video_limits_written_frames(tmp_path: Path) -> None:
    source = tmp_path / "capped.avi"
    _frames_video(source, [_textured_frame(value) for value in (20, 50, 80, 110, 140, 170)])
    records = extract_frames(_paths(tmp_path), [source], session="capped", interval_s=0.5, max_frames_per_video=2)
    assert records[0]["written"] == 2


def test_parser_supports_drone_source_and_extraction_flags() -> None:
    arguments = build_parser().parse_args(["extract-frames", "video.avi", "--session", "drone_run", "--source", "drone", "--interval", "1.5", "--phash-threshold", "7", "-y"])
    assert arguments.source == "drone" and arguments.interval == 1.5 and arguments.phash_threshold == 7


def test_corrupt_video_is_reported_while_valid_video_is_processed(tmp_path: Path) -> None:
    good = tmp_path / "good.avi"
    broken = tmp_path / "broken.avi"
    _frames_video(good, [_textured_frame(30), _textured_frame(120)])
    broken.write_bytes(b"this is not a video")
    records = extract_frames(_paths(tmp_path), [broken, good], session="mixed", interval_s=0.5)
    assert {record["status"] for record in records} == {"failed", "extracted"}
    assert sum(int(record.get("written", 0)) for record in records) == 2
