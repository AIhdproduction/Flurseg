"""Synthetic coverage for the safe input-image deduplication workflow."""

from __future__ import annotations

import json
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFilter

from flurseg.cli import build_parser, main
from flurseg.ingest import dedup
from flurseg.ingest.dedup import DedupOptions, _capture_time, extract_feature, run_dedup, undo_dedup
from flurseg.io.hashing import sha256_file
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths


def _image(path: Path, seed: int, *, when: datetime | None = None, xmp: str = "") -> Path:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(15, 240, size=(180, 240, 3), dtype=np.uint8)
    pixels[30:80, 40:180] = np.clip(pixels[30:80, 40:180].astype(int) + 35, 0, 255)
    exif = Image.Exif()
    if when:
        exif[36867] = when.strftime("%Y:%m:%d %H:%M:%S")
    Image.fromarray(pixels).save(path, format="JPEG", quality=94, exif=exif)
    if xmp:
        with path.open("ab") as stream:
            stream.write(xmp.encode("utf-8"))
    return path


def _subifd_image(path: Path, seed: int, when: datetime) -> Path:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(15, 240, size=(180, 240, 3), dtype=np.uint8)
    pixels[30:80, 40:180] = np.clip(pixels[30:80, 40:180].astype(int) + 35, 0, 255)
    exif = Image.Exif()
    exif[0x8769] = {
        36867: when.strftime("%Y:%m:%d %H:%M:%S"),
        37521: f"{when.microsecond:06d}",
    }
    Image.fromarray(pixels).save(path, format="JPEG", quality=94, exif=exif)
    return path


def _root(tmp_root: Path, source: str = "phone", session: str = "s") -> tuple[RepoPaths, Path]:
    paths = RepoPaths.from_root(tmp_root)
    folder = paths.input_dir / source / session
    folder.mkdir(parents=True)
    return paths, folder


def _record(content_hash: str) -> ImageRecord:
    return ImageRecord(
        id="phone_s_0001", hash=content_hash, source="phone", session="s", gsd_method="unknown",
        original_path="input/phone/s/a.jpg", image_path="work/a.jpg", work_dir="work/a", width=1, height=1,
    )


def test_exact_duplicate_and_ingested_hash(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    original = _image(folder / "a.jpg", 1)
    duplicate = folder / "b.jpg"
    shutil.copy2(original, duplicate)
    repository = ManifestRepository(sqlite3.connect(paths.database))
    repository.insert(_record(sha256_file(original)))
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    assert {item.candidate.path.name: item.reason for item in result.decisions} == {
        "a.jpg": "already_ingested", "b.jpg": "already_ingested"
    }


def test_exact_duplicate_inside_input(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    first = _image(folder / "a.jpg", 4)
    shutil.copy2(first, folder / "b.jpg")
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    reasons = {item.candidate.path.name: item.reason for item in result.decisions}
    assert reasons == {"a.jpg": "unique", "b.jpg": "exact"}


def test_perceptual_brightened_copy_is_close_but_other_texture_is_not(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    source = _image(folder / "a.jpg", 12)
    with Image.open(source) as opened:
        bright = np.clip(np.asarray(opened).astype(int) + 8, 0, 255).astype(np.uint8)
    Image.fromarray(bright).save(folder / "b.jpg", quality=78)
    _image(folder / "c.jpg", 19)
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    status = {item.candidate.path.name: item.decision for item in result.decisions}
    assert sum(value == "drop" for value in status.values()) == 1
    assert status["c.jpg"] == "keep"


def test_drone_overlap_and_cross_session_date(tmp_root: Path) -> None:
    paths, first = _root(tmp_root, "drone", "early")
    second = paths.input_dir / "drone" / "late"
    second.mkdir(parents=True)
    base = datetime(2026, 1, 1, 10)
    xmp = 'drone-dji:RelativeAltitude="50" drone-dji:GimbalPitchDegree="-89.9" drone-dji:GimbalYawDegree="0" drone-dji:GpsLatitude="47.0" drone-dji:GpsLongitude="8.0" drone-dji:CalibratedFocalLength="3725" drone-dji:ImageWidth="5280" drone-dji:ImageHeight="3956"'
    for index in range(5):
        longitude = 8.0 + index * 10 / (111_320 * np.cos(np.deg2rad(47.0)))
        line_xmp = xmp.replace('GpsLongitude="8.0"', f'GpsLongitude="{longitude:.8f}"')
        _image(first / f"DJI_{(base + timedelta(seconds=index)).strftime('%Y%m%d%H%M%S')}_{index}_V.JPG", index + 30, xmp=line_xmp)
    late_xmp = xmp
    _image(second / "DJI_20260201000000_1_V.JPG", 99, xmp=late_xmp)
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0, drone_mode="overlap"))
    early = [item for item in result.decisions if item.candidate.session == "early"]
    assert [item.decision for item in early] == ["keep", "drop", "drop", "drop", "keep"]
    assert next(item for item in result.decisions if item.candidate.session == "late").decision == "keep"


def test_phone_burst_keeps_sharpest(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    moment = datetime(2026, 1, 1, 10)
    sharp = _image(folder / "sharp.jpg", 55, when=moment)
    with Image.open(sharp) as opened:
        for index in (1, 2):
            opened.filter(ImageFilter.GaussianBlur(radius=3)).save(
                folder / f"blur{index}.jpg", quality=90, exif=opened.getexif()
            )
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    kept = [item.candidate.path.name for item in result.decisions if item.decision == "keep"]
    assert kept == ["sharp.jpg"]


def test_minimum_keep_and_max_drop_guard(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    first = _image(folder / "a.jpg", 3)
    for index in range(1, 9):
        shutil.copy2(first, folder / f"b{index}.jpg")
    protected = run_dedup(paths, DedupOptions(min_keep_per_session=3))
    assert sum(item.decision == "keep" for item in protected.decisions) == 1
    guarded = run_dedup(paths, DedupOptions(min_keep_per_session=0, max_drop_fraction=0.4), apply=True)
    assert guarded.exit_code == 1
    assert all(item.candidate.path.exists() for item in guarded.decisions)


def test_apply_undo_and_non_images_are_untouched(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    original = _image(folder / "a.jpg", 23)
    duplicate = folder / "b.jpg"
    shutil.copy2(original, duplicate)
    for name in ("flight.MRK", "flight.nav", "session.yaml"):
        (folder / name).write_text("metadata", encoding="utf-8")
    before = duplicate.read_bytes()
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0), apply=True)
    assert result.exit_code == 0
    manifest = paths.quarantine_dir / "dedup" / result.run_id / "manifest.json"
    listed = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    assert listed and not duplicate.exists()
    assert all((folder / name).is_file() for name in ("flight.MRK", "flight.nav", "session.yaml"))
    assert undo_dedup(paths, result.run_id) == 0
    assert duplicate.read_bytes() == before
    assert undo_dedup(paths, result.run_id) == 0
    moved = Path(listed[0]["quarantine_path"])
    moved.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(duplicate, moved)
    assert undo_dedup(paths, result.run_id) == 0
    assert moved.exists()


def test_report_and_cli_dry_run(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    image = _image(folder / "a.jpg", 78)
    result = run_dedup(paths, DedupOptions())
    assert (result.report_dir / "decisions.csv").is_file()
    assert (result.report_dir / "summary.json").is_file()
    index = (result.report_dir / "index.html").read_text(encoding="utf-8")
    assert "thumbs/" in index
    assert list((result.report_dir / "thumbs").glob("*.jpg"))
    assert main(["dedup", "--root", str(tmp_root)]) == 0
    assert image.exists()
    arguments = build_parser().parse_args(["dedup", "--source", "drone", "--cross-session-days", "7"])
    assert arguments.source == "drone" and arguments.cross_session_days == 7


def test_drone_default_stride_two_keeps_every_second_image(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root, "drone")
    base = datetime(2026, 2, 1, 9)
    for index in range(7):
        _image(
            folder / f"DJI_{(base + timedelta(seconds=index)).strftime('%Y%m%d%H%M%S')}_{index}_V.JPG",
            100 + index,
        )
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    ordered = sorted(result.decisions, key=lambda item: item.candidate.path.name)
    assert [item.decision for item in ordered] == ["keep", "drop", "keep", "drop", "keep", "drop", "keep"]
    assert [item.reason for item in ordered if item.decision == "drop"] == ["drone_stride"] * 3
    assert ordered[1].kept_instead == ordered[0].key
    assert ordered[3].kept_instead == ordered[2].key


def test_drone_stride_three_keeps_indices_zero_three_and_six(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root, "drone")
    base = datetime(2026, 2, 2, 9)
    for index in range(7):
        _image(
            folder / f"DJI_{(base + timedelta(seconds=index)).strftime('%Y%m%d%H%M%S')}_{index}_V.JPG",
            120 + index,
        )
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0, drone_stride=3))
    ordered = sorted(result.decisions, key=lambda item: item.candidate.path.name)
    assert [item.decision for item in ordered] == ["keep", "drop", "drop", "keep", "drop", "drop", "keep"]


def test_drone_stride_uses_dji_sequence_not_directory_sort_order(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root, "drone")
    stamp = "20260203090000"
    for sequence, seed in ((10, 140), (2, 141), (5, 142)):
        _image(folder / f"DJI_{stamp}_{sequence}_V.JPG", seed)
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    decisions = {item.candidate.path.name: item for item in result.decisions}
    assert decisions[f"DJI_{stamp}_2_V.JPG"].decision == "keep"
    assert decisions[f"DJI_{stamp}_5_V.JPG"].reason == "drone_stride"
    assert decisions[f"DJI_{stamp}_10_V.JPG"].decision == "keep"


def test_exact_drone_duplicate_does_not_advance_stride_index(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root, "drone")
    first = _image(folder / "DJI_20260204090000_1_V.JPG", 160)
    shutil.copy2(first, folder / "DJI_20260204090001_2_V.JPG")
    _image(folder / "DJI_20260204090002_3_V.JPG", 161)
    _image(folder / "DJI_20260204090003_4_V.JPG", 162)
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    decisions = {item.candidate.path.name: item for item in result.decisions}
    assert decisions["DJI_20260204090001_2_V.JPG"].reason == "exact"
    assert decisions["DJI_20260204090002_3_V.JPG"].reason == "drone_stride"
    assert decisions["DJI_20260204090003_4_V.JPG"].decision == "keep"


def test_parser_rejects_invalid_drone_stride_and_mode() -> None:
    parser = build_parser()
    assert parser.parse_args(["dedup", "--drone-mode", "overlap"]).drone_mode == "overlap"
    with pytest.raises(SystemExit):
        parser.parse_args(["dedup", "--drone-mode", "unknown"])
    with pytest.raises(SystemExit):
        parser.parse_args(["dedup", "--drone-stride", "0"])


def test_capture_time_reads_subifd_datetime_and_ifd0_fallback(tmp_root: Path) -> None:
    _paths, folder = _root(tmp_root)
    captured = datetime(2025, 7, 30, 10, 29, 12, 345_678)
    subifd = _subifd_image(folder / "subifd.jpg", 210, captured)
    fallback = _image(folder / "fallback.jpg", 211, when=captured.replace(microsecond=0))

    assert _capture_time(subifd, {}) == captured
    assert extract_feature(subifd).captured == captured
    assert _capture_time(fallback, {}) == captured.replace(microsecond=0)


def test_phone_burst_detects_subifd_dated_images(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    moment = datetime(2026, 3, 1, 10, 0, 0, 100_000)
    sharp = _subifd_image(folder / "sharp.jpg", 220, moment)
    with Image.open(sharp) as opened:
        for index in (1, 2):
            opened.filter(ImageFilter.GaussianBlur(radius=3)).save(
                folder / f"blur{index}.jpg", quality=90, exif=opened.getexif()
            )

    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    assert [item.candidate.path.name for item in result.decisions if item.decision == "keep"] == ["sharp.jpg"]


def test_burst_phash_threshold_is_configurable(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    gallery = tmp_root / "gallery"
    gallery.mkdir()
    moment = datetime(2026, 3, 2, 10)
    features: list[tuple[Path, int]] = []
    for seed in range(230, 270):
        image = _image(gallery / f"candidate-{seed}.jpg", seed, when=moment)
        features.append((image, extract_feature(image).phash))
    pair = next(
        (left, right)
        for index, left in enumerate(features)
        for right in features[index + 1 :]
        if 9 <= (left[1] ^ right[1]).bit_count() <= 15
    )
    shutil.copy2(pair[0][0], folder / "first.jpg")
    shutil.copy2(pair[1][0], folder / "second.jpg")

    default = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    strict = run_dedup(
        paths,
        DedupOptions(min_keep_per_session=0, burst_phash_threshold=8),
    )
    assert sum(item.reason == "burst" for item in default.decisions) == 1
    assert all(item.decision == "keep" for item in strict.decisions)
    assert build_parser().parse_args(["dedup", "--burst-phash-threshold", "12"]).burst_phash_threshold == 12


def test_minimum_keep_does_not_revive_hard_duplicates_or_ingested_files(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    original = _image(folder / "a.jpg", 280)
    shutil.copy2(original, folder / "b.jpg")
    repository = ManifestRepository(sqlite3.connect(paths.database))
    repository.insert(_record(sha256_file(original)))

    ingested = run_dedup(paths, DedupOptions(min_keep_per_session=5))
    assert all(item.decision == "drop" and item.reason == "already_ingested" for item in ingested.decisions)

    stride_paths, stride_folder = _root(tmp_root / "stride")
    moment = datetime(2026, 3, 3, 10)
    for index in range(5):
        _subifd_image(
            stride_folder / f"image-{index}.jpg",
            290 + index,
            moment + timedelta(seconds=index * 10),
        )
    stride = run_dedup(
        stride_paths,
        DedupOptions(min_keep_per_session=3, phone_stride=3, phash_threshold=0),
    )
    assert sum(item.decision == "keep" for item in stride.decisions) == 3
    assert sum(item.reason == "minimum_keep" for item in stride.decisions) == 1


def test_manifest_records_pending_entry_before_move_failure(tmp_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths, folder = _root(tmp_root)
    original = _image(folder / "a.jpg", 300)
    shutil.copy2(original, folder / "b.jpg")
    shutil.copy2(original, folder / "c.jpg")
    real_move = dedup.shutil.move
    calls = 0

    def fail_second_move(source: str, target: str) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated move failure")
        return real_move(source, target)

    monkeypatch.setattr(dedup.shutil, "move", fail_second_move)
    with pytest.raises(OSError, match="simulated move failure"):
        run_dedup(
            paths,
            DedupOptions(min_keep_per_session=0),
            apply=True,
            run_id="manifest-failure",
        )

    manifest = paths.quarantine_dir / "dedup" / "manifest-failure" / "manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    assert [entry["status"] for entry in entries] == ["moved", "pending"]
    assert undo_dedup(paths, "manifest-failure") == 0
    assert (folder / "b.jpg").is_file()
    assert (folder / "c.jpg").is_file()


def test_cli_apply_uses_one_analysis_and_declined_apply_moves_nothing(
    tmp_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _paths, folder = _root(tmp_root)
    original = _image(folder / "a.jpg", 310)
    duplicate = folder / "b.jpg"
    shutil.copy2(original, duplicate)
    actual = dedup.run_dedup
    calls = 0

    def spy(*args: object, **kwargs: object) -> dedup.DedupResult:
        nonlocal calls
        calls += 1
        return actual(*args, **kwargs)

    monkeypatch.setattr(dedup, "run_dedup", spy)
    assert main(["dedup", "--root", str(tmp_root), "--apply", "--yes", "--min-keep-per-session", "0"]) == 0
    assert calls == 1
    assert not duplicate.exists()

    declined_root = tmp_root / "declined"
    _, declined_folder = _root(declined_root)
    declined_original = _image(declined_folder / "a.jpg", 311)
    declined_duplicate = declined_folder / "b.jpg"
    shutil.copy2(declined_original, declined_duplicate)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert main(["dedup", "--root", str(declined_root), "--apply", "--min-keep-per-session", "0"]) == 0
    assert declined_duplicate.is_file()


def test_already_archived_hash_is_dropped(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    image = _image(folder / "a.jpg", 320)
    archive = paths.archive_dir / "phone" / "s"
    archive.mkdir(parents=True)
    shutil.copy2(image, archive / "a.jpg")

    result = run_dedup(paths, DedupOptions(min_keep_per_session=0))
    assert result.decisions[0].reason == "already_archived"


def test_drone_overlap_drops_same_footprint_from_nearby_session(tmp_root: Path) -> None:
    paths, first = _root(tmp_root, "drone", "first")
    second = paths.input_dir / "drone" / "second"
    second.mkdir(parents=True)
    xmp = 'drone-dji:RelativeAltitude="50" drone-dji:GimbalPitchDegree="-89.9" drone-dji:GimbalYawDegree="0" drone-dji:GpsLatitude="47.0" drone-dji:GpsLongitude="8.0" drone-dji:CalibratedFocalLength="3725" drone-dji:ImageWidth="5280" drone-dji:ImageHeight="3956"'
    _image(first / "DJI_20260301090000_1_V.JPG", 330, xmp=xmp)
    _image(second / "DJI_20260304090000_1_V.JPG", 331, xmp=xmp)

    result = run_dedup(paths, DedupOptions(min_keep_per_session=0, drone_mode="overlap"))
    later = next(item for item in result.decisions if item.candidate.session == "second")
    assert later.reason == "drone_overlap"


def test_undo_does_not_overwrite_recreated_original(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root)
    original = _image(folder / "a.jpg", 340)
    duplicate = folder / "b.jpg"
    shutil.copy2(original, duplicate)
    result = run_dedup(paths, DedupOptions(min_keep_per_session=0), apply=True)
    entry = json.loads(
        (paths.quarantine_dir / "dedup" / result.run_id / "manifest.json").read_text(encoding="utf-8")
    )["files"][0]
    replacement = b"different replacement content"
    duplicate.write_bytes(replacement)

    assert undo_dedup(paths, result.run_id) == 0
    assert duplicate.read_bytes() == replacement
    assert Path(entry["quarantine_path"]).is_file()


def test_phone_stride_uses_subifd_capture_order_and_parser_rejects_zero(tmp_root: Path) -> None:
    paths, folder = _root(tmp_root, "phone", "walkaround")
    moment = datetime(2026, 3, 5, 10)
    shuffled = [4, 1, 6, 0, 5, 2, 3]
    for index in shuffled:
        _subifd_image(
            folder / f"photo-{index}.jpg",
            350 + index,
            moment + timedelta(seconds=index),
        )

    result = run_dedup(
        paths,
        DedupOptions(min_keep_per_session=0, phone_stride=3, phash_threshold=0, burst_seconds=0),
    )
    decisions = {item.candidate.path.name: item for item in result.decisions}
    assert {name for name, item in decisions.items() if item.decision == "keep"} == {
        "photo-0.jpg",
        "photo-3.jpg",
        "photo-6.jpg",
    }
    assert all(
        item.reason == "phone_stride"
        for name, item in decisions.items()
        if name not in {"photo-0.jpg", "photo-3.jpg", "photo-6.jpg"}
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["dedup", "--phone-stride", "0"])
