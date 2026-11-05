"""Runtime estimate, calibration, and candidate ordering coverage."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from flurseg.config import load_config
from flurseg.estimate import (
    DEFAULT_SECONDS_PER_MP,
    DEFAULT_SECONDS_PER_VARIANT,
    Estimate,
    SourceEstimate,
    append_timing_entry,
    count_variants,
    estimate,
    fit_cost_model,
    format_duration,
    load_timing_entries,
    order_candidates,
    read_size,
    render_estimate,
    timing_fingerprint,
)
from flurseg.ingest.discover import Candidate
from flurseg.paths import RepoPaths
from flurseg.segment.tiling import iter_tiles


def test_format_duration_covers_short_hours_days_and_negative_values() -> None:
    assert format_duration(45) == "45 s"
    assert format_duration(12 * 60) == "12 min"
    assert format_duration(3 * 3600 + 5 * 60) == "3 h 05 min"
    assert format_duration(47 * 3600) == "47 h 00 min"
    assert format_duration(49 * 3600) == "2 d 01 h"
    assert format_duration(-5) == "0 s"


def test_read_size_reads_jpg_tif_and_rejects_corrupt_file(tmp_path: Path) -> None:
    jpg = tmp_path / "image.jpg"
    Image.fromarray(np.zeros((7, 11, 3), np.uint8)).save(jpg)
    tif = tmp_path / "image.tif"
    with rasterio.open(
        tif, "w", driver="GTiff", width=13, height=5, count=1, dtype="uint8", transform=from_origin(0, 0, 1, 1)
    ) as dataset:
        dataset.write(np.zeros((1, 5, 13), np.uint8))
    corrupt = tmp_path / "broken.jpg"
    corrupt.write_bytes(b"not an image")
    assert read_size(jpg) == (11, 7)
    assert read_size(tif) == (13, 5)
    assert read_size(corrupt) is None


def test_count_variants_matches_runner_tiles_for_configured_scales(tmp_root: Path) -> None:
    config = load_config(RepoPaths.from_root(tmp_root)).pipeline
    for width, height in ((4896, 3672), (2048, 1536), (2048, 2048), (700, 900)):
        direct = sum(
            len(
                list(
                    iter_tiles(
                        max(1, round(width * item.factor)),
                        max(1, round(height * item.factor)),
                        config.tiling.tile_size,
                        config.tiling.min_overlap,
                    )
                )
            )
            for item in config.scales
        ) * len(config.tta.variants)
        assert count_variants(width, height, config) == direct
    assert count_variants(4896, 3672, config) == 30
    assert count_variants(2048, 1536, config) == 8
    assert count_variants(2048, 2048, config) == 13
    config.tiling.min_overlap = 0.25
    assert count_variants(4896, 3672, config) == 44


def test_fit_cost_model_uses_current_fingerprint_and_defaults_for_sparse_data() -> None:
    fingerprint = "current"
    entries = [
        {"fingerprint": fingerprint, "variants": index + 1, "megapixels": index * 0.7 + 1, "seconds": 2 * (index + 1) + 3 * (index * 0.7 + 1)}
        for index in range(10)
    ]
    entries.extend({"fingerprint": "old", "variants": 100, "megapixels": 100, "seconds": 1} for _ in range(3))
    per_variant, per_mp = fit_cost_model(entries, fingerprint)
    assert abs(per_variant - 2) < 0.01
    assert abs(per_mp - 3) < 0.01
    assert fit_cost_model(entries[:7], fingerprint) == (DEFAULT_SECONDS_PER_VARIANT, DEFAULT_SECONDS_PER_MP)


def test_timing_entries_append_trim_and_write_atomically(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "prelabel_timing.json"
    for index in range(205):
        append_timing_entry(path, {"fingerprint": "x", "variants": index, "megapixels": 1, "seconds": 1})
    entries = load_timing_entries(path)
    assert len(entries) == 200
    assert entries[0]["variants"] == 5
    assert not list(tmp_path.glob("*.tmp"))
    previous = path.read_bytes()
    monkeypatch.setattr("flurseg.io.atomic.os.replace", lambda *_: (_ for _ in ()).throw(OSError("replace failed")))
    try:
        append_timing_entry(path, {"fingerprint": "x", "variants": 999, "megapixels": 1, "seconds": 1})
    except OSError:
        pass
    else:
        raise AssertionError("atomic write should propagate a replacement failure")
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob("*.tmp"))


def test_estimate_aggregates_sources_and_unreadable_headers(tmp_root: Path) -> None:
    paths = RepoPaths.from_root(tmp_root)
    first = tmp_root / "first.jpg"
    second = tmp_root / "second.jpg"
    broken = tmp_root / "broken.jpg"
    Image.fromarray(np.zeros((100, 200, 3), np.uint8)).save(first)
    Image.fromarray(np.zeros((50, 100, 3), np.uint8)).save(second)
    broken.write_bytes(b"bad")
    candidates = [
        Candidate(first, "drone", "a"),
        Candidate(second, "phone", "b"),
        Candidate(broken, "phone", "b"),
    ]
    result = estimate(candidates, load_config(paths).pipeline, [])
    expected_variants = count_variants(200, 100, load_config(paths).pipeline)
    expected_phone_variants = count_variants(100, 50, load_config(paths).pipeline)
    assert result.files == 2 and result.unreadable == 1
    assert result.sources["drone"].files == 1 and result.sources["phone"].files == 1
    assert result.sources["drone"].megapixels == 0.02 and result.sources["phone"].megapixels == 0.005
    assert result.sources["drone"].variants == expected_variants
    assert result.sources["phone"].variants == expected_phone_variants
    assert result.sources["drone"].seconds == DEFAULT_SECONDS_PER_VARIANT * expected_variants + DEFAULT_SECONDS_PER_MP * 0.02
    assert result.sources["phone"].seconds == DEFAULT_SECONDS_PER_VARIANT * expected_phone_variants + DEFAULT_SECONDS_PER_MP * 0.005


def test_render_estimate_includes_finish_and_only_shows_long_run_tip() -> None:
    now = datetime(2026, 11, 4, 10, 0)
    short = Estimate(sources={"phone": SourceEstimate(files=1, megapixels=1, seconds=60)}, files=1, megapixels=1, seconds=60)
    long = Estimate(sources={"drone": SourceEstimate(files=2, megapixels=100, seconds=13 * 3600)}, files=2, megapixels=100, seconds=13 * 3600)
    short_output = render_estimate(short, now)
    assert "Total: 1 files, 1.0 MP, ≈ 1 min (≈ 0.0 days)" in short_output
    assert "Finish about Wed 04.11. 10:01 if started now" in short_output
    assert "phone        1      1.0  1 min" in short_output
    assert "Upper bound: duplicates are skipped quickly." in short_output
    assert "Tip:" not in short_output
    long_output = render_estimate(long, now)
    assert "drone        2    100.0  13 h 00 min" in long_output
    assert "Total: 2 files, 100.0 MP, ≈ 13 h 00 min (≈ 0.5 days)" in long_output
    assert "Finish about Wed 04.11. 23:00 if started now" in long_output
    assert "Tip:" in long_output


def test_order_candidates_spreads_bit_reversal_and_interleaves_sessions(tmp_path: Path) -> None:
    one = [Candidate(tmp_path / f"one-{index}.jpg", "drone", "one") for index in range(8)]
    assert [item.path.name for item in order_candidates(one)] == [
        "one-0.jpg", "one-4.jpg", "one-2.jpg", "one-6.jpg", "one-1.jpg", "one-5.jpg", "one-3.jpg", "one-7.jpg",
    ]
    candidates = [
        *(Candidate(tmp_path / f"a-{index}.jpg", "drone", "a") for index in range(3)),
        *(Candidate(tmp_path / f"b-{index}.jpg", "drone", "b") for index in range(2)),
        Candidate(tmp_path / "c-0.jpg", "phone", "c"),
    ]
    spread = order_candidates(candidates)
    assert [item.session for item in spread] == ["a", "b", "c", "a", "b", "a"]
    assert order_candidates(candidates, "session") == candidates


def test_timing_fingerprint_changes_when_inference_settings_change(tmp_root: Path) -> None:
    config = load_config(RepoPaths.from_root(tmp_root)).pipeline
    original = timing_fingerprint(config)
    config.tta.variants = ["identity", "hflip"]
    assert timing_fingerprint(config) != original
