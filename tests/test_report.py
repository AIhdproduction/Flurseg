"""Unit and metric validation for Flurseg quality reporting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from flurseg.artifacts import ArtifactStore
from flurseg.config import load_config
from flurseg.io.atomic import atomic_save_png
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths
from flurseg.report.metrics import compute_metrics
from flurseg.report.render import generate_report


def _insert_record(
    paths: RepoPaths,
    image_id: str,
    *,
    status: str = "ingested",
    source: str = "drone",
    session: str = "session_1",
    is_test_session: bool = False,
    spot_check: bool = False,
    class_pixel_counts: dict[str, int] | None = None,
    review_changed_pixels: dict[str, int] | None = None,
    processing_seconds: float | None = None,
    review_seconds: float | None = None,
) -> ImageRecord:
    """Insert a synthetic image record directly into the repository manifest."""
    paths.ensure_runtime_dirs()
    work_dir = f"work/{image_id}"
    record = ImageRecord(
        id=image_id,
        hash=f"hash_{image_id}",
        source=source,
        session=session,
        gsd_method="sensor",
        original_path=f"input/{source}/{session}/{image_id}.jpg",
        image_path=f"{work_dir}/image.jpg",
        work_dir=work_dir,
        width=200,
        height=200,
        status=status,
        is_test_session=is_test_session,
        spot_check=spot_check,
        class_pixel_counts=class_pixel_counts,
        review_changed_pixels=review_changed_pixels,
        processing_seconds=processing_seconds,
        review_seconds=review_seconds,
    )
    with Database(paths.database) as connection:
        return ManifestRepository(connection).insert(record)


def test_status_counts_aggregation(tmp_root: Path) -> None:
    """Status counts correctly tally total, per-status, per-source, and per-session metrics."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(paths, "img_1", status="ingested", source="drone", session="s1")
    _insert_record(paths, "img_2", status="prelabeled", source="drone", session="s1")
    _insert_record(paths, "img_3", status="auto_accepted", source="drone", session="s2")
    _insert_record(paths, "img_4", status="needs_review", source="ortho", session="s3")
    _insert_record(paths, "img_5", status="reviewed", source="ortho", session="s3")
    _insert_record(paths, "img_6", status="rejected", source="phone", session="s4", is_test_session=True)

    metrics = compute_metrics(paths, cfg)

    assert metrics.status_counts.total == 6
    assert metrics.status_counts.by_status["ingested"] == 1
    assert metrics.status_counts.by_status["prelabeled"] == 1
    assert metrics.status_counts.by_status["auto_accepted"] == 1
    assert metrics.status_counts.by_status["needs_review"] == 1
    assert metrics.status_counts.by_status["reviewed"] == 1
    assert metrics.status_counts.by_status["rejected"] == 1

    assert metrics.status_counts.by_source["drone"] == 3
    assert metrics.status_counts.by_source["ortho"] == 2
    assert metrics.status_counts.by_source["phone"] == 1

    assert metrics.status_counts.by_session["s1"] == 2
    assert metrics.status_counts.by_session["s2"] == 1

    test_entries = [row for row in metrics.status_counts.breakdown if row["is_test_session"]]
    assert len(test_entries) == 1
    assert test_entries[0]["session"] == "s4"
    assert test_entries[0]["source"] == "phone"


def test_class_pixel_distribution_from_manifest(tmp_root: Path) -> None:
    """Class pixel distribution aggregates counts per source and overall."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(
        paths,
        "drone_1",
        source="drone",
        session="s1",
        class_pixel_counts={"1": 500, "4": 1500},
    )
    _insert_record(
        paths,
        "drone_2",
        source="drone",
        session="s1",
        class_pixel_counts={"1": 300, "4": 700},
    )
    _insert_record(
        paths,
        "ortho_1",
        source="ortho",
        session="s2",
        class_pixel_counts={"1": 200, "5": 800},
    )

    metrics = compute_metrics(paths, cfg)

    assert metrics.class_distribution.by_source["drone"]["1"] == 800
    assert metrics.class_distribution.by_source["drone"]["4"] == 2200
    assert metrics.class_distribution.by_source["ortho"]["1"] == 200
    assert metrics.class_distribution.by_source["ortho"]["5"] == 800

    assert metrics.class_distribution.total["1"] == 1000
    assert metrics.class_distribution.total["4"] == 2200
    assert metrics.class_distribution.total["5"] == 800

    assert metrics.class_distribution.percentages_by_source["drone"]["1"] == pytest.approx(800 / 3000)
    assert metrics.class_distribution.percentages_total["1"] == pytest.approx(1000 / 4000)


def test_spot_check_error_rate_below_limit(tmp_root: Path) -> None:
    """Spot check error rate below the configured limit does not raise a warning flag."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(
        paths,
        "spot_ok",
        status="reviewed",
        spot_check=True,
        class_pixel_counts={"4": 10000},
        review_changed_pixels={"4": 100},
    )

    metrics = compute_metrics(paths, cfg)

    assert metrics.spot_check.total_error_rate == pytest.approx(0.01)
    assert metrics.spot_check.per_class_error_rate["4"] == pytest.approx(0.01)
    assert metrics.spot_check.warning_flag is False
    assert metrics.warning_flag is False
    assert len(metrics.spot_check.classes_above_limit) == 0


def test_spot_check_error_rate_above_limit_triggers_warning(tmp_root: Path) -> None:
    """Spot check error rate exceeding limit raises warning and lists offending classes."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(
        paths,
        "spot_bad",
        status="reviewed",
        spot_check=True,
        class_pixel_counts={"4": 10000},
        review_changed_pixels={"4": 500},
    )

    metrics = compute_metrics(paths, cfg)

    assert metrics.spot_check.total_error_rate == pytest.approx(0.05)
    assert metrics.spot_check.per_class_error_rate["4"] == pytest.approx(0.05)
    assert metrics.spot_check.warning_flag is True
    assert metrics.warning_flag is True
    assert "4" in metrics.spot_check.classes_above_limit


def test_prompt_and_class_precision_vs_reviewed_masks(tmp_root: Path) -> None:
    """Prompt and class precision correctly evaluate pixels won by SAM 3 against reviewed masks."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)
    paths.ensure_runtime_dirs()

    image_id = "rev_1"
    work_dir = paths.work_dir / image_id
    store = ArtifactStore(work_dir)

    prompt_list = cfg.prompt_list()
    target_prompt, target_class_id = prompt_list[0]

    prompt_winner = np.full((100, 100), 255, dtype=np.uint8)
    prompt_winner[:50, :50] = 0

    reviewed_mask = np.full((100, 100), 1, dtype=np.uint8)
    reviewed_mask[:50, :40] = target_class_id

    store.write_prompt_winner(prompt_winner)
    store.write_proposed_mask(prompt_winner)
    atomic_save_png(paths.pool_masks_dir / f"{image_id}.png", reviewed_mask)

    _insert_record(paths, image_id, status="reviewed")

    metrics = compute_metrics(paths, cfg)

    expected_won = 50 * 50
    expected_correct = 50 * 40
    expected_precision = expected_correct / expected_won

    assert metrics.prompt_precision[target_prompt] == pytest.approx(expected_precision)
    target_class_def = cfg.class_by_id(target_class_id)
    assert metrics.class_precision[target_class_def.name] == pytest.approx(expected_precision)


def test_suggested_thresholds_computation(tmp_root: Path) -> None:
    """Suggested thresholds identify lowest score bin reaching 90% precision on reviewed ground truth."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)
    paths.ensure_runtime_dirs()

    image_id = "thresh_1"
    work_dir = paths.work_dir / image_id
    store = ArtifactStore(work_dir)

    tree_def = cfg.class_by_name("tree")
    tree_id = tree_def.id

    score_map = np.zeros((100, 100), dtype=np.float32)
    score_map[:50, :] = 0.20
    score_map[50:, :] = 0.60

    reviewed_mask = np.zeros((100, 100), dtype=np.uint8)
    reviewed_mask[:50, :] = 1
    reviewed_mask[50:, :] = tree_id

    store.write_scores(scores={"tree": score_map}, scales={"tree": 1.0})
    atomic_save_png(paths.pool_masks_dir / f"{image_id}.png", reviewed_mask)

    _insert_record(paths, image_id, status="reviewed")

    metrics = compute_metrics(paths, cfg)

    assert metrics.suggested_thresholds["tree"] == pytest.approx(0.60, abs=0.05)


def test_generate_report_writes_timestamped_and_latest_html(tmp_root: Path) -> None:
    """generate_report writes self-contained timestamped and latest.html files."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(paths, "sample_1", status="auto_accepted", source="drone", session="s1")

    returned_path = generate_report(paths, cfg)

    assert isinstance(returned_path, Path)
    assert returned_path.is_file()
    assert returned_path.parent == paths.reports_dir
    assert returned_path.name.startswith("report_")
    assert returned_path.name.endswith(".html")

    latest_path = paths.reports_dir / "latest.html"
    assert latest_path.is_file()

    content = returned_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "Flurseg Quality Report" in content
    assert "sample_1" in content or "drone" in content


def test_no_external_urls_in_generated_html(tmp_root: Path) -> None:
    """The generated HTML is entirely self-contained without http:// or https:// links."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(
        paths,
        "sample_url_check",
        status="reviewed",
        spot_check=True,
        class_pixel_counts={"4": 2000},
        review_changed_pixels={"4": 50},
        processing_seconds=2.4,
        review_seconds=15.0,
    )

    report_path = generate_report(paths, cfg)
    content = report_path.read_text(encoding="utf-8")

    assert "http://" not in content
    assert "https://" not in content
    assert "<svg" in content
    assert "</svg>" in content


def test_empty_manifest_handled_gracefully(tmp_root: Path) -> None:
    """generate_report produces valid output even when the database has zero records."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    report_path = generate_report(paths, cfg)
    assert report_path.is_file()

    content = report_path.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in content
    assert "http://" not in content
    assert "https://" not in content


def test_timing_metrics_aggregation(tmp_root: Path) -> None:
    """Processing and review times are accurately averaged."""
    paths = RepoPaths.from_root(tmp_root)
    cfg = load_config(paths)

    _insert_record(paths, "t1", status="prelabeled", processing_seconds=1.0)
    _insert_record(paths, "t2", status="prelabeled", processing_seconds=3.0)
    _insert_record(paths, "t3", status="reviewed", review_seconds=12.0)
    _insert_record(paths, "t4", status="reviewed", review_seconds=18.0)

    metrics = compute_metrics(paths, cfg)

    assert metrics.timing.total_images_processed == 2
    assert metrics.timing.avg_processing_seconds == pytest.approx(2.0)
    assert metrics.timing.total_processing_seconds == pytest.approx(4.0)

    assert metrics.timing.total_images_reviewed == 2
    assert metrics.timing.avg_review_seconds == pytest.approx(15.0)
    assert metrics.timing.total_review_seconds == pytest.approx(30.0)
