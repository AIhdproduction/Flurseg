"""Metric extraction and statistical aggregation for quality reports."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flurseg.artifacts import ArtifactStore
from flurseg.config import FlurConfig
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ManifestRepository
from flurseg.paths import RepoPaths

logger = logging.getLogger(__name__)

_STATUS_ORDER = ("ingested", "prelabeled", "auto_accepted", "needs_review", "reviewed", "rejected")


@dataclass(slots=True)
class StatusCounts:
    """Breakdown of images across ingestion statuses, data sources, and sessions."""

    total: int
    by_status: dict[str, int]
    by_source: dict[str, int]
    by_session: dict[str, int]
    breakdown: list[dict[str, Any]]


@dataclass(slots=True)
class ClassDistribution:
    """Semantic class pixel counts and relative shares per source and overall."""

    by_source: dict[str, dict[str, int]]
    total: dict[str, int]
    percentages_by_source: dict[str, dict[str, float]]
    percentages_total: dict[str, float]


@dataclass(slots=True)
class SpotCheckErrorRate:
    """Auto-accept error rate statistics computed from reviewed spot-check images."""

    total_changed_pixels: int
    total_labelled_pixels: int
    total_error_rate: float
    per_class_changed_pixels: dict[str, int]
    per_class_labelled_pixels: dict[str, int]
    per_class_error_rate: dict[str, float]
    limit: float
    warning: bool
    warning_flag: bool
    classes_above_limit: list[str]


@dataclass(slots=True)
class PromptPrecisionItem:
    """Individual prompt performance against human-reviewed ground truth."""

    index: int
    prompt: str
    class_id: int
    class_name: str
    won_pixels: int
    correct_pixels: int
    precision: float


@dataclass(slots=True)
class ClassPrecisionItem:
    """Aggregated precision and suggested threshold for one trainable class."""

    class_id: int
    class_name: str
    display_color: str
    won_pixels: int
    correct_pixels: int
    precision: float
    suggested_threshold: float | None


@dataclass(slots=True)
class PrecisionMetrics:
    """Prompt-level and class-level accuracy metrics compared to reviewed masks."""

    per_prompt: list[PromptPrecisionItem]
    per_class: list[ClassPrecisionItem]
    prompt_precision: dict[str, float]
    class_precision: dict[str, float]
    suggested_thresholds: dict[str, float | None]


@dataclass(slots=True)
class TimingMetrics:
    """Summary of prelabeling inference latency and interactive review durations."""

    total_images_processed: int
    total_images_reviewed: int
    avg_processing_seconds: float
    avg_review_seconds: float
    total_processing_seconds: float
    total_review_seconds: float


@dataclass(slots=True)
class ReportMetrics:
    """Complete structured metrics suite compiled for HTML reporting."""

    generated_at: str
    status_counts: StatusCounts
    class_distribution: ClassDistribution
    spot_check: SpotCheckErrorRate
    precision: PrecisionMetrics
    timing: TimingMetrics
    warning_flag: bool
    spot_check_error_limit: float

    @property
    def warning(self) -> bool:
        """Alias for warning_flag."""
        return self.warning_flag

    @property
    def spot_check_error_rate_per_class(self) -> dict[str, float]:
        """Convenience accessor for per-class spot check error rate."""
        return self.spot_check.per_class_error_rate

    @property
    def prompt_precision(self) -> dict[str, float]:
        """Convenience accessor mapping prompt text to precision value."""
        return self.precision.prompt_precision

    @property
    def class_precision(self) -> dict[str, float]:
        """Convenience accessor mapping class name to precision value."""
        return self.precision.class_precision

    @property
    def suggested_thresholds(self) -> dict[str, float | None]:
        """Convenience accessor mapping class name to suggested presence threshold."""
        return self.precision.suggested_thresholds

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _collect_rows(paths: RepoPaths) -> list[dict[str, Any]]:
    if not paths.database.is_file():
        return []
    with Database(paths.database) as connection:
        repo = ManifestRepository(connection)
        return repo.query()


def compute_status_counts(rows: list[dict[str, Any]]) -> StatusCounts:
    """Aggregate row counts across statuses, sources, and sessions."""
    by_status = {status: 0 for status in _STATUS_ORDER}
    by_source: dict[str, int] = {}
    by_session: dict[str, int] = {}
    session_map: dict[tuple[str, str, bool], dict[str, int]] = {}

    for row in rows:
        status = row.get("status") or "unknown"
        source = row.get("source") or "unknown"
        session = row.get("session") or "unknown"
        is_test = bool(row.get("is_test_session", False))

        by_status[status] = by_status.get(status, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
        by_session[session] = by_session.get(session, 0) + 1

        key = (source, session, is_test)
        if key not in session_map:
            session_map[key] = {s: 0 for s in _STATUS_ORDER}
        session_map[key][status] = session_map[key].get(status, 0) + 1

    breakdown = [
        {
            "source": src,
            "session": sess,
            "is_test_session": is_test,
            "total": sum(status_counts.values()),
            "by_status": status_counts,
        }
        for (src, sess, is_test), status_counts in sorted(
            session_map.items(), key=lambda entry: (entry[0][0], entry[0][1])
        )
    ]

    return StatusCounts(
        total=len(rows),
        by_status=by_status,
        by_source=by_source,
        by_session=by_session,
        breakdown=breakdown,
    )


def compute_class_distribution(rows: list[dict[str, Any]], cfg: FlurConfig) -> ClassDistribution:
    """Aggregate class pixel frequencies per source and overall from manifest counts."""
    by_source: dict[str, dict[str, int]] = {}
    total: dict[str, int] = {}

    for row in rows:
        source = row.get("source") or "unknown"
        counts = row.get("class_pixel_counts")
        if not counts or not isinstance(counts, dict):
            continue
        if source not in by_source:
            by_source[source] = {}
        for class_id_str, count in counts.items():
            count_int = int(count)
            by_source[source][class_id_str] = by_source[source].get(class_id_str, 0) + count_int
            total[class_id_str] = total.get(class_id_str, 0) + count_int

    percentages_by_source: dict[str, dict[str, float]] = {}
    for source, class_map in by_source.items():
        source_total = sum(class_map.values())
        percentages_by_source[source] = {
            cid: (cnt / source_total if source_total > 0 else 0.0)
            for cid, cnt in class_map.items()
        }

    grand_total = sum(total.values())
    percentages_total = {
        cid: (cnt / grand_total if grand_total > 0 else 0.0)
        for cid, cnt in total.items()
    }

    return ClassDistribution(
        by_source=by_source,
        total=total,
        percentages_by_source=percentages_by_source,
        percentages_total=percentages_total,
    )


def compute_spot_check_error_rate(
    rows: list[dict[str, Any]],
    limit: float,
) -> SpotCheckErrorRate:
    """Compute per-class and overall error rates from reviewed spot checks."""
    spot_checks = [row for row in rows if row.get("status") == "reviewed" and row.get("spot_check")]

    per_class_changed: dict[str, int] = {str(cid): 0 for cid in range(11)}
    per_class_labelled: dict[str, int] = {str(cid): 0 for cid in range(11)}

    for row in spot_checks:
        changed = row.get("review_changed_pixels") or {}
        labelled = row.get("class_pixel_counts") or {}
        for cid in range(11):
            cid_str = str(cid)
            per_class_changed[cid_str] += int(changed.get(cid_str, 0))
            per_class_labelled[cid_str] += int(labelled.get(cid_str, 0))

    total_changed = sum(per_class_changed.values())
    total_labelled = sum(per_class_labelled.values())
    total_error_rate = total_changed / total_labelled if total_labelled > 0 else 0.0

    per_class_error_rate: dict[str, float] = {}
    for cid_str, labelled_count in per_class_labelled.items():
        changed_count = per_class_changed[cid_str]
        per_class_error_rate[cid_str] = (
            changed_count / labelled_count if labelled_count > 0 else 0.0
        )

    classes_above_limit = [
        cid
        for cid, rate in per_class_error_rate.items()
        if rate > limit and per_class_labelled[cid] > 0
    ]
    warning_flag = (total_error_rate > limit) or (len(classes_above_limit) > 0)

    return SpotCheckErrorRate(
        total_changed_pixels=total_changed,
        total_labelled_pixels=total_labelled,
        total_error_rate=total_error_rate,
        per_class_changed_pixels=per_class_changed,
        per_class_labelled_pixels=per_class_labelled,
        per_class_error_rate=per_class_error_rate,
        limit=limit,
        warning=warning_flag,
        warning_flag=warning_flag,
        classes_above_limit=classes_above_limit,
    )


def compute_precision_metrics(
    paths: RepoPaths,
    rows: list[dict[str, Any]],
    cfg: FlurConfig,
    *,
    num_bins: int = 20,
    target_precision: float = 0.9,
) -> PrecisionMetrics:
    """Calculate prompt and class precision and suggest score presence thresholds."""
    prompt_list = cfg.prompt_list()
    num_prompts = len(prompt_list)
    prompt_won = np.zeros(num_prompts, dtype=np.int64)
    prompt_correct = np.zeros(num_prompts, dtype=np.int64)

    bin_size = 1.0 / num_bins
    trainable_classes = [c for c in cfg.classes.classes if c.trainable]
    bin_totals: dict[str, np.ndarray] = {
        c.name: np.zeros(num_bins, dtype=np.int64) for c in trainable_classes
    }
    bin_corrects: dict[str, np.ndarray] = {
        c.name: np.zeros(num_bins, dtype=np.int64) for c in trainable_classes
    }

    reviewed_rows = [row for row in rows if row.get("status") == "reviewed"]

    for row in reviewed_rows:
        image_id = row["id"]
        mask_path = paths.pool_masks_dir / f"{image_id}.png"

        work_dir = Path(row["work_dir"])
        if not work_dir.is_absolute():
            work_dir = paths.root / work_dir

        if not mask_path.is_file():
            alt_candidates = [
                work_dir / "draft_mask.png",
                work_dir / "reviewed_mask.png",
                work_dir / "final_mask.png",
            ]
            for candidate in alt_candidates:
                if candidate.is_file():
                    mask_path = candidate
                    break

        if not mask_path.is_file():
            continue

        reviewed_mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if reviewed_mask is None:
            continue

        winner_path = work_dir / "prompt_winner.png"
        if winner_path.is_file():
            prompt_winner = cv2.imread(str(winner_path), cv2.IMREAD_UNCHANGED)
            if prompt_winner is not None and prompt_winner.shape == reviewed_mask.shape:
                for p_idx, (_, class_id) in enumerate(prompt_list):
                    won_mask = prompt_winner == p_idx
                    won_count = int(won_mask.sum())
                    if won_count > 0:
                        correct_count = int((won_mask & (reviewed_mask == class_id)).sum())
                        prompt_won[p_idx] += won_count
                        prompt_correct[p_idx] += correct_count

        scores_path = work_dir / "scores.npz"
        if scores_path.is_file():
            try:
                store = ArtifactStore(work_dir)
                scores, _ = store.read_scores()
                valid_mask = reviewed_mask != 255
                height, width = reviewed_mask.shape[:2]

                for class_def in trainable_classes:
                    if class_def.name not in scores:
                        continue
                    class_score = scores[class_def.name]
                    if class_score.shape != (height, width):
                        class_score = cv2.resize(
                            class_score, (width, height), interpolation=cv2.INTER_LINEAR
                        )

                    bin_indices = np.clip(
                        (class_score / bin_size).astype(np.int32), 0, num_bins - 1
                    )
                    is_true = (reviewed_mask == class_def.id) & valid_mask

                    for b_idx in range(num_bins):
                        in_bin = (bin_indices == b_idx) & valid_mask
                        in_bin_count = int(in_bin.sum())
                        if in_bin_count > 0:
                            bin_totals[class_def.name][b_idx] += in_bin_count
                            bin_corrects[class_def.name][b_idx] += int(
                                (in_bin & is_true).sum()
                            )
            except (OSError, ValueError, KeyError, cv2.error) as error:
                logger.warning("Failed to evaluate scores for %s: %s", work_dir, error)

    per_prompt_items: list[PromptPrecisionItem] = []
    prompt_precision_dict: dict[str, float] = {}
    for p_idx, (prompt_text, cid) in enumerate(prompt_list):
        won = int(prompt_won[p_idx])
        correct = int(prompt_correct[p_idx])
        precision = correct / won if won > 0 else 0.0
        c_def = cfg.class_by_id(cid)
        per_prompt_items.append(
            PromptPrecisionItem(
                index=p_idx,
                prompt=prompt_text,
                class_id=cid,
                class_name=c_def.name,
                won_pixels=won,
                correct_pixels=correct,
                precision=precision,
            )
        )
        prompt_precision_dict[prompt_text] = precision

    suggested_thresholds: dict[str, float | None] = {}
    for class_def in trainable_classes:
        c_name = class_def.name
        totals = bin_totals.get(c_name, np.zeros(num_bins, dtype=np.int64))
        corrects = bin_corrects.get(c_name, np.zeros(num_bins, dtype=np.int64))
        found_threshold: float | None = None
        for b_idx in range(num_bins):
            total_in_bin = totals[b_idx]
            if total_in_bin > 0:
                bin_precision = corrects[b_idx] / total_in_bin
                if bin_precision >= target_precision:
                    found_threshold = round(b_idx * bin_size, 2)
                    break
        suggested_thresholds[c_name] = found_threshold

    per_class_items: list[ClassPrecisionItem] = []
    class_precision_dict: dict[str, float] = {}
    for class_def in trainable_classes:
        prompts_for_class = [
            item for item in per_prompt_items if item.class_id == class_def.id
        ]
        class_won = sum(item.won_pixels for item in prompts_for_class)
        class_correct = sum(item.correct_pixels for item in prompts_for_class)
        class_precision = class_correct / class_won if class_won > 0 else 0.0

        per_class_items.append(
            ClassPrecisionItem(
                class_id=class_def.id,
                class_name=class_def.name,
                display_color=class_def.display_color,
                won_pixels=class_won,
                correct_pixels=class_correct,
                precision=class_precision,
                suggested_threshold=suggested_thresholds.get(class_def.name),
            )
        )
        class_precision_dict[class_def.name] = class_precision

    return PrecisionMetrics(
        per_prompt=per_prompt_items,
        per_class=per_class_items,
        prompt_precision=prompt_precision_dict,
        class_precision=class_precision_dict,
        suggested_thresholds=suggested_thresholds,
    )


def compute_timing_metrics(rows: list[dict[str, Any]]) -> TimingMetrics:
    """Calculate mean and sum of prelabeling and manual review elapsed times."""
    proc_times = [
        float(row["processing_seconds"])
        for row in rows
        if row.get("processing_seconds") is not None
    ]
    rev_times = [
        float(row["review_seconds"])
        for row in rows
        if row.get("review_seconds") is not None and float(row["review_seconds"]) > 0
    ]

    total_proc = sum(proc_times)
    total_rev = sum(rev_times)

    return TimingMetrics(
        total_images_processed=len(proc_times),
        total_images_reviewed=len(rev_times),
        avg_processing_seconds=total_proc / len(proc_times) if proc_times else 0.0,
        avg_review_seconds=total_rev / len(rev_times) if rev_times else 0.0,
        total_processing_seconds=total_proc,
        total_review_seconds=total_rev,
    )


def compute_metrics(paths: RepoPaths, cfg: FlurConfig) -> ReportMetrics:
    """Compile the full quality report metrics from the manifest database and artifacts."""
    rows = _collect_rows(paths)
    status_counts = compute_status_counts(rows)
    class_distribution = compute_class_distribution(rows, cfg)
    spot_check = compute_spot_check_error_rate(
        rows, cfg.pipeline.report.spot_check_error_limit
    )
    precision = compute_precision_metrics(paths, rows, cfg)
    timing = compute_timing_metrics(rows)

    return ReportMetrics(
        generated_at=datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC"),
        status_counts=status_counts,
        class_distribution=class_distribution,
        spot_check=spot_check,
        precision=precision,
        timing=timing,
        warning_flag=spot_check.warning_flag,
        spot_check_error_limit=cfg.pipeline.report.spot_check_error_limit,
    )
