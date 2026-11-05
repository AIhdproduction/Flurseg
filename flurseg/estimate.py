"""Prelabel runtime estimates, calibration history, and deterministic candidate ordering."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import rasterio
from PIL import Image
from pillow_heif import register_heif_opener

from flurseg.ingest.discover import Candidate
from flurseg.io.atomic import atomic_write_json
from flurseg.segment.tiling import iter_tiles

DEFAULT_SECONDS_PER_VARIANT = 2.9
DEFAULT_SECONDS_PER_MP = 1.5
_TIMING_LIMIT = 200

register_heif_opener()


@dataclass(slots=True)
class SourceEstimate:
    """Estimate totals for one input source."""

    files: int = 0
    megapixels: float = 0.0
    variants: int = 0
    seconds: float = 0.0


@dataclass(slots=True)
class Estimate:
    """Aggregate prelabel work estimate."""

    sources: dict[str, SourceEstimate] = field(default_factory=dict)
    files: int = 0
    megapixels: float = 0.0
    variants: int = 0
    seconds: float = 0.0
    unreadable: int = 0


def _pipeline(config: Any) -> Any:
    return getattr(config, "pipeline", config)


def read_size(path: Path) -> tuple[int, int] | None:
    """Read only image dimensions, returning ``None`` for unreadable inputs."""
    try:
        if path.suffix.lower() in {".tif", ".tiff"}:
            with rasterio.open(path) as dataset:
                return dataset.width, dataset.height
        with Image.open(path) as image:
            return image.size
    except (OSError, ValueError, rasterio.errors.RasterioError):
        return None


def count_variants(width: int, height: int, pipeline_cfg: Any) -> int:
    """Count SAM passes using the runner's tile placement at every configured scale."""
    pipeline = _pipeline(pipeline_cfg)
    variants = 0
    for assignment in pipeline.scales:
        factor = float(assignment.factor)
        scaled_width = max(1, round(width * factor))
        scaled_height = max(1, round(height * factor))
        variants += len(list(iter_tiles(scaled_width, scaled_height, pipeline.tiling.tile_size, pipeline.tiling.min_overlap)))
    return variants * len(pipeline.tta.variants)


def timing_fingerprint(pipeline_cfg: Any) -> str:
    """Return the settings identity used to keep calibration data comparable."""
    pipeline = _pipeline(pipeline_cfg)
    values = {
        "tta": list(pipeline.tta.variants),
        "tile_size": pipeline.tiling.tile_size,
        "min_overlap": pipeline.tiling.min_overlap,
        "scales": [float(item.factor) for item in pipeline.scales],
        "imgsz": pipeline.sam3.imgsz,
        "precision": pipeline.sam3.precision,
    }
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def load_timing_entries(path: Path) -> list[dict[str, Any]]:
    """Load the optional local calibration file without making estimates fragile."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [item for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []


def append_timing_entry(path: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Append a sample, retain the newest 200, and atomically persist the history."""
    entries = [*load_timing_entries(path), entry][-_TIMING_LIMIT:]
    atomic_write_json(path, entries)
    return entries


def fit_cost_model(entries: Iterable[dict[str, Any]], fingerprint: str | None = None) -> tuple[float, float]:
    """Fit seconds per variant and megapixel for one settings fingerprint."""
    selected = [
        item
        for item in entries
        if fingerprint is None or item.get("fingerprint") == fingerprint
    ]
    rows: list[tuple[float, float, float]] = []
    for item in selected:
        try:
            rows.append((float(item["variants"]), float(item["megapixels"]), float(item["seconds"])))
        except (KeyError, TypeError, ValueError):
            continue
    if len(rows) < 8:
        return DEFAULT_SECONDS_PER_VARIANT, DEFAULT_SECONDS_PER_MP
    sum_xx = sum(variants * variants for variants, _, _ in rows)
    sum_xy = sum(variants * megapixels for variants, megapixels, _ in rows)
    sum_yy = sum(megapixels * megapixels for _, megapixels, _ in rows)
    sum_xz = sum(variants * seconds for variants, _, seconds in rows)
    sum_yz = sum(megapixels * seconds for _, megapixels, seconds in rows)
    determinant = sum_xx * sum_yy - sum_xy * sum_xy
    if determinant <= 1e-9:
        return DEFAULT_SECONDS_PER_VARIANT, DEFAULT_SECONDS_PER_MP
    per_variant = (sum_xz * sum_yy - sum_yz * sum_xy) / determinant
    per_mp = (sum_yz * sum_xx - sum_xz * sum_xy) / determinant
    if per_variant < 0 or per_mp < 0:
        return DEFAULT_SECONDS_PER_VARIANT, DEFAULT_SECONDS_PER_MP
    return per_variant, per_mp


def estimate(candidates: Iterable[Candidate], pipeline_cfg: Any, timing_entries: Iterable[dict[str, Any]]) -> Estimate:
    """Estimate all readable candidates; duplicates remain intentionally included as an upper bound."""
    coefficient_variant, coefficient_mp = fit_cost_model(timing_entries, timing_fingerprint(pipeline_cfg))
    result = Estimate()
    for candidate in candidates:
        size = read_size(candidate.path)
        if size is None:
            result.unreadable += 1
            continue
        width, height = size
        megapixels = width * height / 1_000_000
        variants = count_variants(width, height, pipeline_cfg)
        seconds = coefficient_variant * variants + coefficient_mp * megapixels
        source = result.sources.setdefault(candidate.source, SourceEstimate())
        source.files += 1
        source.megapixels += megapixels
        source.variants += variants
        source.seconds += seconds
        result.files += 1
        result.megapixels += megapixels
        result.variants += variants
        result.seconds += seconds
    return result


def format_duration(seconds: float) -> str:
    """Format a non-negative duration without noisy sub-minute precision."""
    total = max(0, round(seconds))
    if total < 60:
        return f"{total} s"
    minutes = total // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 48:
        return f"{hours} h {minutes:02d} min"
    days, hours = divmod(hours, 24)
    return f"{days} d {hours:02d} h"


def render_estimate(value: Estimate, now: datetime) -> str:
    """Render a concise preflight estimate with its conservative assumptions."""
    lines = ["Prelabel estimate", "Source   Files       MP  Est. time"]
    for source, item in sorted(value.sources.items()):
        lines.append(f"{source:<8} {item.files:>5} {item.megapixels:>8.1f}  {format_duration(item.seconds)}")
    days = value.seconds / 86400
    lines.append(f"Total: {value.files} files, {value.megapixels:.1f} MP, ≈ {format_duration(value.seconds)} (≈ {days:.1f} days)")
    finish = now + timedelta(seconds=max(0, value.seconds))
    lines.append(f"Finish about {finish:%a %d.%m. %H:%M} if started now")
    lines.append("Upper bound: duplicates are skipped quickly.")
    if value.unreadable:
        lines.append(f"Unreadable headers: {value.unreadable}; their processing time is not included.")
    if value.seconds > 12 * 3600:
        lines.append("Tip: use --max-hours H or --limit N; processed originals move to archive/, so you can stop and resume any time.")
    return "\n".join(lines)


def _bit_reversal_order(length: int) -> list[int]:
    """Return a low-discrepancy bit-reversal permutation for one session."""
    if length < 2:
        return list(range(length))
    bits = (length - 1).bit_length()
    ordered = sorted(range(length), key=lambda index: (f"{index:0{bits}b}"[::-1], index))
    return ordered


def order_candidates(candidates: Iterable[Candidate], mode: str = "spread") -> list[Candidate]:
    """Keep session order or spread early work evenly across sessions."""
    values = list(candidates)
    if mode == "session":
        return values
    if mode != "spread":
        raise ValueError("mode must be 'spread' or 'session'")
    grouped: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for candidate in values:
        grouped[(candidate.source, candidate.session)].append(candidate)
    sessions = [grouped[key] for key in sorted(grouped)]
    ordered_sessions = [[items[index] for index in _bit_reversal_order(len(items))] for items in sessions]
    result: list[Candidate] = []
    for index in range(max((len(items) for items in ordered_sessions), default=0)):
        result.extend(items[index] for items in ordered_sessions if index < len(items))
    return result
