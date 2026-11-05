"""Planning, retrieval, lock handling, attribution, and session metadata."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import yaml

from flurseg.ingest.session import SessionMetadata
from flurseg.io.atomic import atomic_write_bytes
from flurseg.paths import RepoPaths, ensure_inside
from flurseg.samples.extra.amsterdam import products as amsterdam_products
from flurseg.samples.extra.arcgis import get_arcgis_window
from flurseg.samples.extra.archive_dataset import ArchiveDownloadSkipped
from flurseg.samples.extra.archive_dataset import plan_products as plan_archive_dataset
from flurseg.samples.extra.archive_dataset import products as archive_dataset_products
from flurseg.samples.extra.common import (
    WINDOW_PIXELS,
    FileProduct,
    ImageProduct,
    sampled_centres,
    save_file_product,
    save_product,
)
from flurseg.samples.extra.commons_categories import plan_products as plan_commons_categories
from flurseg.samples.extra.commons_categories import products as commons_category_products
from flurseg.samples.extra.commons_video_frames import plan_products as plan_commons_video_frames
from flurseg.samples.extra.commons_video_frames import products as commons_video_frame_products
from flurseg.samples.extra.config import (
    AmsterdamPanoramaEntry,
    ArcgisEntry,
    ArchiveDatasetEntry,
    CommonsCategoriesEntry,
    CommonsVideoFramesEntry,
    Entry,
    ExtraSourcesConfig,
    InternetArchiveEntry,
    PanoramaxEntry,
    StacSearchEntry,
    StacStaticEntry,
    VegbilderWfsEntry,
    WmsEntry,
    XyzEntry,
    load_extra_config,
)
from flurseg.samples.extra.internet_archive import plan_products as plan_internet_archive
from flurseg.samples.extra.internet_archive import products as internet_archive_products
from flurseg.samples.extra.panoramax import plan_products as plan_panoramax
from flurseg.samples.extra.panoramax import products as panoramax_products
from flurseg.samples.extra.stac_cog import search_products, static_products
from flurseg.samples.extra.vegbilder import hit_count as vegbilder_hit_count
from flurseg.samples.extra.vegbilder import products as vegbilder_products
from flurseg.samples.extra.wms import get_wms_window
from flurseg.samples.extra.xyz import get_xyz_window

logger = logging.getLogger(__name__)


def _config_path(paths: RepoPaths) -> Path:
    return paths.config_dir / "extra_sources.yaml"


def _lock_path(paths: RepoPaths) -> Path:
    return paths.config_dir / "extra_sources.lock.yaml"


def _load_lock(paths: RepoPaths) -> list[dict[str, object]]:
    path = _lock_path(paths)
    if not path.is_file():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = raw.get("entries", []) if isinstance(raw, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_lock(paths: RepoPaths, new_records: list[dict[str, object]]) -> None:
    """Merge records into the stable per-file extra-source lockfile."""
    merged: dict[tuple[str, str, str], dict[str, object]] = {}
    for record in _load_lock(paths) + new_records:
        key = (str(record.get("source")), str(record.get("session")), str(record.get("filename")))
        merged[key] = record
    payload = {
        "entries": sorted(
            merged.values(),
            key=lambda item: (str(item["source"]), str(item["session"]), str(item["filename"])),
        )
    }
    atomic_write_bytes(
        _lock_path(paths), yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")
    )


def _vegbilder_existing(paths: RepoPaths, entry: VegbilderWfsEntry, session_dir: Path) -> tuple[set[str], int]:
    """Return already stored Vegbilder feature IDs and their cap count without changing provenance files."""
    identifiers = {
        str(record.get("title") or "")
        for record in _load_lock(paths)
        if record.get("id") == entry.id and str(record.get("title") or "")
    }
    image_count = 0
    prefix = f"{entry.session}_"
    if session_dir.is_dir():
        for path in session_dir.glob("*.jpg"):
            image_count += 1
            remainder = path.stem.removeprefix(prefix)
            year, separator, feature_id = remainder.partition("_")
            if separator and year.isdigit() and feature_id:
                identifiers.add(feature_id)
    return identifiers, max(image_count, len(identifiers))


def _notes(product: ImageProduct, filename: str) -> str:
    metadata_lines = tuple(
        f"{key.replace('_', ' ').title()}: {value}"
        for key, value in (getattr(product, "metadata", None) or {}).items()
    )
    return "\n".join(
        (
            "Source: fetch-extra",
            f"Licence: {product.licence}",
            f"Attribution: {product.attribution}",
            "",
            "Files:",
            f"- {filename}: Licence: {product.licence}, Attribution: {product.attribution}, URL: {product.url}",
            *metadata_lines,
        )
    )


def _write_session(session_dir: Path, entry: Entry, product: ImageProduct, filename: str) -> None:
    path = session_dir / "session.yaml"
    if path.exists():
        return
    metadata: SessionMetadata = entry.session_yaml.metadata(
        gsd_m=product.gsd_m, notes=_notes(product, filename)
    )
    atomic_write_bytes(
        path,
        yaml.safe_dump(metadata.model_dump(exclude_none=True), sort_keys=False, allow_unicode=True).encode(
            "utf-8"
        ),
    )


def _plan_count(entry: Entry, max_per_entry: int | None = None) -> int:
    limit = min(entry.max_per_entry, max_per_entry) if max_per_entry is not None else entry.max_per_entry
    if isinstance(entry, (WmsEntry, XyzEntry, ArcgisEntry)):
        return min(limit, len(sampled_centres(entry.windows, entry.bbox, entry.count)))
    if isinstance(entry, StacSearchEntry):
        return min(limit, len(entry.bboxes) * entry.max_items * entry.windows_per_item)
    if isinstance(entry, PanoramaxEntry):
        try:
            return min(limit, plan_panoramax(entry, limit=limit))
        except TypeError:
            return min(limit, plan_panoramax(entry))
    if isinstance(entry, AmsterdamPanoramaEntry):
        return min(limit, entry.max_per_entry)
    if isinstance(entry, InternetArchiveEntry):
        return min(limit, plan_internet_archive(entry, limit=limit))
    if isinstance(entry, ArchiveDatasetEntry):
        return min(limit, plan_archive_dataset(entry, limit=limit))
    if isinstance(entry, CommonsCategoriesEntry):
        try:
            return min(limit, plan_commons_categories(entry, limit=limit))
        except TypeError:
            return min(limit, plan_commons_categories(entry))
    if isinstance(entry, CommonsVideoFramesEntry):
        try:
            return min(limit, plan_commons_video_frames(entry, limit=limit))
        except TypeError:
            return min(limit, plan_commons_video_frames(entry))
    if isinstance(entry, VegbilderWfsEntry):
        return min(
            limit,
            sum(min(entry.per_bbox, vegbilder_hit_count(entry, year, bbox)) for year in entry.layer_year for bbox in entry.bboxes),
        )
    return min(limit, len(entry.bboxes))


def _products(
    entry: Entry,
    licence_overrides: tuple[str, ...] = (),
    limit: int | None = None,
) -> Iterable[ImageProduct | None]:
    max_count = min(entry.max_per_entry, limit) if limit is not None else entry.max_per_entry
    if isinstance(entry, WmsEntry):
        return (
            get_wms_window(entry, lon, lat)
            for lon, lat in sampled_centres(entry.windows, entry.bbox, entry.count)[:max_count]
        )
    if isinstance(entry, XyzEntry):
        return (
            get_xyz_window(entry, lon, lat)
            for lon, lat in sampled_centres(entry.windows, entry.bbox, entry.count)[:max_count]
        )
    if isinstance(entry, ArcgisEntry):
        return (
            get_arcgis_window(entry, lon, lat)
            for lon, lat in sampled_centres(entry.windows, entry.bbox, entry.count)[:max_count]
        )
    if isinstance(entry, StacStaticEntry):
        return static_products(entry, licence_overrides, limit=max_count)
    if isinstance(entry, StacSearchEntry):
        return search_products(entry, licence_overrides, limit=max_count)
    if isinstance(entry, PanoramaxEntry):
        return panoramax_products(entry, limit=max_count)
    if isinstance(entry, AmsterdamPanoramaEntry):
        return amsterdam_products(entry, limit=max_count)
    if isinstance(entry, ArchiveDatasetEntry):
        return archive_dataset_products(entry, Path("work") / "cache" / "datasets", limit=max_count)
    if isinstance(entry, CommonsCategoriesEntry):
        return commons_category_products(entry, limit=max_count)
    if isinstance(entry, VegbilderWfsEntry):
        return vegbilder_products(entry, limit=max_count)
    raise TypeError(f"unsupported extra source type: {entry.type}")


def _write_attribution(
    paths: RepoPaths, config: ExtraSourcesConfig, lock_records: list[dict[str, object]]
) -> None:
    """Write a human-readable attribution inventory after a non-dry run."""
    counts: dict[str, int] = defaultdict(int)
    for record in lock_records:
        counts[str(record.get("id", "unknown"))] += 1
    lines = ["# Extra training-data attribution", "", "This file records downloaded third-party imagery.", ""]
    for entry in config.sources:
        lines.extend(
            (
                f"## {entry.id}",
                "",
                f"- Source: {entry.attribution}",
                f"- Licence: {entry.licence}",
                f"- Files: {counts.get(entry.id, 0)}",
                "",
            )
        )
    path = paths.reports_dir / "attribution" / "extra_sources.md"
    ensure_inside(path, paths.root)
    atomic_write_bytes(path, "\n".join(lines).encode("utf-8"))


def _failed_entry(entry: Entry, error: Exception) -> dict[str, object]:
    """Create one concise failure record without exposing a traceback to the batch run."""
    reason = str(error).replace("\n", " ")[:200] or type(error).__name__
    logger.warning("Failed to fetch extra source %s: %s", entry.id, reason)
    return {
        "id": entry.id,
        "type": entry.type,
        "source": entry.source,
        "session": entry.session,
        "status": "failed",
        "licence": entry.licence,
        "error": reason,
    }


def _skipped_entry(entry: Entry, error: Exception) -> dict[str, object]:
    """Record a policy-based source skip in the result table."""
    reason = str(error).replace("\n", " ")[:200] or type(error).__name__
    logger.warning("Skipped extra source %s: %s", entry.id, reason)
    return {
        "id": entry.id,
        "type": entry.type,
        "source": entry.source,
        "session": entry.session,
        "status": "skipped",
        "licence": entry.licence,
        "error": reason,
    }


def fetch_extra(
    paths: RepoPaths | Path | str | None = None,
    *,
    dry_run: bool = False,
    only_types: set[str] | None = None,
    only_ids: set[str] | None = None,
    only_source: str | None = None,
    max_per_entry: int | None = None,
) -> list[dict[str, object]]:
    """Fetch enabled configured sources, or return a side-effect-free download plan."""
    resolved = paths if isinstance(paths, RepoPaths) else RepoPaths.from_root(paths)
    config = load_extra_config(_config_path(resolved))
    selected = [
        entry
        for entry in config.sources
        if entry.enabled or (dry_run and only_types is not None and entry.type in only_types)
    ]
    if only_types:
        selected = [entry for entry in selected if entry.type in only_types]
    if only_ids:
        selected = [entry for entry in selected if entry.id in only_ids]
    if only_source:
        selected = [entry for entry in selected if entry.source == only_source]

    if dry_run:
        plan: list[dict[str, object]] = []
        for entry in selected:
            limit = (
                min(entry.max_per_entry, max_per_entry) if max_per_entry is not None else entry.max_per_entry
            )
            stats = None
            try:
                if (
                    isinstance(entry, PanoramaxEntry)
                    and getattr(plan_panoramax, "__module__", "") == "flurseg.samples.extra.panoramax"
                ):
                    from flurseg.samples.extra.panoramax import accepted_items_with_stats

                    accepted, stats = accepted_items_with_stats(entry, limit=limit, cache_dir=resolved.work_dir / "cache")
                    planned_files = sum(entry.views if getattr(entry, "views", 1) > 1 and "field_of_view" in str(item) else 1 for item, _ in accepted)
                else:
                    planned_files = _plan_count(entry, max_per_entry=limit)
                status, error = "dry_run", None
            except Exception as caught:  # noqa: BLE001 - unavailable public metadata must not abort a dry run.
                logger.warning("Failed to plan extra source %s: %s", entry.id, caught)
                planned_files, status, error = 0, "failed", str(caught)
            record: dict[str, object] = {
                "id": entry.id,
                "type": entry.type,
                "source": entry.source,
                "session": entry.session,
                "filename": f"{entry.session}_<idx>{'.jpg' if entry.source == 'phone' else '.tif'}",
                "planned_files": planned_files,
                "estimate_size": int(WINDOW_PIXELS * WINDOW_PIXELS * 3 * 0.35),
                "licence": entry.licence,
                "status": status,
            }
            if stats is not None:
                record.update(
                    {
                        "seen": stats["seen"],
                        "accepted": stats["accepted"],
                        "rejected_by_licence": stats["rejected_by_licence"],
                        "skipped_360": stats["skipped_360"],
                        "thinned": stats["skipped_thinned"],
                        "skipped_thinned": stats["skipped_thinned"],
                    }
                )
            if error:
                record["error"] = error
            plan.append(record)
        return plan

    records: list[dict[str, object]] = []
    written_locks: list[dict[str, object]] = []
    for entry in selected:
        session_dir = resolved.input_dir / entry.source / entry.session
        limit = min(entry.max_per_entry, max_per_entry) if max_per_entry is not None else entry.max_per_entry
        write_limit = limit
        try:
            if isinstance(entry, CommonsVideoFramesEntry):
                products = (
                    commons_video_frame_products(entry, resolved.work_dir / "cache" / "videos", limit=limit)
                    if max_per_entry is not None
                    else commons_video_frame_products(entry, resolved.work_dir / "cache" / "videos")
                )
            elif isinstance(entry, InternetArchiveEntry):
                products = internet_archive_products(entry, resolved.work_dir / "cache" / "videos", limit=limit)
            elif isinstance(entry, ArchiveDatasetEntry):
                products = archive_dataset_products(entry, resolved.work_dir / "cache" / "datasets", limit=limit)
            elif isinstance(entry, VegbilderWfsEntry):
                existing_ids, existing_count = _vegbilder_existing(resolved, entry, session_dir)
                # The cap includes durable files from an earlier run; the source only receives the new remainder.
                write_limit = max(0, limit - existing_count)
                products = (
                    vegbilder_products(entry, limit=write_limit, existing_feature_ids=existing_ids)
                    if write_limit
                    else ()
                )
            else:
                products = (
                    _products(entry, config.licence_overrides, limit=limit)
                    if max_per_entry is not None
                    else _products(entry, config.licence_overrides)
                )
            selected_products = list(islice(products, write_limit))
        except ArchiveDownloadSkipped as error:
            records.append(_skipped_entry(entry, error))
            continue
        except Exception as error:  # noqa: BLE001 - one external source must not abort the batch.
            records.append(_failed_entry(entry, error))
            continue
        for index, product in enumerate(selected_products):
            if product is None:
                records.append(
                    {
                        "id": entry.id,
                        "type": entry.type,
                        "source": entry.source,
                        "session": entry.session,
                        "status": "skipped",
                        "licence": entry.licence,
                        "error": "empty tile",
                    }
                )
                continue
            try:
                session_dir.mkdir(parents=True, exist_ok=True)
                ensure_inside(session_dir, resolved.input_dir)
                if isinstance(product, FileProduct):
                    path, digest, size, exists = save_file_product(session_dir, product)
                else:
                    path, digest, size, exists = save_product(
                        session_dir, f"{entry.session}_{index}", product
                    )
                ensure_inside(path, resolved.input_dir)
                _write_session(session_dir, entry, product, path.name)
            except Exception as error:  # noqa: BLE001 - each public source should fail independently.
                records.append(_failed_entry(entry, error))
                continue
            status = "skipped" if exists else "downloaded"
            record = {
                "id": entry.id,
                "type": entry.type,
                "source": entry.source,
                "session": entry.session,
                "filename": path.name,
                "path": path,
                "url": product.url,
                "title": product.title,
                "sha256": digest,
                "size": size,
                "licence": product.licence,
                "attribution": product.attribution,
                "metadata": getattr(product, "metadata", None),
                "status": status,
            }
            records.append(record)
            written_locks.append(
                {key: value for key, value in record.items() if key not in {"path", "status", "error"}}
            )
    if written_locks:
        _write_lock(resolved, written_locks)
    _write_attribution(resolved, config, _load_lock(resolved))
    return records
