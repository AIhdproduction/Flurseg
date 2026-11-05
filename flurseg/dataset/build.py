"""Dataset construction from pool images, session splitting, tiling, and manifest export."""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from flurseg.config import FlurConfig, load_config
from flurseg.dataset.split import (
    group_images_by_session,
    load_split_registry,
    save_split_registry,
    split_sessions,
    update_split_registry,
)
from flurseg.dataset.tiles import extract_tiles
from flurseg.dataset.validate import MaskValidationError, validate_pool
from flurseg.io.atomic import atomic_save_jpg, atomic_save_png, atomic_write_json
from flurseg.io.images import load_image
from flurseg.paths import RepoPaths, ensure_inside
from flurseg.samples.extra.licences import is_acceptable_licence

logger = logging.getLogger(__name__)

_DEFAULT_CLASSES: list[dict[str, Any]] = [
    {"id": 0, "name": "other", "color": "#9E9E9E", "colour": "#9E9E9E", "cyclone_code": 64, "trainable": True},
    {"id": 1, "name": "ground", "color": "#A0522D", "colour": "#A0522D", "cyclone_code": 2, "trainable": True},
    {"id": 2, "name": "paved_surface", "color": "#5C6BC0", "colour": "#5C6BC0", "cyclone_code": 71, "trainable": True},
    {"id": 3, "name": "low_vegetation", "color": "#9CCC65", "colour": "#9CCC65", "cyclone_code": 3, "trainable": True},
    {"id": 4, "name": "tree", "color": "#1B5E20", "colour": "#1B5E20", "cyclone_code": 5, "trainable": True},
    {"id": 5, "name": "building", "color": "#E53935", "colour": "#E53935", "cyclone_code": 6, "trainable": True},
    {"id": 6, "name": "vehicle", "color": "#FFEB3B", "colour": "#FFEB3B", "cyclone_code": 91, "trainable": True},
    {"id": 7, "name": "construction", "color": "#FF9800", "colour": "#FF9800", "cyclone_code": 151, "trainable": True},
    {"id": 8, "name": "curb", "color": "#00BCD4", "colour": "#00BCD4", "cyclone_code": 74, "trainable": True},
    {"id": 9, "name": "road_marking", "color": "#FFFFFF", "colour": "#FFFFFF", "cyclone_code": 82, "trainable": True},
    {"id": 10, "name": "utility_cover", "color": "#E040FB", "colour": "#E040FB", "cyclone_code": 75, "trainable": True},
    {"id": 255, "name": "ignore", "color": "#000000", "colour": "#000000", "cyclone_code": "not exported", "trainable": False},
]


def _guard_dataset_dir(paths: Any) -> tuple[Path, Path]:
    """Ensure dataset_dir matches the resolved <root>/train/dataset path and is inside root."""
    root_raw = getattr(paths, "root", None)
    if root_raw is None:
        raise ValueError("paths must have a root attribute")
    root = Path(root_raw).resolve()

    expected_dataset_dir = (root / "train" / "dataset").resolve()
    dataset_dir_raw = getattr(paths, "dataset_dir", expected_dataset_dir)
    dataset_dir = Path(dataset_dir_raw).resolve()

    ensure_inside(dataset_dir, root)
    if os.path.normcase(str(dataset_dir)) != os.path.normcase(str(expected_dataset_dir)):
        raise ValueError(f"dataset target {dataset_dir} must be inside {expected_dataset_dir}")

    return root, dataset_dir


def _normalize_classes(raw_classes: Any, paths: Any, root: Path) -> list[dict[str, Any]]:
    """Normalize class catalogue into consistent dictionaries."""
    if not raw_classes:
        classes_yaml = getattr(paths, "classes_config", root / "config" / "classes.yaml")
        if Path(classes_yaml).is_file():
            try:
                from flurseg.config import ClassesConfig, _read_yaml

                parsed = ClassesConfig.model_validate(_read_yaml(Path(classes_yaml)))
                raw_classes = parsed.classes
            except (OSError, ValueError, yaml.YAMLError) as error:
                logger.warning("Failed to load classes config from %s: %s", classes_yaml, error)
                raw_classes = None

    if not raw_classes:
        return _DEFAULT_CLASSES

    result: list[dict[str, Any]] = []
    for item in raw_classes:
        if isinstance(item, dict):
            cid = int(item["id"])
            name = str(item["name"])
            color = str(item.get("display_color") or item.get("color") or item.get("colour") or "#000000")
            cyclone = item.get("cyclone_code", "")
            trainable = bool(item.get("trainable", cid != 255))
        else:
            cid = int(item.id)
            name = str(item.name)
            color = str(getattr(item, "display_color", getattr(item, "color", "#000000")))
            cyclone = getattr(item, "cyclone_code", "")
            trainable = bool(getattr(item, "trainable", cid != 255))
        result.append({
            "id": cid,
            "name": name,
            "color": color,
            "colour": color,
            "cyclone_code": cyclone,
            "trainable": trainable,
        })
    return result


def _extract_config(cfg: Any, paths: Any, root: Path) -> tuple[int, float, Any, int, int, list[dict[str, Any]]]:
    """Extract tiling, splitting, ingest parameters and class catalogue."""
    if cfg is None:
        try:
            cfg = load_config(paths if isinstance(paths, RepoPaths) else RepoPaths.from_root(root))
        except (OSError, ValueError, yaml.YAMLError) as error:
            logger.warning("Failed to load configuration: %s", error)
            cfg = None

    pipeline = getattr(cfg, "pipeline", cfg)
    dataset_cfg = getattr(pipeline, "dataset", None)

    tile_size = int(getattr(dataset_cfg, "tile_size", 1024))
    max_ignore_fraction = float(getattr(dataset_cfg, "max_ignore_fraction", 0.9))
    split_ratios = getattr(dataset_cfg, "split", (0.70, 0.15, 0.15))
    seed = int(getattr(dataset_cfg, "seed", 42))

    ingest_cfg = getattr(pipeline, "ingest", None)
    jpeg_quality = int(getattr(ingest_cfg, "jpeg_quality", 95))

    classes_cfg = getattr(cfg, "classes", None)
    raw_classes = getattr(classes_cfg, "classes", classes_cfg) if classes_cfg is not None else None
    normalized_classes = _normalize_classes(raw_classes, paths, root)

    return tile_size, max_ignore_fraction, split_ratios, seed, jpeg_quality, normalized_classes


def _split_overrides(root: Path) -> dict[str, str]:
    """Read optional manual split overrides from the isolated training config."""
    config_path = root / "config" / "train.yaml"
    if not config_path.is_file():
        return {}
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    overrides = raw.get("dataset_split", {}).get("overrides", {})
    if not isinstance(overrides, dict) or any(value not in {"train", "val", "test"} for value in overrides.values()):
        raise ValueError("config/train.yaml dataset_split.overrides must map sessions to train, val, or test")
    return {str(key): str(value) for key, value in overrides.items()}


def _discover_pool_items(
    pool_images_dir: Path,
    pool_masks_dir: Path,
    db_path: Path | None,
) -> list[dict[str, Any]]:
    """Discover pool images, matching masks, and resolve source/session metadata."""
    image_extensions = {".jpg", ".jpeg", ".png"}
    if not pool_images_dir.exists():
        return []

    image_files = sorted(
        [p for p in pool_images_dir.iterdir() if p.is_file() and p.suffix.lower() in image_extensions]
    )
    if not image_files:
        return []

    if db_path is None:
        raise ValueError("Manifest database path is required to discover pool items")

    records_by_id: dict[str, dict[str, Any]] = {}
    try:
        db_file = Path(db_path)
        if not db_file.is_file():
            raise FileNotFoundError(f"Manifest database file not found at '{db_path}'")

        from flurseg.manifest.db import Database
        from flurseg.manifest.repo import ManifestRepository

        with Database(db_file) as connection:
            repo = ManifestRepository(connection)
            for record in repo.query():
                records_by_id[record["id"]] = record
    except (sqlite3.Error, OSError) as error:
        raise RuntimeError(f"Failed to open or query manifest database at '{db_path}': {error}") from error

    missing = [img_path.name for img_path in image_files if img_path.stem not in records_by_id]
    if missing:
        raise ValueError(
            f"Cannot build dataset: {len(missing)} pool image(s) missing manifest record: {', '.join(missing)}"
        )

    items: list[dict[str, Any]] = []
    for img_path in image_files:
        image_id = img_path.stem
        mask_path = pool_masks_dir / f"{image_id}.png"
        rec = records_by_id[image_id]
        source = str(rec.get("source", "default"))
        session = str(rec.get("session", "default"))
        is_test = bool(rec.get("is_test_session", False))

        items.append({
            "id": image_id,
            "image_id": image_id,
            "source": source,
            "session": session,
            "is_test_session": is_test,
            "image_path": img_path,
            "mask_path": mask_path,
            "hash": str(rec.get("hash") or ""),
            "parent_hash": str(rec.get("parent_hash") or ""),
        })

    return items


def _locked_licences(root: Path) -> dict[str, str]:
    """Load optional fetch lockfiles as a SHA-256 to licence lookup."""
    licences: dict[str, str] = {}
    for name in ("samples.lock.yaml", "extra_sources.lock.yaml"):
        path = root / "config" / name
        if not path.is_file():
            continue
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as error:
            logger.warning("Could not read licence lockfile %s: %s", path, error)
            continue
        for record in payload.get("entries", []) if isinstance(payload, dict) else []:
            if not isinstance(record, dict):
                continue
            digest = str(record.get("sha256") or "").strip().lower()
            licence = str(record.get("licence") or record.get("license") or "").strip()
            if digest and licence:
                licences[digest] = licence
    return licences


def _exclude_unacceptable_licences(
    items: list[dict[str, Any]], root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Exclude only pool images tied to a locked non-compatible licence."""
    locked = _locked_licences(root)
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    for item in items:
        digest = str(item.get("parent_hash") or item.get("hash") or "").lower()
        licence = locked.get(digest)
        if licence is not None and not is_acceptable_licence(licence):
            excluded.append({"id": str(item["id"]), "hash": digest, "licence": licence})
            logger.warning("excluded_licence id=%s licence=%s", item["id"], licence)
            continue
        included.append(item)
    return included, excluded


def build_dataset(
    paths: RepoPaths | Any,
    cfg: FlurConfig | Any | None = None,
) -> dict[str, Any]:
    """Validate pool masks, partition sessions into splits, extract tiles, and export manifest."""
    if isinstance(paths, (str, Path)):
        paths = RepoPaths.from_root(Path(paths))

    root, dataset_dir = _guard_dataset_dir(paths)
    pool_images_dir = Path(getattr(paths, "pool_images_dir", root / "train" / "pool" / "images")).resolve()
    pool_masks_dir = Path(getattr(paths, "pool_masks_dir", root / "train" / "pool" / "masks")).resolve()
    ensure_inside(pool_images_dir, root)
    ensure_inside(pool_masks_dir, root)

    # 1. Validate all pool masks first (abort with list of bad files, nothing deleted)
    bad_files = validate_pool(pool_images_dir, pool_masks_dir)
    if bad_files:
        error_lines = [f"{path}: {msg}" for path, msg in bad_files]
        raise MaskValidationError(
            f"Validation failed for {len(bad_files)} pool mask(s):\n" + "\n".join(error_lines)
        )

    tile_size, max_ignore_fraction, split_ratios, seed, jpeg_quality, normalized_classes = _extract_config(
        cfg, paths, root
    )

    db_path = getattr(paths, "database", root / "flurseg.db")
    items = _discover_pool_items(pool_images_dir, pool_masks_dir, db_path)
    items, excluded_licence = _exclude_unacceptable_licences(items, root)

    # 2. Keep prior split assignments in a registry outside the rebuilt dataset tree.
    registry_path = root / "train" / "split_registry.json"
    registry = load_split_registry(registry_path, seed)
    groups = group_images_by_session(items)
    split_res = split_sessions(
        items,
        seed=seed,
        split_ratios=split_ratios,
        registry=registry,
        overrides=_split_overrides(root),
    )
    registry = update_split_registry(registry, split_res, groups, seed)

    # 3. Rebuild train/dataset/ (guarded delete + recreate)
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        (dataset_dir / sp / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / sp / "masks").mkdir(parents=True, exist_ok=True)

    # 4. Extract and save tiles for each split
    tile_counts = {"train": 0, "val": 0, "test": 0}
    tile_counts_by_source: dict[str, int] = defaultdict(int)
    tile_counts_by_split_source: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }
    tile_counts_per_source: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "val": defaultdict(int),
        "test": defaultdict(int),
    }
    tile_metadata: dict[str, dict[str, str]] = {}

    pixel_counts: dict[str, dict[int, int]] = {
        "train": {c["id"]: 0 for c in normalized_classes},
        "val": {c["id"]: 0 for c in normalized_classes},
        "test": {c["id"]: 0 for c in normalized_classes},
        "overall": {c["id"]: 0 for c in normalized_classes},
    }

    for sp in ("train", "val", "test"):
        for item in split_res[sp]:
            img_path = Path(item["image_path"])
            mask_path = Path(item["mask_path"])
            try:
                image = load_image(img_path)
            except (OSError, ValueError) as error:
                raise RuntimeError(f"Failed to load dataset image '{img_path}': {error}") from error

            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                try:
                    with Image.open(mask_path) as pil_m:
                        mask = np.asarray(pil_m, dtype=np.uint8)
                except (OSError, ValueError) as error:
                    raise RuntimeError(f"Failed to load dataset mask '{mask_path}': {error}") from error

            if mask is None:
                raise RuntimeError(f"Failed to read dataset mask data from '{mask_path}'")

            tiles = extract_tiles(
                image,
                mask,
                tile_size=tile_size,
                max_ignore_fraction=max_ignore_fraction,
            )

            for tile in tiles:
                tile_name = f"{item['id']}_{tile.index:04d}"
                out_img = dataset_dir / sp / "images" / f"{tile_name}.jpg"
                out_mask = dataset_dir / sp / "masks" / f"{tile_name}.png"

                atomic_save_jpg(out_img, tile.image, quality=jpeg_quality)
                atomic_save_png(out_mask, tile.mask)
                tile_metadata[tile_name] = {
                    "source": str(item["source"]),
                    "session": str(item["session"]),
                    "split": sp,
                }

                tile_counts[sp] += 1
                source = item["source"]
                tile_counts_by_source[source] += 1
                tile_counts_by_split_source[sp][source] += 1
                tile_counts_per_source[sp][source] += 1

                uniques, counts = np.unique(tile.mask, return_counts=True)
                for u, c in zip(uniques, counts):
                    cid = int(u)
                    if cid in pixel_counts[sp]:
                        pixel_counts[sp][cid] += int(c)
                        pixel_counts["overall"][cid] += int(c)
                    else:
                        pixel_counts[sp][cid] = int(c)
                        pixel_counts["overall"][cid] = pixel_counts["overall"].get(cid, 0) + int(c)

    # 5. Compute class pixel frequencies
    frequencies: dict[str, dict[str, float]] = {}
    frequencies_by_id: dict[str, dict[str, float]] = {}
    counts_by_name: dict[str, dict[str, int]] = {}
    counts_by_id: dict[str, dict[str, int]] = {}

    for sp in ("train", "val", "test", "overall"):
        total_pixels = sum(pixel_counts[sp].values())
        frequencies[sp] = {}
        frequencies_by_id[sp] = {}
        counts_by_name[sp] = {}
        counts_by_id[sp] = {}
        for c in normalized_classes:
            cid = c["id"]
            cname = c["name"]
            cnt = pixel_counts[sp].get(cid, 0)
            freq = float(cnt / total_pixels) if total_pixels > 0 else 0.0
            frequencies[sp][cname] = freq
            frequencies_by_id[sp][str(cid)] = freq
            counts_by_name[sp][cname] = cnt
            counts_by_id[sp][str(cid)] = cnt

    tile_counts_per_source = {
        sp: dict(tile_counts_per_source[sp]) for sp in ("train", "val", "test")
    }
    tile_counts_by_split_source = tile_counts_per_source

    # 6. Assemble dataset.json
    dataset_info = {
        "version": "1.0.0",
        "dataset_version": registry["version"],
        "created_at": datetime.now(UTC).isoformat(),
        "tile_size": tile_size,
        "seed": seed,
        "val_empty": tile_counts["val"] == 0,
        "ignore_index": 255,
        "classes": normalized_classes,
        "split_per_session": split_res.session_to_split,
        "split_to_sessions": split_res.split_to_sessions,
        "source_session_to_split": {
            f"{k[0]}/{k[1]}": v for k, v in split_res.source_session_to_split.items()
        },
        "tile_counts": {
            "train": tile_counts["train"],
            "val": tile_counts["val"],
            "test": tile_counts["test"],
            "total": sum(tile_counts.values()),
            "by_split": dict(tile_counts),
            "by_source": dict(tile_counts_by_source),
            "by_split_source": {sp: dict(tile_counts_by_split_source[sp]) for sp in ("train", "val", "test")},
            "per_split": dict(tile_counts),
            "per_source": dict(tile_counts_per_source),
        },
        "tile_counts_per_split": dict(tile_counts),
        "tile_counts_per_source": dict(tile_counts_by_source),
        "tile_counts_by_source": dict(tile_counts_by_source),
        "tile_counts_by_split_source": dict(tile_counts_per_source),
        "class_pixel_frequencies": frequencies,
        "class_pixel_frequencies_by_id": frequencies_by_id,
        "class_pixel_counts": counts_by_name,
        "class_pixel_counts_by_id": counts_by_id,
        "source_image_ids_per_split": split_res.source_image_ids_per_split,
        "image_ids_per_split": split_res.source_image_ids_per_split,
        "tile_metadata": tile_metadata,
        "new_sessions": [
            {"source": source, "session": session, "split": split_res.source_session_to_split[(source, session)]}
            for source, session in split_res.new_sessions
        ],
        "excluded_licence": excluded_licence,
    }

    # 7. Write dataset.json atomically
    atomic_write_json(dataset_dir / "dataset.json", dataset_info)
    save_split_registry(registry_path, registry)

    if dataset_info["val_empty"]:
        print(
            "WARNING: validation split is empty; training will use a deterministic internal holdout."
        )

    additions = ", ".join(
        f"{source}/{session} -> {split_res.source_session_to_split[(source, session)]}"
        for source, session in split_res.new_sessions
    ) or "no new sessions"
    print(
        f"Dataset v{registry['version']}: +{len(split_res.new_sessions)} sessions ({additions}), "
        f"train {tile_counts['train']} tiles, val {tile_counts['val']} tiles, test {tile_counts['test']}"
    )
    if excluded_licence:
        print(f"excluded_licence: {len(excluded_licence)} image(s)")

    return dataset_info
