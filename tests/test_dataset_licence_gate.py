"""Dataset-build coverage for locked third-party data licence exclusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from flurseg.config import load_config
from flurseg.dataset.build import build_dataset
from flurseg.io.atomic import atomic_save_jpg, atomic_save_png
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths


def _add_pool_item(paths: RepoPaths, image_id: str, *, parent_hash: str | None) -> None:
    """Create one reviewed pool item with an optional source-image hash."""
    paths.pool_images_dir.mkdir(parents=True, exist_ok=True)
    paths.pool_masks_dir.mkdir(parents=True, exist_ok=True)
    image_path = paths.pool_images_dir / f"{image_id}.jpg"
    atomic_save_jpg(image_path, np.full((1024, 1024, 3), 100, dtype=np.uint8))
    atomic_save_png(paths.pool_masks_dir / f"{image_id}.png", np.ones((1024, 1024), dtype=np.uint8))
    with Database(paths.database) as connection:
        ManifestRepository(connection).insert(
            ImageRecord(
                id=image_id,
                hash=f"tile-{image_id}",
                parent_hash=parent_hash,
                source="phone",
                session="licence_test",
                gsd_method="fixed",
                original_path=str(image_path),
                image_path=str(image_path),
                work_dir=str(paths.root / "work" / image_id),
                width=1024,
                height=1024,
                status="reviewed",
            )
        )


def test_build_excludes_only_locked_incompatible_licences(tmp_root: Path) -> None:
    """BY-SA lock records are excluded while CC0 and untracked own images remain."""
    paths = RepoPaths.from_root(tmp_root)
    _add_pool_item(paths, "phone_licence_test_0001", parent_hash="by-sa-hash")
    _add_pool_item(paths, "phone_licence_test_0002", parent_hash="cc0-hash")
    _add_pool_item(paths, "phone_licence_test_0003", parent_hash=None)
    (tmp_root / "config" / "samples.lock.yaml").write_text(
        yaml.safe_dump({"entries": [
            {"sha256": "by-sa-hash", "licence": "CC BY-SA 3.0"},
            {"sha256": "cc0-hash", "licence": "CC0-1.0"},
        ]}),
        encoding="utf-8",
    )

    dataset = build_dataset(paths, load_config(paths))

    assert dataset["tile_counts"]["total"] == 2
    assert dataset["excluded_licence"] == [
        {"id": "phone_licence_test_0001", "hash": "by-sa-hash", "licence": "CC BY-SA 3.0"}
    ]


def test_build_allows_missing_optional_lockfiles(tmp_root: Path) -> None:
    """No lockfiles is valid for owner-collected imagery."""
    paths = RepoPaths.from_root(tmp_root)
    _add_pool_item(paths, "phone_licence_test_0001", parent_hash=None)

    dataset = build_dataset(paths, load_config(paths))

    assert dataset["tile_counts"]["total"] == 1
    assert dataset["excluded_licence"] == []


def test_build_includes_swisstopo_ogd_licence(tmp_root: Path) -> None:
    """swisstopo OGD lock records are included by the dataset-build licence gate."""
    paths = RepoPaths.from_root(tmp_root)
    _add_pool_item(paths, "ortho_swiss_0001", parent_hash="swisstopo-hash")
    (tmp_root / "config" / "samples.lock.yaml").write_text(
        yaml.safe_dump({
            "entries": [
                {"sha256": "swisstopo-hash", "licence": "swisstopo OGD, free use with attribution"},
            ]
        }),
        encoding="utf-8",
    )

    dataset = build_dataset(paths, load_config(paths))

    assert dataset["tile_counts"]["total"] == 1
    assert dataset["excluded_licence"] == []
