"""Candidate ingestion and guarded original archiving."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from flurseg.artifacts import ArtifactStore
from flurseg.ingest.anonymize import anonymize
from flurseg.ingest.discover import Candidate
from flurseg.ingest.gsd import resolve_gsd
from flurseg.ingest.ortho import OrthoTile, read_ortho_tiles
from flurseg.ingest.session import load_session
from flurseg.io.hashing import sha256_file
from flurseg.io.images import load_image, read_image_metadata
from flurseg.manifest.repo import ImageRecord, ManifestRepository
from flurseg.paths import RepoPaths, is_inside


def _contained(path: Path, root: Path) -> bool:
    return is_inside(path, root)


def _tile_hash(parent_hash: str, tile: OrthoTile) -> str:
    window = tile.window
    data = f"{parent_hash}:{int(window.col_off)}:{int(window.row_off)}:{int(window.width)}:{int(window.height)}".encode()
    return hashlib.sha256(data).hexdigest()


def _insert_image(
    image: Any, candidate: Candidate, paths: RepoPaths, repository: ManifestRepository, session: Any,
    *, parent_hash: str | None, tile_index: int | None, gsd_m: float | None, gsd_method: str,
    content_hash: str, metadata: dict[str, object] | None = None, quality: int = 95,
) -> ImageRecord:
    sequence = repository.next_sequence(candidate.source, candidate.session)
    image_id = f"{candidate.source}_{candidate.session}_{sequence:04d}"
    work_dir = paths.work_dir / image_id
    artifact = ArtifactStore(work_dir)
    artifact.write_image(image, quality)
    if metadata:
        artifact.write_meta(metadata)
    height, width = image.shape[:2]
    return repository.insert(
        ImageRecord(
            id=image_id, hash=content_hash, parent_hash=parent_hash, tile_index=tile_index,
            source=candidate.source, session=candidate.session, camera=session.camera,
            gsd_m=gsd_m, gsd_method=gsd_method, original_path=str(candidate.path),
            image_path=str(work_dir / "image.jpg"), work_dir=str(work_dir), width=width, height=height,
            is_test_session=session.test,
        )
    )


def ingest_candidate(
    candidate: Candidate, paths: RepoPaths, repository: ManifestRepository, config: Any, backend: Any | None = None
) -> list[ImageRecord]:
    """Normalize a candidate, deduplicate it, and create one or more ingested records."""
    if not is_inside(candidate.path, paths.root):
        raise ValueError("candidate path must be inside the configured repository root")
    parent_hash = sha256_file(candidate.path)
    if candidate.source == "ortho":
        if repository.connection.execute("SELECT 1 FROM images WHERE parent_hash = ? LIMIT 1", (parent_hash,)).fetchone():
            return []
    elif repository.find_by_hash(parent_hash) is not None:
        return []
    session = load_session(candidate.path.parent / "session.yaml")
    quality = int(getattr(getattr(config, "pipeline", config).ingest, "jpeg_quality", 95))
    if candidate.source == "ortho":
        ortho_config = getattr(config, "pipeline", config).ortho
        records: list[ImageRecord] = []
        for tile in read_ortho_tiles(
            candidate.path, ortho_config.tile_size, ortho_config.max_nodata_fraction, ortho_config.black_threshold
        ):
            records.append(
                _insert_image(
                    tile.image, candidate, paths, repository, session, parent_hash=parent_hash, tile_index=tile.index,
                    gsd_m=ortho_config.gsd_m, gsd_method="fixed_ortho", content_hash=_tile_hash(parent_hash, tile),
                    metadata={"window": list(tile.window.flatten()), "transform": list(tile.transform), "crs_wkt": tile.crs_wkt},
                    quality=quality,
                )
            )
        return records
    image = load_image(candidate.path)
    pipeline = getattr(config, "pipeline", config)
    if candidate.source in pipeline.anonymization.enabled_for and backend is not None:
        image, blurred_fraction = anonymize(image, backend, pipeline.anonymization.prompts, pipeline.tiling, pipeline.anonymization)
    else:
        blurred_fraction = 0.0
    metadata = read_image_metadata(candidate.path)
    gsd = resolve_gsd(candidate.source, session, metadata, image.shape[1], ortho_gsd_m=pipeline.ortho.gsd_m)
    return [
        _insert_image(
            image, candidate, paths, repository, session, parent_hash=None, tile_index=None, gsd_m=gsd.gsd_m,
            gsd_method=gsd.method, content_hash=parent_hash, metadata={"blurred_fraction": blurred_fraction}, quality=quality,
        )
    ]


def archive_original(path: Path, paths: RepoPaths, source: str, session: str) -> Path:
    """Move an ingested original into its guarded archive directory without overwriting files."""
    if not is_inside(path, paths.root):
        raise ValueError("original path must be inside the configured repository root")
    destination_dir = paths.archive_dir / source / session
    if not is_inside(destination_dir, paths.root):
        raise ValueError("archive destination must be inside the configured repository root")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name
    counter = 1
    while destination.exists():
        destination = destination_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    os.replace(path, destination)
    return destination
