"""End-to-end prelabel and score-only reprocess workflows."""

from __future__ import annotations

import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.segmentation import slic

from flurseg.artifacts import ArtifactStore
from flurseg.estimate import (
    append_timing_entry,
    count_variants,
    format_duration,
    order_candidates,
    read_size,
    timing_fingerprint,
)
from flurseg.fusion import fuse
from flurseg.ingest.discover import Candidate, discover
from flurseg.ingest.ingest import archive_original, ingest_candidate
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ManifestRepository
from flurseg.progress import ProgressReporter
from flurseg.refine.registry import RefinementContext, refine
from flurseg.routing import decide, materialize
from flurseg.segment.runner import score_image
from flurseg.uncertainty import calculate


@dataclass(slots=True)
class PipelineSummary:
    auto_accepted: int = 0
    needs_review: int = 0
    spot_checks: int = 0
    failed: int = 0
    skipped_duplicates: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def _factory(factory: Any, pipeline: Any) -> Any:
    try:
        return factory()
    except TypeError:
        return factory(pipeline.sam3)


def _ids(cfg: Any) -> dict[str, int]:
    return {definition.name: definition.id for definition in cfg.classes.classes}


def _counts(mask: np.ndarray) -> dict[str, int]:
    return {str(int(value)): int((mask == value).sum()) for value in np.unique(mask)}


def _derive_class_maps(
    final_mask: np.ndarray,
    per_class_winner: Mapping[str, np.ndarray],
    per_class_std: Mapping[str, np.ndarray],
    cfg: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Derive full-resolution prompt winner and TTA disagreement maps for the refined mask."""
    height, width = shape
    prompt_winner = np.full((height, width), 255, dtype=np.uint8)
    disagreement = np.zeros((height, width), dtype=np.uint8)
    prompt_indices: dict[str, list[int]] = {}
    for index, (_, class_id) in enumerate(cfg.prompt_list()):
        prompt_indices.setdefault(cfg.class_by_id(class_id).name, []).append(index)
    for definition in cfg.classes.classes:
        if definition.id == 255:
            continue
        selected = final_mask == definition.id
        if not selected.any():
            continue
        name = definition.name
        if name in per_class_winner:
            local = cv2.resize(per_class_winner[name], (width, height), interpolation=cv2.INTER_NEAREST)
            globals_for_class = np.asarray(prompt_indices.get(name, []), dtype=np.uint8)
            if len(globals_for_class) > 0:
                local_indices = np.clip(local[selected], 0, len(globals_for_class) - 1)
                prompt_winner[selected] = globals_for_class[local_indices]
        if name in per_class_std:
            expanded = cv2.resize(per_class_std[name], (width, height), interpolation=cv2.INTER_LINEAR)
            disagreement[selected] = np.clip(expanded[selected], 0, 255).astype(np.uint8)
    return prompt_winner, disagreement


def _compute_coarse_superpixels(image: np.ndarray, cfg: Any) -> np.ndarray:
    """Compute coarse SLIC superpixels downscaled by 0.5 (INTER_AREA) and upsampled to full resolution."""
    height, width = image.shape[:2]
    small_size = (max(1, round(width * 0.5)), max(1, round(height * 0.5)))
    downscaled = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA)
    count = max(1, round(height * width / 1_000_000 * cfg.pipeline.review.coarse_superpixels_per_mp))
    labels_small = slic(
        downscaled,
        n_segments=count,
        compactness=float(cfg.pipeline.refinement.edge_snap.compactness),
        start_label=1,
        channel_axis=-1,
    )
    try:
        labels = cv2.resize(labels_small.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    except cv2.error:
        labels = cv2.resize(labels_small.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    return labels.astype(np.int32, copy=False)


def _route_record(record: dict[str, Any], image: np.ndarray, mask: np.ndarray, artifact: ArtifactStore, cfg: Any, paths: Any, repository: ManifestRepository) -> Any:
    uncertainty = artifact.read_mask("uncertainty.png")
    if uncertainty is None:
        raise FileNotFoundError(artifact.directory / "uncertainty.png")
    decision = decide(mask, uncertainty.astype(np.float32) / 255, record["hash"], record["is_test_session"], cfg.pipeline)
    materialize(paths, record["id"], image, mask, uncertainty, decision)
    repository.reroute(
        record["id"], decision.status, routing_reason=decision.reason, uncertain_fraction=decision.uncertain_fraction,
        ignore_fraction=decision.ignore_fraction, spot_check=decision.spot_check, class_pixel_counts=_counts(mask),
    )
    return decision


def _write_derived_artifacts(
    artifact: ArtifactStore,
    image: np.ndarray,
    fused_mask: np.ndarray,
    refined: Any,
    coarse_superpixels: np.ndarray,
    prompt_winner: np.ndarray,
    disagreement: np.ndarray,
    uncertainty: Any,
    cfg: Any,
) -> None:
    """Write every artifact derived from fusion, refinement, and uncertainty results."""
    artifact.write_raw_mask(fused_mask)
    if refined.superpixels is not None:
        artifact.write_superpixels(refined.superpixels)
    artifact.write_superpixels_coarse(coarse_superpixels)
    artifact.write_prompt_winner(prompt_winner)
    artifact.write_tta_disagreement(disagreement)
    artifact.write_uncertainty(uncertainty.artifact)
    artifact.write_auto_change(refined.auto_change)
    artifact.write_proposed_mask(uncertainty.mask)
    colors = {definition.id: definition.display_color for definition in cfg.classes.classes}
    artifact.write_overlay(image, uncertainty.mask, colors, cfg.pipeline.ingest.jpeg_quality)


def _process_record(
    record: dict[str, Any], cfg: Any, paths: Any, repository: ManifestRepository, backend: Any, *, clock: Any = time.monotonic
) -> Any:
    artifact = ArtifactStore(Path(record["work_dir"]))
    image = artifact.read_image()
    scored = score_image(image, backend, cfg, exemplars_dir=paths.exemplars_dir)
    artifact_seconds = 0.0
    artifact_started = clock()
    artifact.write_scores(scored.scores, scored.scales, scored.winner_per_class, scored.tta_std_per_class)
    artifact_seconds += clock() - artifact_started
    fusion_started = clock()
    fused = fuse(scored.scores, image.shape[:2], cfg)
    fusion_seconds = clock() - fusion_started
    refinement_started = clock()
    refined = refine(
        RefinementContext(
            image,
            fused.mask,
            record["source"],
            record["gsd_m"],
            cfg.pipeline,
            _ids(cfg),
            class_scores=fused.scores,
        )
    )
    coarse_superpixels = _compute_coarse_superpixels(image, cfg)
    prompt_winner, disagreement = _derive_class_maps(
        refined.mask, scored.winner_per_class, scored.tta_std_per_class, cfg, image.shape[:2]
    )
    uncertain = calculate(
        refined.mask,
        fused.top_score,
        fused.margin,
        fused.conflict,
        disagreement,
        refined.auto_change,
        cfg.pipeline.uncertainty,
        cfg.pipeline.fusion.priority_margin,
    )
    refinement_seconds = clock() - refinement_started
    artifact_started = clock()
    _write_derived_artifacts(
        artifact, image, fused.mask, refined, coarse_superpixels, prompt_winner, disagreement, uncertain, cfg
    )
    artifact_seconds += clock() - artifact_started
    repository.transition(record["id"], "prelabeled", processing_seconds=scored.seconds, class_pixel_counts=_counts(uncertain.mask))
    current = repository.get(record["id"])
    assert current is not None
    artifact_started = clock()
    decision = _route_record(current, image, uncertain.mask, artifact, cfg, paths, repository)
    artifact_seconds += clock() - artifact_started
    tile_variants = int(scored.timings.get("tile_variants", 0.0))
    scale_tiles = ", ".join(
        f"{factor.removeprefix('tiles_')}={int(value)}"
        for factor, value in scored.timings.items()
        if factor.startswith("tiles_")
    )
    sam_seconds = scored.timings.get("sam", 0.0)
    total_seconds = scored.seconds + fusion_seconds + refinement_seconds + artifact_seconds
    print(
        f"Timing {record['id']}: image={image.shape[1]}x{image.shape[0]}, tiles={scale_tiles}, "
        f"TTA={len(cfg.pipeline.tta.variants)}, variants={tile_variants}, total={total_seconds:.2f}s, "
        f"per-variant={scored.seconds / max(1, tile_variants):.2f}s, SAM={sam_seconds:.2f}s, "
        f"score-overhead={scored.seconds - sam_seconds:.2f}s, fusion={fusion_seconds:.2f}s, "
        f"refinement={refinement_seconds:.2f}s, artifacts={artifact_seconds:.2f}s"
    )
    return decision


def prelabel(
    paths: Any,
    cfg: Any,
    backend_factory: Any,
    *,
    limit: int | None = None,
    max_hours: float | None = None,
    order: str = "spread",
    candidates: list[Candidate] | None = None,
    clock: Any = time.monotonic,
) -> PipelineSummary:
    """Ingest each new source, create all artifacts, route it and archive only successful originals."""
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max_hours must be positive")
    paths.ensure_runtime_dirs()
    summary = PipelineSummary()
    candidates = order_candidates(discover(paths), order) if candidates is None else list(candidates)
    if limit is not None:
        candidates = candidates[:limit]
    # Duplicates receive weight zero after ingest, but total_weight retains their image cost.
    # They are only discoverable while running, so the live ETA intentionally stays an upper bound.
    weights = {
        candidate.path: (size[0] * size[1] / 1_000_000 if (size := read_size(candidate.path)) else 0.0)
        for candidate in candidates
    }
    reporter = ProgressReporter(len(candidates), total_weight=sum(weights.values()), human=True, clock=clock)
    started = clock()
    budget_reached = False
    uncertain_reviews = 0
    with Database(paths.database) as connection:
        repository = ManifestRepository(connection)
        backend = _factory(backend_factory, cfg.pipeline)
        try:
            for candidate in candidates:
                try:
                    records = ingest_candidate(candidate, paths, repository, cfg, backend)
                    if not records:
                        summary.skipped_duplicates += 1
                        reporter.advance(weight=0)
                        if max_hours is not None and clock() - started >= max_hours * 3600:
                            budget_reached = True
                            break
                        continue
                    successful = True
                    for inserted in records:
                        record = repository.get(inserted.id)
                        assert record is not None
                        try:
                            record_started = clock()
                            decision = _process_record(record, cfg, paths, repository, backend, clock=clock)
                            elapsed = clock() - record_started
                            megapixels = record["width"] * record["height"] / 1_000_000
                            append_timing_entry(
                                paths.work_dir / "prelabel_timing.json",
                                {
                                    "fingerprint": timing_fingerprint(cfg.pipeline),
                                    "variants": count_variants(record["width"], record["height"], cfg.pipeline),
                                    "megapixels": megapixels,
                                    "seconds": elapsed,
                                },
                            )
                            summary.auto_accepted += decision.status == "auto_accepted"
                            summary.needs_review += decision.status == "needs_review"
                            summary.spot_checks += decision.spot_check
                            uncertain_reviews += decision.reason == "uncertain"
                        except Exception as error:  # noqa: BLE001 - one image must never stop the batch.
                            successful = False
                            summary.failed += 1
                            repository.record_error(record["original_path"], "prelabel", str(error), traceback.format_exc(), record["id"])
                    if successful:
                        archive_original(candidate.path, paths, candidate.source, candidate.session)
                except Exception as error:  # noqa: BLE001 - corrupt candidates have no image record yet.
                    summary.failed += 1
                    repository.record_error(str(candidate.path), "ingest", str(error), traceback.format_exc())
                reporter.advance(weight=weights[candidate.path])
                if max_hours is not None and clock() - started >= max_hours * 3600:
                    budget_reached = True
                    break
        finally:
            backend.close()
    print(f"auto-accepted={summary.auto_accepted} review={summary.needs_review} spot-checks={summary.spot_checks} failed={summary.failed} skipped-duplicates={summary.skipped_duplicates}")
    elapsed = max(0.0, clock() - started)
    processed = summary.auto_accepted + summary.needs_review + summary.failed
    per_hour = processed / (elapsed / 3600) if elapsed > 0 else 0.0
    uncertain = uncertain_reviews
    print(
        f"Duration: {format_duration(elapsed)} | images per hour: {per_hour:.1f} | auto-accepted: {summary.auto_accepted} | "
        f"needs review (uncertain): {uncertain} | needs review (spot check): {summary.spot_checks} | "
        f"failed: {summary.failed} | skipped duplicates: {summary.skipped_duplicates}"
    )
    print(
        f"Open review.bat to check the {summary.needs_review} images that need review "
        f"({uncertain} uncertain, {summary.spot_checks} spot checks)."
    )
    if budget_reached:
        print("Time budget reached – run prelabel again to continue")
        print(f"Remaining files: {len(candidates) - reporter.completed}")
    return summary


def reprocess(paths: Any, cfg: Any, image_ids: list[str] | None = None) -> PipelineSummary:
    """Re-run fusion, refinement, uncertainty and routing from stored score artifacts without a backend."""
    summary = PipelineSummary()
    with Database(paths.database) as connection:
        repository = ManifestRepository(connection)
        records = repository.query()
        for record in records:
            if image_ids is not None and record["id"] not in image_ids:
                continue
            if record["status"] in {"reviewed", "rejected"}:
                continue
            try:
                artifact = ArtifactStore(Path(record["work_dir"]))
                image = artifact.read_image()
                scores, _ = artifact.read_scores()
                fused = fuse(scores, image.shape[:2], cfg)
                winners, std_per_class = artifact.read_score_diagnostics()
                old_superpixels = artifact.read_superpixels() if (artifact.directory / "superpixels.npz").is_file() else None
                refined = refine(
                    RefinementContext(
                        image,
                        fused.mask,
                        record["source"],
                        record["gsd_m"],
                        cfg.pipeline,
                        _ids(cfg),
                        old_superpixels,
                        fused.scores,
                    )
                )
                existing_coarse = artifact.read_superpixels_coarse()
                coarse_superpixels = (
                    _compute_coarse_superpixels(image, cfg)
                    if existing_coarse is None or existing_coarse.shape != image.shape[:2]
                    else existing_coarse
                )
                prompt_winner, disagreement = _derive_class_maps(
                    refined.mask, winners, std_per_class, cfg, image.shape[:2]
                )
                uncertain = calculate(
                    refined.mask,
                    fused.top_score,
                    fused.margin,
                    fused.conflict,
                    disagreement,
                    refined.auto_change,
                    cfg.pipeline.uncertainty,
                    cfg.pipeline.fusion.priority_margin,
                )
                _write_derived_artifacts(
                    artifact,
                    image,
                    fused.mask,
                    refined,
                    coarse_superpixels,
                    prompt_winner,
                    disagreement,
                    uncertain,
                    cfg,
                )
                decision = _route_record(record, image, uncertain.mask, artifact, cfg, paths, repository)
                summary.auto_accepted += decision.status == "auto_accepted"
                summary.needs_review += decision.status == "needs_review"
                summary.spot_checks += decision.spot_check
            except Exception as error:  # noqa: BLE001
                summary.failed += 1
                repository.record_error(record["original_path"], "reprocess", str(error), traceback.format_exc(), record["id"])
    return summary
