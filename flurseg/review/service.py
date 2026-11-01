"""File and manifest operations for the local review workflow."""

from __future__ import annotations

import gzip
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from flurseg.artifacts import ArtifactStore
from flurseg.io.atomic import atomic_save_jpg, atomic_save_png
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ManifestRepository
from flurseg.paths import RepoPaths, ensure_inside
from flurseg.review.pyramid import Pyramid
from flurseg.segment.exemplars import find_similar

_VALID_VALUES = set(range(11)) | {255}


class ReviewError(ValueError):
    """A safe, client-facing review operation error."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ReviewService:
    """Unit-testable review operations with all data contained in one root."""

    def __init__(self, paths: RepoPaths, cfg: Any, backend_factory: Any | None = None) -> None:
        self.paths = paths
        self.cfg = cfg
        self.backend_factory = backend_factory
        self._backend: Any | None = None

    def _repository(self) -> tuple[Database, ManifestRepository]:
        database = Database(self.paths.database)
        connection = database.connect()
        return database, ManifestRepository(connection)

    def _record(self, image_id: str) -> dict[str, Any]:
        database, repository = self._repository()
        try:
            record = repository.get(image_id)
        finally:
            database.__exit__(None, None, None)
        if record is None:
            raise ReviewError("unknown image")
        return record

    def _work(self, record: dict[str, Any]) -> ArtifactStore:
        directory = Path(record["work_dir"])
        if not directory.is_absolute():
            directory = self.paths.root / directory
        return ArtifactStore(ensure_inside(directory, self.paths.root))

    def _review_dir(self, image_id: str) -> Path:
        return ensure_inside(self.paths.review_dir / image_id, self.paths.root)

    def _draft_path(self, image_id: str) -> Path:
        return self._review_dir(image_id) / "draft_mask.png"

    def _proposal(self, record: dict[str, Any]) -> np.ndarray:
        mask = self._work(record).read_mask("proposed_mask.png")
        if mask is None:
            raise ReviewError("proposed mask is missing")
        return mask.astype(np.uint8)

    def _working_mask(self, record: dict[str, Any]) -> np.ndarray:
        draft = self._draft_path(record["id"])
        if draft.is_file():
            value = cv2.imread(str(draft), cv2.IMREAD_UNCHANGED)
            if value is not None:
                return value.astype(np.uint8)
        return self._proposal(record)

    def _image(self, record: dict[str, Any]) -> np.ndarray:
        value = self._work(record).read_image()
        if value is None:
            raise ReviewError("working image is missing")
        return value

    def config(self) -> dict[str, Any]:
        shortcuts = [
            *[{"keys": str(class_id), "action": f"Select class {class_id}"} for class_id in range(10)],
            {"keys": "-", "action": "Select utility cover"}, {"keys": "I", "action": "Select ignore"},
            {"keys": "[", "action": "Decrease brush size"}, {"keys": "]", "action": "Increase brush size"},
            {"keys": "E", "action": "Ignore eraser"}, {"keys": "P", "action": "Superpixel tool"},
            {"keys": "F", "action": "Flood fill"}, {"keys": "G", "action": "Polygon fill"},
            {"keys": "A", "action": "Accept region"}, {"keys": "B", "action": "Find similar"},
            {"keys": "Y", "action": "Accept proposal"}, {"keys": "N", "action": "Discard proposal"},
            *[{"keys": f"Shift+{number}", "action": "Toggle layer"} for number in range(1, 7)],
            {"keys": "O", "action": "Cycle mask opacity"}, {"keys": "Ctrl+Z", "action": "Undo"},
            {"keys": "Ctrl+Y", "action": "Redo"}, {"keys": "Enter", "action": "Accept image or finish drawing"},
            {"keys": "R", "action": "Reject image"}, {"keys": "S", "action": "Skip image"},
            {"keys": "←", "action": "Previous image"}, {"keys": "→", "action": "Next image"},
            {"keys": "Home", "action": "Fit image"},
            {"keys": "?", "action": "Open help"}, {"keys": "F1", "action": "Open help"},
            {"keys": "Esc", "action": "Close help"},
        ]
        return {
            "classes": [
                {"id": definition.id, "name": definition.name, "color": definition.display_color,
                 "hotkey": "I" if definition.id == 255 else "-" if definition.id == 10 else str(definition.id)}
                for definition in self.cfg.classes.classes
            ],
            "labeling_rules": self.cfg.classes.labeling_rules,
            "shortcuts": shortcuts,
            "rule_names": ["edge_snap", "morphology", "shape", "context", "color", "ortho"],
            "find_similar_available": bool(
                self.backend_factory and self.cfg.pipeline.exemplars.find_similar.enabled
            ),
            "superpixel_levels": ["fine", "coarse"],
        }

    def queue(self, *, source: str | None = None, session: str | None = None, reason: str | None = None,
              status: str | None = None) -> list[dict[str, Any]]:
        database, repository = self._repository()
        try:
            requested_status = status or "needs_review"
            rows = repository.query(status=requested_status, source=source, session=session)
        finally:
            database.__exit__(None, None, None)
        if reason is not None:
            rows = [row for row in rows if row["routing_reason"] == reason]
        return sorted(rows, key=lambda row: row["uncertain_fraction"] if row["uncertain_fraction"] is not None else -1, reverse=True)

    def pool(self) -> list[dict[str, Any]]:
        return self.queue(status="auto_accepted")

    def image_meta(self, image_id: str) -> dict[str, Any]:
        record = self._record(image_id)
        pyramid = Pyramid(self.paths, image_id, self._work(record).directory)
        height, width = pyramid.image_shape()
        return {
            "id": image_id, "width": width, "height": height, "size": {"width": width, "height": height},
            "source": record["source"], "session": record["session"], "gsd_m": record["gsd_m"],
            "routing_reason": record["routing_reason"], "uncertain_fraction": record["uncertain_fraction"],
            "pyramid_levels": pyramid.levels(), "levels": pyramid.levels(),
        }

    def tile(self, image_id: str, layer: str, level: int, x: int, y: int) -> Path:
        record = self._record(image_id)
        return Pyramid(self.paths, image_id, self._work(record).directory).tile(layer, level, x, y)

    def mask_bytes(self, image_id: str) -> bytes:
        return gzip.compress(self._working_mask(self._record(image_id)).tobytes())

    def superpixels_bytes(self, image_id: str, level: str = "fine") -> bytes:
        if level not in {"fine", "coarse"}:
            raise ReviewError(f"unknown superpixel level: {level}")
        work = self._work(self._record(image_id))
        if level == "coarse":
            labels = work.read_superpixels_coarse()
            if labels is None:
                raise ReviewError("coarse superpixels not available")
            return gzip.compress(labels.astype("<u4", copy=False).tobytes())
        labels = work.read_superpixels().astype("<u4", copy=False)
        return gzip.compress(labels.tobytes())

    def proposal_classes_bytes(self, image_id: str) -> bytes:
        mask = self._work(self._record(image_id)).read_mask("raw_mask.png")
        if mask is None:
            raise ReviewError("proposal classes are missing")
        return gzip.compress(mask.astype(np.uint8).tobytes())

    def save_draft(self, image_id: str, data: bytes) -> None:
        record = self._record(image_id)
        expected = record["width"] * record["height"]
        if len(data) != expected:
            raise ReviewError(f"mask length must be {expected} bytes")
        mask = np.frombuffer(data, dtype=np.uint8).reshape(record["height"], record["width"])
        if not set(np.unique(mask)).issubset(_VALID_VALUES):
            raise ReviewError("mask values must be 0 through 10 or 255")
        target = self._draft_path(image_id)
        ensure_inside(target, self.paths.root)
        atomic_save_png(target, mask)

    @staticmethod
    def _changes(proposed: np.ndarray, edited: np.ndarray) -> dict[str, int]:
        changed = proposed != edited
        return {str(class_id): int(((edited == class_id) & changed).sum()) for class_id in range(11) if ((edited == class_id) & changed).any()} | (
            {"255": int(((edited == 255) & changed).sum())} if ((edited == 255) & changed).any() else {}
        )

    def _unlink_known(self, *targets: Path) -> None:
        for target in targets:
            ensure_inside(target, self.paths.root)
            target.unlink(missing_ok=True)

    def accept(self, image_id: str, review_seconds: float = 0.0) -> dict[str, Any]:
        record = self._record(image_id)
        if record["status"] not in {"needs_review", "auto_accepted"}:
            raise ReviewError("image cannot be accepted from its current status")
        proposal, edited, image = self._proposal(record), self._working_mask(record), self._image(record)
        changes = self._changes(proposal, edited)
        image_target = ensure_inside(self.paths.pool_images_dir / f"{image_id}.jpg", self.paths.root)
        mask_target = ensure_inside(self.paths.pool_masks_dir / f"{image_id}.png", self.paths.root)
        atomic_save_jpg(image_target, image, self.cfg.pipeline.ingest.jpeg_quality)
        atomic_save_png(mask_target, edited)
        database, repository = self._repository()
        try:
            rel_image_path = str(image_target.relative_to(self.paths.root))
        except ValueError:
            rel_image_path = str(image_target.relative_to(ensure_inside(self.paths.root, self.paths.root)))
        try:
            repository.transition(
                image_id, "reviewed", review_seconds=max(0.0, float(review_seconds)), review_changed_pixels=changes,
                image_path=rel_image_path,
            )
            repository.add_review(image_id, _now(), _now(), "accept", changes, max(0.0, float(review_seconds)))
        finally:
            database.__exit__(None, None, None)
        self._unlink_known(self._review_dir(image_id) / "image.jpg", self._review_dir(image_id) / "proposed_mask.png",
                           self._review_dir(image_id) / "uncertainty.png", self._draft_path(image_id))
        return {"id": image_id, "changed_pixels": changes, "status": "reviewed"}

    def reject(self, image_id: str, review_seconds: float = 0.0) -> dict[str, Any]:
        record = self._record(image_id)
        if record["status"] != "needs_review":
            raise ReviewError("image cannot be rejected from its current status")
        database, repository = self._repository()
        try:
            repository.transition(image_id, "rejected", review_seconds=max(0.0, float(review_seconds)))
            repository.add_review(image_id, _now(), _now(), "reject", None, max(0.0, float(review_seconds)))
        finally:
            database.__exit__(None, None, None)
        return {"id": image_id, "status": "rejected"}

    def skip(self, image_id: str, review_seconds: float = 0.0) -> dict[str, Any]:
        record = self._record(image_id)
        if record["status"] != "needs_review":
            raise ReviewError("only queued images can be skipped")
        database, repository = self._repository()
        try:
            repository.add_review(image_id, _now(), _now(), "skip", None, max(0.0, float(review_seconds)))
        finally:
            database.__exit__(None, None, None)
        return {"id": image_id, "status": "needs_review"}

    def to_review(self, image_id: str) -> dict[str, Any]:
        record = self._record(image_id)
        if record["status"] != "auto_accepted":
            raise ReviewError("only pool images can be returned to review")
        image, mask = self._image(record), self._proposal(record)
        directory = self._review_dir(image_id)
        atomic_save_jpg(ensure_inside(directory / "image.jpg", self.paths.root), image, self.cfg.pipeline.ingest.jpeg_quality)
        atomic_save_png(ensure_inside(directory / "proposed_mask.png", self.paths.root), mask)
        uncertainty = self._work(record).read_mask("uncertainty.png")
        if uncertainty is not None:
            atomic_save_png(ensure_inside(directory / "uncertainty.png", self.paths.root), uncertainty)
        database, repository = self._repository()
        try:
            repository.transition(image_id, "needs_review", routing_reason="manual_review")
        finally:
            database.__exit__(None, None, None)
        self._unlink_known(self.paths.pool_images_dir / f"{image_id}.jpg", self.paths.pool_masks_dir / f"{image_id}.png")
        return {"id": image_id, "status": "needs_review"}

    def stats(self) -> dict[str, Any]:
        database, repository = self._repository()
        try:
            all_rows = repository.query()
        finally:
            database.__exit__(None, None, None)
        done = sum(row["status"] in {"reviewed", "rejected"} for row in all_rows)
        open_count = sum(row["status"] == "needs_review" for row in all_rows)
        spot_checks = [row for row in all_rows if row["status"] == "reviewed" and row["spot_check"]]
        changed_total = sum(sum((row["review_changed_pixels"] or {}).values()) for row in spot_checks)
        labelled_total = sum(
            sum(count for class_id, count in (row["class_pixel_counts"] or {}).items() if class_id != "255")
            for row in spot_checks
        )
        per_class: dict[str, float] = {}
        for class_id in map(str, range(11)):
            changed = sum((row["review_changed_pixels"] or {}).get(class_id, 0) for row in spot_checks)
            labelled = sum((row["class_pixel_counts"] or {}).get(class_id, 0) for row in spot_checks)
            per_class[class_id] = changed / labelled if labelled else 0.0
        error = {"total": changed_total / labelled_total if labelled_total else 0.0, "per_class": per_class}
        return {"done": done, "open": open_count, "auto_accept_error_rate": error, "error_rate": error}

    @staticmethod
    def _similar_window(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
        """Return the full-image coordinates of the fixed SAM search window."""
        height, width = image.shape[:2]
        cx = round(sum((x0 + x1) / 2 for x0, _, x1, _ in boxes) / len(boxes))
        cy = round(sum((y0 + y1) / 2 for _, y0, _, y1 in boxes) / len(boxes))
        window_width, window_height = min(1008, width), min(1008, height)
        left = min(max(0, cx - window_width // 2), max(0, width - window_width))
        top = min(max(0, cy - window_height // 2), max(0, height - window_height))
        return left, top, left + window_width, top + window_height

    def _find_similar_result(self, image_id: str, class_id: int, boxes: list[list[int]]) -> tuple[bytes, tuple[int, int, int, int]]:
        if not self.backend_factory or not self.cfg.pipeline.exemplars.find_similar.enabled:
            raise ReviewError("Find similar is unavailable: configure a CUDA backend and SAM 3 checkpoint.")
        if class_id not in range(11):
            raise ReviewError("class_id must be a trainable class")
        normalized = [tuple(map(int, box)) for box in boxes if len(box) == 4]
        if not normalized:
            raise ReviewError("at least one valid box is required")
        if self._backend is None:
            self._backend = self.backend_factory()
        image = self._image(self._record(image_id))
        result = find_similar(image, normalized, self.cfg.class_by_id(class_id).prompts[0], self._backend)
        data = gzip.compress(np.rint(np.clip(result, 0, 1) * 255).astype(np.uint8).tobytes())
        return data, self._similar_window(image, normalized)

    def find_similar(self, image_id: str, class_id: int, boxes: list[list[int]]) -> bytes:
        """Return binary scores for existing callers that do not need window metadata."""
        return self._find_similar_result(image_id, class_id, boxes)[0]

    def find_similar_with_window(self, image_id: str, class_id: int, boxes: list[list[int]]) -> tuple[bytes, tuple[int, int, int, int]]:
        """Return binary scores and the corresponding full-image search window."""
        return self._find_similar_result(image_id, class_id, boxes)
