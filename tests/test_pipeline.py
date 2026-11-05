from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from flurseg import pipeline
from flurseg.config import load_config
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ManifestRepository
from flurseg.paths import RepoPaths
from flurseg.pipeline import prelabel
from flurseg.segment.fake_backend import FakeBackend


def _write_rgb(path, value: int = 120, *, format: str | None = None) -> None:
    Image.fromarray(np.full((20, 24, 3), value, np.uint8)).save(path, format=format)


def _write_inputs(paths: RepoPaths) -> None:
    (paths.drone_input_dir / "drone-session").mkdir(parents=True)
    _write_rgb(paths.drone_input_dir / "drone-session" / "drone.jpg", 70)
    (paths.phone_input_dir / "phone-session").mkdir(parents=True)
    _write_rgb(paths.phone_input_dir / "phone-session" / "phone.heic", 120, format="JPEG")
    session = paths.ortho_input_dir / "ortho-session"
    session.mkdir(parents=True)
    with rasterio.open(
        session / "ortho.tif", "w", driver="GTiff", height=18, width=22, count=3, dtype="uint8",
        transform=from_origin(2600000, 1200000, 0.1, 0.1), crs="EPSG:2056",
    ) as dataset:
        dataset.write(np.full((3, 18, 22), 160, np.uint8))


def _records(paths: RepoPaths) -> list[dict[str, object]]:
    with Database(paths.database) as connection:
        return ManifestRepository(connection).query()


def _configured(paths: RepoPaths):
    cfg = load_config(paths)
    cfg.pipeline.routing.auto_accept_max_uncertain_fraction = 1.0
    cfg.pipeline.routing.auto_accept_max_ignore_fraction = 1.0
    cfg.pipeline.routing.spot_check_fraction = 0.0
    return cfg


def test_prelabel_processes_drone_phone_and_ortho_inputs(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    _write_inputs(paths)
    summary = prelabel(paths, _configured(paths), FakeBackend)
    assert summary.auto_accepted + summary.needs_review == 3
    assert {record["source"] for record in _records(paths)} == {"drone", "phone", "ortho"}
    assert {record["status"] for record in _records(paths)} <= {"auto_accepted", "needs_review"}


def test_every_processed_image_has_the_complete_artifact_set(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    _write_inputs(paths)
    prelabel(paths, _configured(paths), FakeBackend)
    expected = {
        "image.jpg", "scores.npz", "prompt_winner.png", "tta_disagreement.png", "superpixels.npz",
        "superpixels_coarse.npz", "raw_mask.png", "proposed_mask.png", "uncertainty.png",
        "auto_change.png", "overlay.jpg", "meta.json",
    }
    assert all({path.name for path in (paths.work_dir / str(record["id"])).iterdir()} >= expected for record in _records(paths))


def test_successful_originals_are_archived_by_source_and_session(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    _write_inputs(paths)
    prelabel(paths, _configured(paths), FakeBackend)
    assert (paths.archive_dir / "drone" / "drone-session" / "drone.jpg").is_file()
    assert (paths.archive_dir / "phone" / "phone-session" / "phone.heic").is_file()
    assert (paths.archive_dir / "ortho" / "ortho-session" / "ortho.tif").is_file()


def test_second_run_skips_a_readded_identical_original(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "same"
    session.mkdir(parents=True)
    _write_rgb(session / "image.jpg", 90)
    cfg = _configured(paths)
    prelabel(paths, cfg, FakeBackend)
    _write_rgb(session / "image.jpg", 90)
    summary = prelabel(paths, cfg, FakeBackend)
    assert summary.as_dict() == {"auto_accepted": 0, "needs_review": 0, "spot_checks": 0, "failed": 0, "skipped_duplicates": 1}


def test_corrupt_jpg_is_recorded_without_stopping_other_images(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "mixed"
    session.mkdir(parents=True)
    _write_rgb(session / "good.jpg", 90)
    (session / "broken.jpg").write_bytes(b"not an image")
    summary = prelabel(paths, _configured(paths), FakeBackend)
    with Database(paths.database) as connection:
        errors = connection.execute("SELECT COUNT(*) FROM processing_errors").fetchone()[0]
    assert summary.failed == 1 and summary.auto_accepted + summary.needs_review == 1 and errors == 1


def test_summary_routing_numbers_match_manifest_counts(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    _write_inputs(paths)
    summary = prelabel(paths, _configured(paths), FakeBackend)
    records = _records(paths)
    assert summary.auto_accepted == sum(record["status"] == "auto_accepted" for record in records)
    assert summary.needs_review == sum(record["status"] == "needs_review" for record in records)


def test_prompt_winner_is_valid_for_each_assigned_class(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    _write_inputs(paths)
    cfg = _configured(paths)
    prelabel(paths, cfg, FakeBackend)
    global_indices: dict[int, set[int]] = {}
    for index, (_, class_id) in enumerate(cfg.prompt_list()):
        global_indices.setdefault(class_id, set()).add(index)
    for record in _records(paths):
        work = paths.work_dir / str(record["id"])
        mask = cv2.imread(str(work / "proposed_mask.png"), cv2.IMREAD_UNCHANGED)
        winners = cv2.imread(str(work / "prompt_winner.png"), cv2.IMREAD_UNCHANGED)
        assert all(value == 255 or int(value) in global_indices.get(class_id, set()) for value, class_id in zip(winners.ravel(), mask.ravel(), strict=True))


def test_prompt_winner_follows_refined_class(tmp_root, monkeypatch) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "drone-session"
    session.mkdir(parents=True)
    image = np.full((30, 30, 3), 120, dtype=np.uint8)
    image[10:20, 10:20] = [20, 240, 20]
    Image.fromarray(image).save(session / "drone.jpg")

    cfg = _configured(paths)
    cfg.pipeline.refinement.color.enabled = True

    ground_id = cfg.class_by_name("ground").id
    real_fuse = pipeline.fuse

    def patched_fuse(scores, shape, config):
        result = real_fuse(scores, shape, config)
        result.mask[10:20, 10:20] = ground_id
        return result

    monkeypatch.setattr(pipeline, "fuse", patched_fuse)

    prelabel(paths, cfg, FakeBackend)

    records = _records(paths)
    assert len(records) == 1
    work = paths.work_dir / str(records[0]["id"])
    winners = cv2.imread(str(work / "prompt_winner.png"), cv2.IMREAD_UNCHANGED)
    low_veg_id = cfg.class_by_name("low_vegetation").id
    low_veg_globals = {
        index for index, (_, class_id) in enumerate(cfg.prompt_list()) if class_id == low_veg_id
    }
    patch_winners = winners[10:20, 10:20].ravel()
    assert len(patch_winners) > 0
    assert all(int(value) == 255 or int(value) in low_veg_globals for value in patch_winners)


def test_prompt_winner_is_low_vegetation_prompt_when_backend_has_evidence(tmp_root, monkeypatch) -> None:
    class CustomBackend:
        name = "custom"

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def score_tile(self, tile: np.ndarray, prompts: Sequence[str]) -> np.ndarray:
            height, width = tile.shape[:2]
            scores = np.full((len(prompts), height, width), 0.1, dtype=np.float32)
            for idx, prompt in enumerate(prompts):
                if prompt == "grass":
                    scores[idx] = 0.9
            return scores

        def score_boxes(
            self, image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], prompt: str
        ) -> np.ndarray:
            return np.zeros(image.shape[:2], dtype=np.float32)

        def close(self) -> None:
            pass

    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "drone-session"
    session.mkdir(parents=True)
    image = np.full((30, 30, 3), 120, dtype=np.uint8)
    image[10:20, 10:20] = [20, 240, 20]
    Image.fromarray(image).save(session / "drone.jpg")

    cfg = _configured(paths)
    cfg.pipeline.refinement.color.enabled = True
    cfg.pipeline.uncertainty.threshold = 1.0

    ground_id = cfg.class_by_name("ground").id
    real_fuse = pipeline.fuse

    def patched_fuse(scores, shape, config):
        result = real_fuse(scores, shape, config)
        result.mask[10:20, 10:20] = ground_id
        return result

    monkeypatch.setattr(pipeline, "fuse", patched_fuse)

    prelabel(paths, cfg, CustomBackend)

    records = _records(paths)
    assert len(records) == 1
    work = paths.work_dir / str(records[0]["id"])
    winners = cv2.imread(str(work / "prompt_winner.png"), cv2.IMREAD_UNCHANGED)
    proposed = cv2.imread(str(work / "proposed_mask.png"), cv2.IMREAD_UNCHANGED)
    low_veg_id = cfg.class_by_name("low_vegetation").id
    grass_global = next(
        index for index, (prompt, _) in enumerate(cfg.prompt_list()) if prompt == "grass"
    )
    patch_winners = winners[10:20, 10:20]
    patch_proposed = proposed[10:20, 10:20]
    low_veg_pixels = patch_proposed == low_veg_id
    assert low_veg_pixels.sum() >= 1
    assert (patch_winners[low_veg_pixels] == grass_global).all()


def test_prelabel_and_reprocess_coarse_superpixels(tmp_root) -> None:
    from flurseg.artifacts import ArtifactStore
    from flurseg.pipeline import reprocess

    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "drone-session"
    session.mkdir(parents=True)
    image = np.full((200, 200, 3), 120, dtype=np.uint8)
    image[50:150, 50:150] = 40
    image[100:150, 100:150] = 200
    Image.fromarray(image).save(session / "drone.jpg")

    cfg = _configured(paths)
    prelabel(paths, cfg, FakeBackend)

    records = _records(paths)
    assert len(records) == 1
    record = records[0]
    store = ArtifactStore(paths.work_dir / str(record["id"]))

    fine = store.read_superpixels()
    coarse = store.read_superpixels_coarse()
    assert coarse is not None
    assert coarse.shape == (200, 200)
    assert coarse.dtype == np.int32
    fine_label_count = len(np.unique(fine))
    coarse_label_count = len(np.unique(coarse))
    assert coarse_label_count < fine_label_count

    # Reprocess reuses existing coarse superpixels if shape matches
    marker = np.full((200, 200), 9999, dtype=np.int32)
    store.write_superpixels_coarse(marker)
    reprocess(paths, cfg, [str(record["id"])])
    reused = store.read_superpixels_coarse()
    assert reused is not None
    assert np.array_equal(reused, marker)

    # Reprocess recomputes coarse superpixels if shape does not match
    mismatched = np.full((10, 10), 9999, dtype=np.int32)
    store.write_superpixels_coarse(mismatched)
    reprocess(paths, cfg, [str(record["id"])])
    recomputed = store.read_superpixels_coarse()
    assert recomputed is not None
    assert recomputed.shape == (200, 200)
    assert len(np.unique(recomputed)) == coarse_label_count


def test_prelabel_stops_after_time_budget_and_second_run_continues(tmp_root, capsys) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "budget"
    session.mkdir(parents=True)
    _write_rgb(session / "first.jpg", 30)
    _write_rgb(session / "second.jpg", 60)
    cfg = _configured(paths)
    calls = 0

    def clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 15 else 3700.0

    first = prelabel(paths, cfg, FakeBackend, max_hours=1, clock=clock)
    assert first.auto_accepted == 1
    output = capsys.readouterr().out
    assert "Time budget reached – run prelabel again to continue" in output
    assert "Remaining files: 1" in output
    second = prelabel(paths, cfg, FakeBackend)
    assert second.auto_accepted == 1


def test_prelabel_default_spread_order_and_duplicate_weight_zero(tmp_root, monkeypatch) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "spread"
    session.mkdir(parents=True)
    for index in range(4):
        _write_rgb(session / f"image-{index}.jpg", index)
    seen: list[str] = []
    weights: list[float] = []

    class Reporter:
        def __init__(self, *args, **kwargs) -> None:
            self.completed = 0

        def advance(self, weight: float = 1.0) -> str:
            self.completed += 1
            weights.append(weight)
            return ""

    def duplicate(candidate, *args, **kwargs):
        seen.append(candidate.path.name)
        return []

    monkeypatch.setattr(pipeline, "ProgressReporter", Reporter)
    monkeypatch.setattr(pipeline, "ingest_candidate", duplicate)
    prelabel(paths, _configured(paths), FakeBackend)
    assert seen == ["image-0.jpg", "image-2.jpg", "image-1.jpg", "image-3.jpg"]
    assert weights == [0, 0, 0, 0]


def test_prelabel_final_summary_names_review_counts(tmp_root, capsys) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "review"
    session.mkdir(parents=True)
    _write_rgb(session / "image.jpg")
    cfg = _configured(paths)
    cfg.pipeline.routing.spot_check_fraction = 1.0
    summary = prelabel(paths, cfg, FakeBackend)
    output = capsys.readouterr().out
    assert summary.needs_review == 1
    assert "Open review.bat to check the 1 images that need review (0 uncertain, 1 spot checks)." in output


def test_prelabel_summary_counts_uncertain_and_test_session_reasons_separately(tmp_root, monkeypatch, capsys) -> None:
    paths = RepoPaths.from_root(tmp_root)
    session = paths.drone_input_dir / "reasons"
    session.mkdir(parents=True)
    _write_rgb(session / "first.jpg", 70)
    _write_rgb(session / "second.jpg", 90)
    decisions = iter(
        [
            SimpleNamespace(status="needs_review", reason="uncertain", spot_check=False),
            SimpleNamespace(status="needs_review", reason="test_session", spot_check=True),
        ]
    )
    monkeypatch.setattr(pipeline, "_process_record", lambda *args, **kwargs: next(decisions))
    summary = prelabel(paths, _configured(paths), FakeBackend)
    output = capsys.readouterr().out
    assert summary.needs_review == 2 and summary.spot_checks == 1
    assert "needs review (uncertain): 1 | needs review (spot check): 1" in output
    assert "(1 uncertain, 1 spot checks)." in output
