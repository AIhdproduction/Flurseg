from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from flurseg.artifacts import ArtifactStore
from flurseg.config import load_config
from flurseg.manifest.db import Database
from flurseg.manifest.repo import ManifestRepository
from flurseg.paths import RepoPaths
from flurseg.pipeline import prelabel, reprocess
from flurseg.segment.fake_backend import FakeBackend


def _prepare(paths: RepoPaths, names: tuple[str, ...] = ("one",)):
    session = paths.drone_input_dir / "s1"
    session.mkdir(parents=True)
    for index, name in enumerate(names):
        Image.fromarray(np.full((20, 24, 3), 80 + index * 20, np.uint8)).save(session / f"{name}.jpg")
    cfg = load_config(paths)
    cfg.pipeline.routing.auto_accept_max_uncertain_fraction = 1.0
    cfg.pipeline.routing.auto_accept_max_ignore_fraction = 1.0
    cfg.pipeline.routing.spot_check_fraction = 0.0
    prelabel(paths, cfg, FakeBackend)
    with Database(paths.database) as connection:
        records = ManifestRepository(connection).query()
    return cfg, records


def _records(paths: RepoPaths):
    with Database(paths.database) as connection:
        return ManifestRepository(connection).query()


def test_lowering_uncertain_fraction_reroutes_without_a_backend(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg, _ = _prepare(paths)
    cfg.pipeline.routing.auto_accept_max_uncertain_fraction = 0.0
    cfg.pipeline.uncertainty.threshold = 0.0
    result = reprocess(paths, cfg)
    assert result.needs_review == 1 and _records(paths)[0]["status"] == "needs_review"


def test_reviewed_and_rejected_records_are_not_changed(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg, records = _prepare(paths, ("one", "two"))
    with Database(paths.database) as connection:
        repository = ManifestRepository(connection)
        repository.transition(str(records[0]["id"]), "reviewed")
        repository.transition(str(records[1]["id"]), "rejected")
    before = [(paths.work_dir / str(record["id"]) / "proposed_mask.png").read_bytes() for record in records]
    assert reprocess(paths, cfg).as_dict() == {"auto_accepted": 0, "needs_review": 0, "spot_checks": 0, "failed": 0, "skipped_duplicates": 0}
    assert [record["status"] for record in _records(paths)] == ["reviewed", "rejected"]
    assert before == [(paths.work_dir / str(record["id"]) / "proposed_mask.png").read_bytes() for record in records]


def test_reprocess_reuses_stored_superpixels(monkeypatch: pytest.MonkeyPatch, tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg, records = _prepare(paths)
    assert (paths.work_dir / str(records[0]["id"]) / "superpixels.npz").is_file()
    monkeypatch.setattr("flurseg.refine.edge_snap.slic", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must reuse labels")))
    assert reprocess(paths, cfg).failed == 0


def test_reprocess_applies_priority_margin_to_raw_mask_and_prompt_winner(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg, records = _prepare(paths)
    work = paths.work_dir / str(records[0]["id"])
    scores = {"ground": np.full((20, 24), 0.95, np.float32), "vehicle": np.full((20, 24), 0.85, np.float32)}
    diagnostics = {name: np.zeros((20, 24), np.uint8) for name in scores}
    ArtifactStore(work).write_scores(scores, {"ground": 1.0, "vehicle": 1.0}, diagnostics, diagnostics)
    reprocess(paths, cfg)
    first_mask = ArtifactStore(work).read_mask("raw_mask.png").copy()
    first_winner = ArtifactStore(work).read_mask("prompt_winner.png").copy()
    cfg.pipeline.fusion.priority_margin = 0.0
    reprocess(paths, cfg)
    second_mask = ArtifactStore(work).read_mask("raw_mask.png")
    second_winner = ArtifactStore(work).read_mask("prompt_winner.png")
    assert not np.array_equal(first_mask, second_mask) and not np.array_equal(first_winner, second_winner)


def test_reprocess_rewrites_overlay_when_the_mask_changes(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg, records = _prepare(paths)
    work = paths.work_dir / str(records[0]["id"])
    scores = {"ground": np.full((20, 24), 0.95, np.float32), "vehicle": np.full((20, 24), 0.85, np.float32)}
    diagnostics = {name: np.zeros((20, 24), np.uint8) for name in scores}
    ArtifactStore(work).write_scores(scores, {"ground": 1.0, "vehicle": 1.0}, diagnostics, diagnostics)
    reprocess(paths, cfg)
    first_overlay = (work / "overlay.jpg").read_bytes()
    cfg.pipeline.fusion.priority_margin = 0.0
    reprocess(paths, cfg)
    assert first_overlay != (work / "overlay.jpg").read_bytes()


def test_reprocess_subset_only_touches_requested_image(tmp_root) -> None:
    paths = RepoPaths.from_root(tmp_root)
    cfg, records = _prepare(paths, ("one", "two"))
    selected, untouched = (str(record["id"]) for record in records)
    before = {record["id"]: record["updated_at"] for record in records}
    result = reprocess(paths, cfg, [selected])
    after = {record["id"]: record["updated_at"] for record in _records(paths)}
    assert result.auto_accepted + result.needs_review == 1 and after[untouched] == before[untouched]
