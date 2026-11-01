"""CPU-only regression tests for the training engine."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from flurseg.train.config import TrainConfig
from flurseg.train.engine import (
    anneal_attention_masks,
    build_model,
    build_scheduler,
    confusion_matrix,
    deterministic_holdout_ids,
    metrics_from_confusion,
    resolve_warmup_steps,
    run_training,
    setup,
)
from flurseg.train.predict import sliding_window_scores


@pytest.fixture(autouse=True)
def _force_cpu_training(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep mandatory training regressions independent of a local CUDA device."""
    monkeypatch.setattr("flurseg.train.engine.torch.cuda.is_available", lambda: False)


def _config(total_steps: int, eval_every: int = 1, patience: int = 8) -> TrainConfig:
    """Return a small fully offline configuration suitable for CPU tests."""
    return TrainConfig.model_validate(
        {
            "model": {"pretrained": "offline", "tiny_random": True},
            "data": {
                "crop_size": 32,
                "scale_min": 1,
                "scale_max": 1,
                "color_jitter": 0,
            },
            "inference": {"window_size": 32, "stride": 16},
            "training": {
                "batch_size": 2,
                "gradient_accumulation": 1,
                "total_steps": total_steps,
                "warmup_steps": 0,
                "eval_every": eval_every,
                "patience": patience,
                "head_lr": 0.01,
                "backbone_lr": 0.01,
            },
        }
    )


def _write_dataset(root: Path, *, val_empty: bool = False) -> None:
    """Create a two-class training dataset entirely under a pytest temporary root."""
    dataset = root / "train" / "dataset"
    metadata: dict[str, dict[str, str]] = {}
    split_counts = (("train", 4),) if val_empty else (("train", 2), ("val", 1))
    for split, count in split_counts:
        for folder in ("images", "masks"):
            (dataset / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f"{split}-{index}"
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[8:24, 8:24] = 220
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 1
            cv2.imwrite(str(dataset / split / "images" / f"{name}.png"), image)
            cv2.imwrite(str(dataset / split / "masks" / f"{name}.png"), mask)
            metadata[name] = {
                "source": "drone",
                "session": f"session-{split}",
                "split": split,
            }
    manifest = {
        "ignore_index": 255,
        "dataset_version": 7,
        "classes": [
            {"id": 0, "name": "other", "color": "#000000"},
            {"id": 1, "name": "ground", "color": "#ffffff"},
        ],
        "tile_metadata": metadata,
        "val_empty": val_empty,
    }
    (dataset / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_config(root: Path, config: TrainConfig) -> Path:
    """Serialize a test configuration and return its path."""
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return path


def test_tiny_overfit_reduces_loss_after_fifteen_steps(tmp_path: Path) -> None:
    """A tiny random EoMT model learns the repeated two-tile training pattern."""
    _write_dataset(tmp_path)
    config = _config(total_steps=15, eval_every=15)
    run = run_training(tmp_path, _write_config(tmp_path, config), config, steps=15)
    with (run / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 15
    assert float(rows[-1]["loss"]) < float(rows[0]["loss"])


def test_internal_holdout_is_used_for_an_empty_validation_split(tmp_path: Path) -> None:
    """A flagged dataset evaluates one stable train tile without using augmentation."""
    _write_dataset(tmp_path, val_empty=True)
    config = _config(total_steps=1, eval_every=1)
    config_path = _write_config(tmp_path, config)

    context = setup(tmp_path, config_path, config, resume=False, steps=1, dry_run=False)

    assert len(context.train_set) == 3
    assert len(context.val_set) == 1
    assert context.val_set.split == "train"
    assert context.val_set.training is False
    assert {path.stem for path in context.val_set.images} == deterministic_holdout_ids(
        ["train-0", "train-1", "train-2", "train-3"]
    )
    assert "WARNING: validation split is empty" in (context.run_dir / "train.log").read_text(
        encoding="utf-8"
    )


def test_resume_restores_state_and_keeps_metrics_contiguous(tmp_path: Path) -> None:
    """A resumed run retains step, optimizer state, and all earlier CSV rows."""
    _write_dataset(tmp_path)
    config = _config(total_steps=6, eval_every=3)
    config_path = _write_config(tmp_path, config)
    first = run_training(tmp_path, config_path, config, steps=6)
    resumed = run_training(tmp_path, config_path, config, resume=True, steps=10)
    state = torch.load(
        resumed / "last" / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    with (resumed / "metrics.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert first != resumed
    assert state["step"] == 10
    assert state["optimizer"]["state"]
    assert [int(row["step"]) for row in rows] == list(range(1, 11))


def test_early_stopping_uses_patience_evaluations(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Repeated non-improving validation metrics stop after the configured patience."""
    _write_dataset(tmp_path)

    def constant_evaluation(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "iou": [0.0, 0.0],
            "miou": 0.0,
            "pixel_accuracy": 0.0,
            "confusion": np.zeros((2, 2), dtype=np.int64),
        }

    monkeypatch.setattr("flurseg.train.engine.evaluate", constant_evaluation)
    config = _config(total_steps=10, eval_every=1, patience=2)
    run = run_training(tmp_path, _write_config(tmp_path, config), config, steps=10)
    state = torch.load(run / "last" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 3
    assert state["stale"] == 2


def test_confusion_metrics_ignore_handmade_pixels() -> None:
    """Ignore labels do not affect the confusion matrix or derived mIoU."""
    prediction = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    target = np.array([[0, 1, 255], [1, 1, 0]], dtype=np.uint8)
    matrix = confusion_matrix(prediction, target, classes=2)
    iou, miou, accuracy = metrics_from_confusion(matrix)
    assert matrix.tolist() == [[1, 1], [1, 2]]
    assert iou == pytest.approx([1 / 3, 1 / 2])
    assert miou == pytest.approx(5 / 12)
    assert accuracy == pytest.approx(3 / 5)


def test_oom_fallback_halves_batch_and_doubles_accumulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A simulated first CUDA OOM recovers on CPU without abandoning the run."""
    _write_dataset(tmp_path)
    from flurseg.train import engine

    original_loss = engine.ignore_aware_loss
    calls = 0

    def fail_once(*args: object, **kwargs: object):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return original_loss(*args, **kwargs)

    monkeypatch.setattr(engine, "ignore_aware_loss", fail_once)
    monkeypatch.setattr(
        engine,
        "evaluate",
        lambda *_args, **_kwargs: {
            "iou": [0.0, 0.0],
            "miou": 0.0,
            "pixel_accuracy": 0.0,
            "confusion": np.zeros((2, 2), dtype=np.int64),
        },
    )
    config = _config(total_steps=1, eval_every=1)
    run = run_training(tmp_path, _write_config(tmp_path, config), config, steps=1)
    state = torch.load(run / "last" / "training_state.pt", map_location="cpu", weights_only=False)
    assert state["step"] == 1
    assert state["batch_size"] == 1
    assert state["accumulation"] == 2


def test_attention_mask_annealing_is_staggered_per_block() -> None:
    """Models with per-block probabilities do not receive one shared schedule value."""
    model = torch.nn.Module()
    model.register_buffer("attn_mask_probs", torch.ones(3))
    anneal_attention_masks(model, step=4, total_steps=12)
    assert torch.equal(model.attn_mask_probs, torch.tensor([0.0, 1.0, 1.0]))
    anneal_attention_masks(model, step=10, total_steps=12)
    assert torch.equal(model.attn_mask_probs, torch.tensor([0.0, 0.0, 0.5]))


def test_train_dry_run_cli_writes_publishable_artifacts(tmp_path: Path) -> None:
    """The dry-run CLI writes checkpoints, metrics, report, and model card offline."""
    from flurseg.cli import main

    _write_dataset(tmp_path)
    config = _config(total_steps=1, eval_every=1)
    config_path = _write_config(tmp_path, config)
    exit_code = main(
        [
            "train",
            "--root",
            str(tmp_path),
            "--config",
            str(config_path),
            "--steps",
            "1",
            "--dry-run",
        ]
    )
    runs = list((tmp_path / "train" / "runs").iterdir())
    assert exit_code == 0
    assert len(runs) == 1
    run = runs[0]
    assert (run / "metrics.csv").is_file()
    assert (run / "best").is_dir()
    assert (run / "last").is_dir()
    report = (run / "report.html").read_text(encoding="utf-8")
    card = (run / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Loss curve" in report
    assert "mIoU curve" in report
    assert "Per-class IoU" in report
    assert "Confusion matrix" in report
    for section in (
        "Classes",
        "Data sources and sessions",
        "Validation metrics",
        "Training configuration",
        "Base checkpoint",
        "Intended use",
        "Limitations",
        "Licence",
    ):
        assert section in card


def test_window_size_must_match_crop_size() -> None:
    """TrainConfig validation rejects mismatched crop size and inference window size."""
    base = _config(total_steps=1).model_dump()
    base["data"]["crop_size"] = 96
    base["inference"]["window_size"] = 64
    with pytest.raises(ValueError, match="inference.window_size must equal data.crop_size"):
        TrainConfig.model_validate(base)


def test_sliding_window_rejects_model_size_mismatch() -> None:
    """Sliding-window scores rejects a window size differing from model image size."""
    base = _config(total_steps=1).model_dump()
    base["data"]["crop_size"] = 64
    base["inference"]["window_size"] = 64
    config = TrainConfig.model_validate(base)
    model = build_model(config, {0: "other", 1: "ground"})
    assert model.config.image_size == 64
    image = torch.zeros((3, 96, 96), dtype=torch.float32)
    with pytest.raises(ValueError, match="window_size"):
        sliding_window_scores(model, image, window_size=96)


def test_warmup_auto_is_ten_percent_capped() -> None:
    """Warmup 'auto' is 10% capped at 1000, while explicit steps are preserved."""
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)

    cfg_300 = _config(total_steps=300)
    cfg_300.training.warmup_steps = "auto"
    sched_300 = build_scheduler(optimizer, cfg_300, 300)
    assert resolve_warmup_steps(cfg_300, 300) == 30
    assert sched_300.warmup_steps == 30
    assert sched_300.lr_lambdas[0](0) == pytest.approx(1 / 30)
    assert sched_300.lr_lambdas[0](29) == pytest.approx(1.0)

    cfg_40k = _config(total_steps=40000)
    cfg_40k.training.warmup_steps = "auto"
    sched_40k = build_scheduler(optimizer, cfg_40k, 40000)
    assert resolve_warmup_steps(cfg_40k, 40000) == 1000
    assert sched_40k.warmup_steps == 1000
    assert sched_40k.lr_lambdas[0](0) == pytest.approx(1 / 1000)
    assert sched_40k.lr_lambdas[0](999) == pytest.approx(1.0)

    cfg_explicit = _config(total_steps=300)
    cfg_explicit.training.warmup_steps = 50
    sched_explicit = build_scheduler(optimizer, cfg_explicit, 300)
    assert resolve_warmup_steps(cfg_explicit, 300) == 50
    assert sched_explicit.warmup_steps == 50
    assert sched_explicit.lr_lambdas[0](0) == pytest.approx(1 / 50)
    assert sched_explicit.lr_lambdas[0](49) == pytest.approx(1.0)


def test_ignore_invariance_holds_under_fixed_seed() -> None:
    """Changing target labels only inside ignore pixels leaves the loss unchanged."""
    from flurseg.train.loss import ignore_aware_loss

    config = _config(total_steps=1).loss
    valid_mask = torch.ones((32, 32), dtype=torch.bool)
    valid_mask[:8, :] = False

    target1 = torch.zeros((1, 32, 32), dtype=torch.float32)
    target1[0, 8:24, 8:24] = 1.0

    target2 = target1.clone()
    target2[0, :8, :] = 1.0

    pred_masks = torch.randn(1, 8, 8, 8)
    pred_classes = torch.randn(1, 8, 3)
    class_labels = [torch.tensor([1], dtype=torch.long)]

    torch.manual_seed(42)
    loss1 = ignore_aware_loss(pred_masks, pred_classes, [target1], class_labels, [valid_mask], config)
    torch.manual_seed(42)
    loss2 = ignore_aware_loss(pred_masks, pred_classes, [target2], class_labels, [valid_mask], config)

    assert torch.allclose(loss1.total, loss2.total, atol=1e-6)
    assert torch.allclose(loss1.mask, loss2.mask, atol=1e-6)
    assert torch.allclose(loss1.dice, loss2.dice, atol=1e-6)
    assert torch.allclose(loss1.classification, loss2.classification, atol=1e-6)


def test_deep_supervision_records_pairs_and_respects_flag(tmp_path: Path) -> None:
    """Deep supervision records num_blocks+1 pairs and deep_supervision=False uses only the final pair."""
    from flurseg.train.engine import train_step
    from flurseg.train.loss import ignore_aware_loss

    _write_dataset(tmp_path)
    config = _config(total_steps=1, eval_every=1)
    config_path = _write_config(tmp_path, config)
    context = setup(tmp_path, config_path, config, resume=False, steps=1, dry_run=False)

    loss = train_step(context)
    assert loss is not None
    assert hasattr(context, "latest_predictions")
    expected_pairs = context.model.config.num_blocks + 1
    assert len(context.latest_predictions) == expected_pairs

    recorded = context.latest_predictions
    final_masks, final_classes = recorded[-1]
    assert context.loader is not None
    batch = next(iter(context.loader))

    cfg_no_deep = config.loss.model_copy(update={"deep_supervision": False})
    torch.manual_seed(123)
    loss_disabled = ignore_aware_loss(
        final_masks,
        final_classes,
        batch["mask_labels"],
        batch["class_labels"],
        batch["valid_masks"],
        cfg_no_deep,
        auxiliary_predictions=recorded[:-1],
    )
    torch.manual_seed(123)
    loss_final_only = ignore_aware_loss(
        final_masks,
        final_classes,
        batch["mask_labels"],
        batch["class_labels"],
        batch["valid_masks"],
        cfg_no_deep,
        auxiliary_predictions=None,
    )
    assert torch.allclose(loss_disabled.total, loss_final_only.total, atol=1e-6)
    assert torch.allclose(loss_disabled.mask, loss_final_only.mask, atol=1e-6)
    assert torch.allclose(loss_disabled.dice, loss_final_only.dice, atol=1e-6)
    assert torch.allclose(loss_disabled.classification, loss_final_only.classification, atol=1e-6)


def test_sampled_points_never_hit_ignore_pixels() -> None:
    """Point sampling for matching and importance sampling never selects ignore pixels."""
    from flurseg.train.loss import sample_importance_points, sample_valid_pixels

    valid = torch.zeros((64, 64), dtype=torch.bool)
    valid[16:48, 16:48] = True

    sampled_pixels = sample_valid_pixels(valid, count=2000)
    assert len(sampled_pixels) == 2000
    assert valid[sampled_pixels[:, 0], sampled_pixels[:, 1]].all()

    pred_masks = torch.randn(2, 1, 16, 16)
    _norm_coords, pix_coords = sample_importance_points(
        pred_masks,
        valid,
        num_points=500,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
    )
    assert pix_coords.shape == (2, 500, 2)
    assert valid[pix_coords[..., 0], pix_coords[..., 1]].all()


def test_sampling_works_with_fewer_valid_pixels_than_num_points() -> None:
    """When a crop has fewer valid pixels than requested points, sampling with replacement succeeds."""
    from flurseg.train.config import LossConfig
    from flurseg.train.loss import ignore_aware_loss, sample_importance_points, sample_valid_pixels

    valid = torch.zeros((32, 32), dtype=torch.bool)
    valid[5, 5] = True
    valid[10, 10] = True
    valid[15, 15] = True
    valid[20, 20] = True

    requested_points = 500
    sampled = sample_valid_pixels(valid, count=requested_points)
    assert sampled.shape == (requested_points, 2)
    assert valid[sampled[:, 0], sampled[:, 1]].all()

    pred_masks = torch.randn(1, 1, 8, 8)
    norm_coords, pix_coords = sample_importance_points(
        pred_masks,
        valid,
        num_points=requested_points,
        oversample_ratio=2.0,
        importance_sample_ratio=0.5,
    )
    assert norm_coords.shape == (1, requested_points, 2)
    assert pix_coords.shape == (1, requested_points, 2)
    assert valid[pix_coords[..., 0], pix_coords[..., 1]].all()

    target = torch.zeros((1, 32, 32), dtype=torch.float32)
    target[0, 5, 5] = 1.0
    cfg = LossConfig(num_points=100)
    loss = ignore_aware_loss(
        torch.randn(1, 8, 8, 8),
        torch.randn(1, 8, 3),
        [target],
        [torch.tensor([1], dtype=torch.long)],
        [valid],
        cfg,
    )
    assert torch.isfinite(loss.total)

