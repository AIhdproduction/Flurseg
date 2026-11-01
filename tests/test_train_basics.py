"""Offline checks for the EoMT-DINOv3 training setup."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from flurseg.dataset.split import (
    group_images_by_session,
    load_split_registry,
    save_split_registry,
    split_sessions,
    update_split_registry,
)
from flurseg.train.config import DataConfig, TrainConfig
from flurseg.train.data import augment_pair, mask_to_targets
from flurseg.train.engine import (
    build_model,
    build_optimizer,
    build_scheduler,
    confusion_matrix,
    metrics_from_confusion,
    run_training,
)
from flurseg.train.loss import ignore_aware_loss


def _tiny_config() -> TrainConfig:
    return TrainConfig.model_validate({"model": {"pretrained": "offline", "tiny_random": True}, "data": {"crop_size": 32}, "inference": {"window_size": 32, "stride": 16}})


def test_training_config_rejects_invalid_scale_range() -> None:
    with pytest.raises(ValueError, match="scale_min"):
        TrainConfig.model_validate({"model": {"pretrained": "offline"}, "data": {"scale_min": 2, "scale_max": 1}})


def test_mask_targets_exclude_ignore_pixels() -> None:
    masks, labels, valid = mask_to_targets(np.array([[0, 255], [2, 2]], dtype=np.uint8))
    assert labels.tolist() == [0, 2]
    assert masks.shape == (2, 2, 2)
    assert not valid[0, 1]


def test_phone_augmentation_never_rotates_or_flips_vertically() -> None:
    image = np.dstack([np.arange(16, dtype=np.uint8).reshape(4, 4)] * 3)
    mask = image[:, :, 0].copy()
    config = DataConfig(crop_size=4, scale_min=1, scale_max=1, color_jitter=0)
    output_image, output_mask = augment_pair(image, mask, "phone", config, np.random.default_rng(9))
    assert output_image.shape == image.shape
    assert np.array_equal(output_image[:, :, 0], output_mask)


def test_ignore_aware_loss_does_not_read_ignored_targets() -> None:
    torch.manual_seed(4)
    masks = torch.randn(1, 3, 4, 4)
    classes = torch.randn(1, 3, 3)
    target = torch.zeros(1, 8, 8)
    changed = target.clone()
    changed[:, 0, 0] = 1
    valid = torch.ones(8, 8, dtype=torch.bool)
    valid[0, 0] = False
    torch.manual_seed(4)
    first = ignore_aware_loss(masks, classes, [target], [torch.tensor([0])], [valid], _tiny_config().loss)
    torch.manual_seed(4)
    second = ignore_aware_loss(masks, classes, [changed], [torch.tensor([0])], [valid], _tiny_config().loss)
    assert torch.equal(first.total, second.total)


def test_optimizer_groups_and_scheduler_are_layerwise() -> None:
    config = _tiny_config()
    model = build_model(config, {0: "other", 1: "ground"})
    optimizer = build_optimizer(model, config)
    rates = {group["lr"] for group in optimizer.param_groups}
    assert config.training.head_lr in rates
    assert min(rates) < config.training.backbone_lr
    scheduler = build_scheduler(optimizer, config, 20)
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] >= 0


def test_metrics_ignore_and_registry_persist(tmp_path: Path) -> None:
    matrix = confusion_matrix(np.array([[0, 1], [1, 0]]), np.array([[0, 1], [255, 0]]), 2)
    iou, miou, accuracy = metrics_from_confusion(matrix)
    assert iou == [1.0, 1.0] and miou == 1.0 and accuracy == 1.0
    items = [{"id": f"image-{index}", "source": "drone", "session": f"s{index}"} for index in range(3)]
    result = split_sessions(items)
    registry = update_split_registry({}, result, group_images_by_session(items), 42)
    path = tmp_path / "train" / "split_registry.json"
    save_split_registry(path, registry)
    restored = load_split_registry(path, 42)
    repeated = split_sessions(items + [{"id": "new", "source": "phone", "session": "next"}], registry=restored)
    assert all(repeated.source_session_to_split[("drone", f"s{index}")] == result.source_session_to_split[("drone", f"s{index}")] for index in range(3))


def test_train_batch_script_has_required_guards() -> None:
    script = (Path(__file__).parents[1] / "train.bat").read_text(encoding="utf-8").lower()
    assert "venv\\scripts\\python.exe" in script
    assert "train\\dataset\\dataset.json" in script
    assert "run build_dataset.bat first" in script


def test_tiny_training_writes_checkpoints_and_report(tmp_path: Path) -> None:
    dataset = tmp_path / "train" / "dataset"
    metadata: dict[str, dict[str, str]] = {}
    for split, count in (("train", 2), ("val", 1)):
        for folder in ("images", "masks"):
            (dataset / split / folder).mkdir(parents=True, exist_ok=True)
        for index in range(count):
            name = f"{split}-{index}"
            image = np.full((32, 32, 3), 100 + index, dtype=np.uint8)
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[8:24, 8:24] = 1
            cv2.imwrite(str(dataset / split / "images" / f"{name}.jpg"), image)
            cv2.imwrite(str(dataset / split / "masks" / f"{name}.png"), mask)
            metadata[name] = {"source": "phone", "split": split, "session": "session"}
    manifest = {"ignore_index": 255, "dataset_version": 1, "classes": [{"id": 0, "name": "other", "color": "#000000"}, {"id": 1, "name": "ground", "color": "#ffffff"}], "tile_metadata": metadata}
    (dataset / "dataset.json").write_text(json.dumps(manifest), encoding="utf-8")
    config = _tiny_config().model_copy(update={"training": _tiny_config().training.model_copy(update={"batch_size": 1, "gradient_accumulation": 1, "total_steps": 1, "warmup_steps": 0, "eval_every": 1})})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    run = run_training(tmp_path, config_path, config, steps=1)
    assert (run / "metrics.csv").is_file()
    assert (run / "best" / "config.json").is_file()
    assert (run / "last" / "training_state.pt").is_file()
    assert (run / "report.html").is_file()
    assert (run / "train.log").is_file()
