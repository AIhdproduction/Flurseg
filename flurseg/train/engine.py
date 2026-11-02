"""Training, evaluation, checkpoints, and reporting for EoMT-DINOv3."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import random
import shutil
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import EomtDinov3Config, EomtDinov3ForUniversalSegmentation

from flurseg.train.config import TrainConfig
from flurseg.train.data import SegmentationDataset, collate_samples
from flurseg.train.loss import LossValues, ignore_aware_loss
from flurseg.train.predict import sliding_window_scores

MetricsRow = dict[str, float | int | str]
Logger = Callable[[str], None]


@dataclass
class TrainingContext:
    """Keep the mutable state of one training run together."""

    root: Path
    config: TrainConfig
    train_set: SegmentationDataset
    val_set: SegmentationDataset
    classes: list[dict[str, Any]]
    labels: dict[int, str]
    device: torch.device
    model: EomtDinov3ForUniversalSegmentation
    optimizer: AdamW
    scheduler: LambdaLR
    run_dir: Path
    log: Logger
    total_steps: int
    batch_size: int
    accumulation: int
    step: int = 0
    best_miou: float = -1.0
    stale: int = 0
    rows: list[MetricsRow] | None = None
    loader: DataLoader[dict[str, Any]] | None = None
    iterator: Any = None
    latest_result: dict[str, Any] | None = None
    latest_predictions: list[tuple[Tensor, Tensor]] | None = None

    def __post_init__(self) -> None:
        """Initialize the mutable metrics collection after dataclass construction."""
        if self.rows is None:
            self.rows = []


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for reproducible tile ordering."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    config: TrainConfig,
    id2label: dict[int, str],
) -> EomtDinov3ForUniversalSegmentation:
    """Create the configured checkpoint model or an offline tiny random model."""
    labels = {str(key): value for key, value in id2label.items()}
    inverse = {value: key for key, value in id2label.items()}
    if config.model.tiny_random:
        tiny = EomtDinov3Config(
            hidden_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=128,
            image_size=config.data.crop_size,
            patch_size=16,
            num_blocks=2,
            num_upscale_blocks=1,
            num_queries=8,
            num_register_tokens=0,
            num_labels=len(id2label),
            id2label=labels,
            label2id=inverse,
        )
        return EomtDinov3ForUniversalSegmentation(tiny)
    return EomtDinov3ForUniversalSegmentation.from_pretrained(
        config.model.pretrained,
        num_labels=len(id2label),
        id2label=labels,
        label2id=inverse,
        image_size=config.data.crop_size,
        ignore_mismatched_sizes=True,
    )


def _no_decay(name: str) -> bool:
    """Return whether a named parameter must have no weight decay."""
    lowered = name.lower()
    return (
        name.endswith("bias")
        or "norm" in lowered
        or "position" in lowered
        or "register" in lowered
        or ".query" in lowered
    )


def build_optimizer(model: torch.nn.Module, config: TrainConfig) -> AdamW:
    """Create head and layer-wise-decayed backbone parameter groups."""
    layer_count = max(len(getattr(model, "layers", [])), 1)
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        layer = _layer_number(name)
        is_backbone = name.startswith(("embeddings", "layers", "layernorm", "rope_embeddings"))
        if is_backbone:
            depth = layer if layer is not None else 0
            learning_rate = config.training.backbone_lr * (
                config.training.layer_decay ** (layer_count - 1 - depth)
            )
        else:
            learning_rate = config.training.head_lr
        decay = 0.0 if _no_decay(name) else config.training.weight_decay
        groups.setdefault((learning_rate, decay), []).append(parameter)
    parameter_groups = [
        {"params": values, "lr": rate, "weight_decay": decay} for (rate, decay), values in groups.items()
    ]
    return AdamW(parameter_groups, betas=(0.9, 0.999))


def _layer_number(name: str) -> int | None:
    """Extract an EoMT transformer layer number from a parameter name."""
    if not name.startswith("layers."):
        return None
    try:
        return int(name.split(".")[1])
    except (IndexError, ValueError):
        return None


def resolve_total_steps(config: TrainConfig, train_tiles: int) -> int:
    """Resolve explicit steps or the bounded epoch-based default."""
    value = config.training.total_steps
    if value != "auto":
        return int(value)
    effective_batch = config.training.batch_size * config.training.gradient_accumulation
    derived_steps = round(config.training.epochs_target * train_tiles / effective_batch)
    return max(2000, min(40000, derived_steps))


def resolve_warmup_steps(config: TrainConfig, total_steps: int) -> int:
    """Resolve explicit warmup steps or the bounded ten-percent default."""
    if config.training.warmup_steps == "auto":
        return min(1000, max(1, round(0.1 * total_steps)))
    return min(int(config.training.warmup_steps), total_steps)


def build_scheduler(
    optimizer: AdamW,
    config: TrainConfig,
    total_steps: int,
) -> LambdaLR:
    """Build linear warmup followed by polynomial decay."""
    warmup = resolve_warmup_steps(config, total_steps)
    power = config.training.polynomial_power

    def schedule(step: int) -> float:
        """Return the learning-rate multiplier for one completed optimization step."""
        if warmup and step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return max(0.0, (1.0 - progress) ** power)

    scheduler = LambdaLR(optimizer, schedule)
    scheduler.warmup_steps = warmup  # type: ignore[attr-defined]
    return scheduler


def anneal_attention_masks(model: torch.nn.Module, step: int, total_steps: int) -> None:
    """Apply EoMT's block-wise, staggered linear mask-disable schedule.

    Transformers exposes one ``attn_mask_probs`` value for each EoMT block. A
    value of one disables query attention masks, while zero enables them. Each
    block receives an equal, non-overlapping step range over the full run and
    linearly transitions from one to zero in its own range. Models exposing a
    scalar retain the original whole-run linear schedule.
    """
    probabilities = getattr(model, "attn_mask_probs", None)
    if not isinstance(probabilities, torch.Tensor):
        return
    count = probabilities.numel()
    if count <= 1:
        probabilities.fill_(max(0.0, 1.0 - step / max(total_steps - 1, 1)))
        return
    values = torch.ones_like(probabilities)
    for index in range(count):
        start = total_steps * index / count
        end = total_steps * (index + 1) / count
        progress = (step - start) / max(end - start, 1.0)
        values.flatten()[index] = 1.0 - min(max(progress, 0.0), 1.0)
    probabilities.copy_(values)


def confusion_matrix(
    prediction: np.ndarray,
    target: np.ndarray,
    classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Compute a class confusion matrix while excluding ignored target pixels."""
    valid = (
        (target != ignore_index)
        & (target >= 0)
        & (target < classes)
        & (prediction >= 0)
        & (prediction < classes)
    )
    encoded = classes * target[valid].astype(np.int64) + prediction[valid].astype(np.int64)
    return np.bincount(encoded, minlength=classes * classes).reshape(classes, classes)


def metrics_from_confusion(matrix: np.ndarray) -> tuple[list[float], float, float]:
    """Return per-class IoU, mean IoU, and pixel accuracy."""
    diagonal = np.diag(matrix).astype(float)
    union = matrix.sum(0) + matrix.sum(1) - diagonal
    iou = np.divide(
        diagonal,
        union,
        out=np.full_like(diagonal, np.nan),
        where=union > 0,
    )
    mean_iou = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
    pixel_accuracy = float(diagonal.sum() / max(matrix.sum(), 1))
    return iou.tolist(), mean_iou, pixel_accuracy


def evaluate(
    model: torch.nn.Module,
    dataset: SegmentationDataset,
    config: TrainConfig,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    """Run sliding-window validation and compute semantic metrics."""
    del device
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for index in range(len(dataset)):
        sample = dataset[index]
        scores = sliding_window_scores(
            model,
            sample["pixel_values"],
            config.inference.window_size,
            config.inference.stride,
        )
        prediction = scores.argmax(0).cpu().numpy()
        target = np.zeros(prediction.shape, dtype=np.uint8)
        for target_mask, class_id in zip(
            sample["mask_labels"],
            sample["class_labels"],
            strict=True,
        ):
            target[target_mask.numpy().astype(bool)] = int(class_id)
        valid = sample["valid_mask"].numpy()
        target[~valid] = 255
        matrix += confusion_matrix(prediction, target, num_classes)
    iou, miou, accuracy = metrics_from_confusion(matrix)
    return {
        "iou": iou,
        "miou": miou,
        "pixel_accuracy": accuracy,
        "confusion": matrix,
    }


def _capture_rng_state() -> dict[str, Any]:
    """Capture all RNG state needed to reproduce a resumed run."""
    state: dict[str, Any] = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    """Restore saved Python, NumPy, Torch, and CUDA RNG state."""
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


def _save_checkpoint(context: TrainingContext, path: Path) -> None:
    """Save model, optimizer, scheduler, training state, and RNG state."""
    path.mkdir(parents=True, exist_ok=True)
    context.model.save_pretrained(path)
    preprocessor = {
        "image_mean": context.config.model.image_mean,
        "image_std": context.config.model.image_std,
    }
    (path / "preprocessor_config.json").write_text(
        json.dumps(preprocessor),
        encoding="utf-8",
    )
    state = {
        "model": context.model.state_dict(),
        "step": context.step,
        "best_miou": context.best_miou,
        "stale": context.stale,
        "optimizer": context.optimizer.state_dict(),
        "scheduler": context.scheduler.state_dict(),
        "rows": context.rows,
        "batch_size": context.batch_size,
        "accumulation": context.accumulation,
        **_capture_rng_state(),
    }
    torch.save(state, path / "training_state.pt")


def _load_checkpoint(context: TrainingContext, checkpoint: Path) -> None:
    """Restore model, optimizer, scheduler, counters, rows, and RNG state."""
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    model_state = state.get("model")
    if model_state is None:
        restored_model = EomtDinov3ForUniversalSegmentation.from_pretrained(checkpoint)
        model_state = restored_model.state_dict()
    context.model.load_state_dict(model_state)
    context.model.to(context.device)
    context.optimizer.load_state_dict(state["optimizer"])
    context.scheduler.load_state_dict(state["scheduler"])
    context.step = int(state["step"])
    context.best_miou = float(state["best_miou"])
    context.stale = int(state.get("stale", 0))
    context.rows = list(state.get("rows", []))
    context.batch_size = int(state.get("batch_size", context.batch_size))
    context.accumulation = int(state.get("accumulation", context.accumulation))
    _restore_rng_state(state)


def _create_run_dir(root: Path) -> Path:
    """Create a collision-free timestamped runtime directory."""
    runs_dir = root / "train" / "runs"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for suffix in range(1000):
        name = stamp if suffix == 0 else f"{stamp}-{suffix:02d}"
        candidate = runs_dir / name
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("Could not create a unique training run directory")


def _make_loader(context: TrainingContext) -> None:
    """Build a loader after setup, resume, or a recoverable OOM."""
    context.loader = DataLoader(
        context.train_set,
        batch_size=context.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
    context.iterator = iter(context.loader)


def _next_batch(context: TrainingContext) -> dict[str, Any]:
    """Return the next batch and restart the loader at its end."""
    assert context.loader is not None
    try:
        return next(context.iterator)
    except StopIteration:
        context.train_set.set_epoch(context.train_set.epoch + 1)
        _make_loader(context)
        return next(context.iterator)


def deterministic_holdout_ids(tile_ids: list[str]) -> set[str]:
    """Select a stable ten-percent validation holdout from train tile IDs."""
    count = max(1, math.ceil(len(tile_ids) * 0.10))
    ordered = sorted(
        tile_ids,
        key=lambda tile_id: (hashlib.sha256(tile_id.encode("utf-8")).hexdigest(), tile_id),
    )
    return set(ordered[:count])


def setup(
    root: Path,
    config_path: Path,
    config: TrainConfig,
    resume: bool,
    steps: int | None,
    dry_run: bool,
) -> TrainingContext:
    """Create datasets, model, optimizer, run directory, and resume state."""
    seed_everything(config.training.seed)
    dataset_root = root / "train" / "dataset"
    train_set = SegmentationDataset(
        dataset_root,
        "train",
        config.data,
        config.model,
        training=True,
        sampling_config=config.sampling,
        sampling_seed=config.training.seed,
    )
    val_empty = bool(train_set.manifest.get("val_empty", False))
    if val_empty:
        holdout_ids = deterministic_holdout_ids([path.stem for path in train_set.images])
        if len(holdout_ids) >= len(train_set):
            raise ValueError("Internal validation holdout leaves no training tiles")
        train_ids = {path.stem for path in train_set.images} - holdout_ids
        train_set = SegmentationDataset(
            dataset_root,
            "train",
            config.data,
            config.model,
            training=True,
            tile_ids=train_ids,
            sampling_config=config.sampling,
            sampling_seed=config.training.seed,
        )
        val_set = SegmentationDataset(
            dataset_root,
            "train",
            config.data,
            config.model,
            training=False,
            tile_ids=holdout_ids,
        )
    else:
        val_set = SegmentationDataset(
            dataset_root,
            "val",
            config.data,
            config.model,
            training=False,
        )
    classes = [item for item in train_set.manifest["classes"] if item["id"] != 255]
    labels = {int(item["id"]): str(item["name"]) for item in classes}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, labels).to(device)
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    total_steps = steps or resolve_total_steps(config, len(train_set))
    if dry_run:
        total_steps = min(total_steps, 20)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, total_steps)
    run_dir = _create_run_dir(root)
    log_path = run_dir / "train.log"

    def log(message: str) -> None:
        """Print and persist one training progress message."""
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{message}\n")

    if val_empty:
        log(
            "WARNING: validation split is empty; using a deterministic internal holdout of "
            f"{len(val_set)} train tiles for evaluation."
        )
    for line in train_set.sampling_summary_lines():
        log(line)

    context = TrainingContext(
        root=root,
        config=config,
        train_set=train_set,
        val_set=val_set,
        classes=classes,
        labels=labels,
        device=device,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        run_dir=run_dir,
        log=log,
        total_steps=total_steps,
        batch_size=config.training.batch_size,
        accumulation=config.training.gradient_accumulation,
    )
    shutil.copy2(config_path, run_dir / "config.yaml")
    shutil.copy2(dataset_root / "dataset.json", run_dir / "dataset.json")
    if resume:
        checkpoints = sorted((root / "train" / "runs").glob("*/last/training_state.pt"))
        checkpoints = [path for path in checkpoints if path.parent.parent != run_dir]
        if not checkpoints:
            raise ValueError("No last checkpoint is available to resume")
        _load_checkpoint(context, checkpoints[-1].parent)
    _make_loader(context)
    return context


@contextmanager
def record_predictions(model: Any) -> Iterator[list[tuple[Tensor, Tensor]]]:
    """Record every predict call during model execution while in training mode."""
    recorded: list[tuple[Tensor, Tensor]] = []
    original = model.predict

    def predict_wrapper(*args: Any, **kwargs: Any) -> tuple[Tensor, Tensor]:
        pair = original(*args, **kwargs)
        recorded.append(pair)
        return pair

    try:
        if getattr(model, "training", False):
            model.predict = predict_wrapper
        yield recorded
    finally:
        model.predict = original


def train_step(context: TrainingContext) -> LossValues | None:
    """Perform one optimizer step, recovering once from a CUDA OOM if possible."""
    assert context.loader is not None
    context.model.train()
    anneal_attention_masks(context.model, context.step, context.total_steps)
    context.optimizer.zero_grad(set_to_none=True)
    current: LossValues | None = None
    for _ in range(context.accumulation):
        batch = _next_batch(context)
        pixels = batch["pixel_values"].to(context.device)
        try:
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.bfloat16,
                enabled=context.device.type == "cuda",
            ):
                with record_predictions(context.model) as recorded_predictions:
                    output = context.model(pixel_values=pixels)

                context.latest_predictions = recorded_predictions
                auxiliary = (
                    recorded_predictions[:-1]
                    if (context.config.loss.deep_supervision and len(recorded_predictions) > 1)
                    else None
                )
                current = ignore_aware_loss(
                    output.masks_queries_logits,
                    output.class_queries_logits,
                    batch["mask_labels"],
                    batch["class_labels"],
                    batch["valid_masks"],
                    context.config.loss,
                    auxiliary_predictions=auxiliary,
                )
                (current.total / context.accumulation).backward()
        except torch.cuda.OutOfMemoryError:
            context.optimizer.zero_grad(set_to_none=True)
            if context.batch_size <= 1:
                raise
            context.batch_size //= 2
            context.accumulation *= 2
            if context.device.type == "cuda":
                torch.cuda.empty_cache()
            _make_loader(context)
            context.log(
                "CUDA OOM: reducing physical batch to "
                f"{context.batch_size} and increasing accumulation to {context.accumulation}"
            )
            return None
    torch.nn.utils.clip_grad_norm_(
        context.model.parameters(),
        context.config.training.max_grad_norm,
    )
    context.optimizer.step()
    context.scheduler.step()
    context.step += 1
    return current


def _evaluation_due(context: TrainingContext, dry_run: bool) -> bool:
    """Return whether validation is due after the current optimization step."""
    if dry_run:
        interval = max(1, min(200, context.total_steps))
    elif context.config.training.eval_every == "auto":
        physical_batch = max(context.batch_size * context.accumulation, 1)
        interval = max(200, round(2 * math.ceil(len(context.train_set) / physical_batch)))
    else:
        interval = int(context.config.training.eval_every)
    return context.step % interval == 0 or context.step == context.total_steps


def evaluate_and_checkpoint(context: TrainingContext) -> bool:
    """Evaluate, persist `last` and `best`, and return whether early stopping fires."""
    result = evaluate(
        context.model,
        context.val_set,
        context.config,
        context.device,
        len(context.labels),
    )
    context.latest_result = result
    if result["miou"] > context.best_miou:
        context.best_miou = float(result["miou"])
        context.stale = 0
        improved = True
    else:
        context.stale += 1
        improved = False
    _save_checkpoint(context, context.run_dir / "last")
    if improved:
        _save_checkpoint(context, context.run_dir / "best")
    context.log(f"validation mIoU {result['miou']:.4f}; best {context.best_miou:.4f}; stale {context.stale}")
    return context.stale >= context.config.training.patience


def _metric_row(context: TrainingContext, loss: LossValues) -> MetricsRow:
    """Create the CSV row for one completed optimizer step."""
    return {
        "step": context.step,
        "loss": float(loss.total.detach()),
        "loss_mask": float(loss.mask.detach()),
        "loss_dice": float(loss.dice.detach()),
        "loss_class": float(loss.classification.detach()),
        "lr": float(max(group["lr"] for group in context.optimizer.param_groups)),
        "val_miou": "",
    }


def _write_metrics(run_dir: Path, rows: list[MetricsRow]) -> None:
    """Write contiguous metrics rows to the run CSV."""
    fields = ["step", "loss", "loss_mask", "loss_dice", "loss_class", "lr", "val_miou"]
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _polyline(values: list[float]) -> str:
    """Return SVG points for a compact fixed-size trend line."""
    if not values:
        return ""
    maximum = max(values)
    minimum = min(values)
    points = []
    for index, value in enumerate(values):
        x_value = index * 300 / max(len(values) - 1, 1)
        y_value = 110 - 100 * (value - minimum) / max(maximum - minimum, 1e-9)
        points.append(f"{x_value:.1f},{y_value:.1f}")
    return " ".join(points)


def _write_report(
    run_dir: Path,
    rows: list[MetricsRow],
    result: dict[str, Any],
    classes: list[dict[str, Any]],
    dataset_version: int,
    elapsed: float,
) -> None:
    """Create a self-contained report with inline SVG and HTML diagnostics."""
    losses = [float(row["loss"]) for row in rows if row.get("loss") != ""]
    miou = [float(row["val_miou"]) for row in rows if row.get("val_miou") not in (None, "")]
    iou_rows = []
    for item, value in zip(classes, result.get("iou", []), strict=False):
        iou_rows.append(
            "<tr>"
            f"<td>{int(item['id'])}</td>"
            f"<td>{html.escape(str(item['name']))}</td>"
            f"<td>{html.escape(str(item.get('color', '')))}</td>"
            f"<td>{float(value):.4f}</td>"
            "</tr>"
        )
    matrix = np.asarray(result.get("confusion", np.zeros((0, 0), dtype=int)))
    headers = "".join(f"<th>{index}</th>" for index in range(matrix.shape[1]))
    matrix_rows = "".join(
        f"<tr><th>{row_index}</th>" + "".join(f"<td>{int(value)}</td>" for value in row) + "</tr>"
        for row_index, row in enumerate(matrix)
    )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Flurseg training report</title>
<style>
body{{font-family:sans-serif;max-width:900px;margin:2rem auto}}
table,td,th{{border:1px solid #bbb;border-collapse:collapse;padding:.35rem}}
svg{{border:1px solid #bbb}}
</style>
</head><body><h1>Flurseg training report</h1>
<p>Dataset v{dataset_version}; total time {elapsed:.1f}s.</p>
<h2>Loss curve</h2>
<svg width="320" height="120" viewBox="0 0 320 120">
<polyline fill="none" stroke="#1769aa" points="{_polyline(losses)}"/></svg>
<h2>mIoU curve</h2>
<svg width="320" height="120" viewBox="0 0 320 120">
<polyline fill="none" stroke="#188038" points="{_polyline(miou)}"/></svg>
<h2>Validation summary</h2>
<p>mIoU {float(result.get("miou", 0.0)):.4f};
pixel accuracy {float(result.get("pixel_accuracy", 0.0)):.4f}</p>
<h2>Per-class IoU</h2>
<table><thead><tr><th>ID</th><th>Class</th><th>Colour</th><th>IoU</th></tr></thead>
<tbody>{"".join(iou_rows)}</tbody></table>
<h2>Confusion matrix</h2>
<table><thead><tr><th>Target / prediction</th>{headers}</tr></thead>
<tbody>{matrix_rows}</tbody></table>
</body></html>"""
    (run_dir / "report.html").write_text(document, encoding="utf-8")


def _dataset_summary(manifest: dict[str, Any]) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Summarize source, session, and split tile counts from a dataset manifest."""
    sources: dict[str, int] = {}
    sessions: dict[str, set[str]] = {}
    tile_counts: dict[str, int] = {}
    for metadata in manifest.get("tile_metadata", {}).values():
        source = str(metadata.get("source", "unknown"))
        session = str(metadata.get("session", "unknown"))
        split = str(metadata.get("split", "unknown"))
        sources[source] = sources.get(source, 0) + 1
        sessions.setdefault(source, set()).add(session)
        tile_counts[split] = tile_counts.get(split, 0) + 1
    if tile_counts:
        tile_counts["total"] = sum(tile_counts.values())
    else:
        manifest_counts = manifest.get("tile_counts", {})
        by_split = manifest_counts.get("by_split", manifest_counts)
        tile_counts = {str(split): int(count) for split, count in by_split.items() if isinstance(count, int)}
    return sources, {key: len(value) for key, value in sessions.items()}, tile_counts


def _write_model_card(context: TrainingContext, result: dict[str, Any]) -> None:
    """Write the publishable provenance, metrics, and licence model card."""
    sources, sessions, tile_counts = _dataset_summary(context.train_set.manifest)
    class_rows = "\n".join(
        f"| {int(item['id'])} | {item['name']} | {item.get('color', '')} |" for item in context.classes
    )
    metric_rows = "\n".join(
        f"| {int(item['id'])} | {item['name']} | {float(value):.4f} |"
        for item, value in zip(context.classes, result.get("iou", []), strict=False)
    )
    source_rows = "\n".join(
        f"| {source} | {sources[source]} | {sessions.get(source, 0)} |" for source in sorted(sources)
    )
    split_rows = "\n".join(f"| {split} | {count} |" for split, count in sorted(tile_counts.items()))
    config_summary = json.dumps(context.config.model_dump(mode="json"), indent=2)
    card = f"""# Flurseg EoMT-DINOv3

## Base checkpoint

{context.config.model.pretrained}

## Dataset

Dataset version: {int(context.train_set.manifest.get("dataset_version", 0))}

### Classes

| ID | Class | Colour |
| --- | --- | --- |
{class_rows}

### Data sources and sessions

| Source | Tiles | Sessions |
| --- | --- | --- |
{source_rows}

### Tile counts

| Split | Tiles |
| --- | --- |
{split_rows}

## Validation metrics

Final validation mIoU: {float(result.get("miou", 0.0)):.4f}

| ID | Class | IoU |
| --- | --- | --- |
{metric_rows}

## Training configuration

```json
{config_summary}
```

## Intended use

This model is intended for local semantic segmentation of the documented Flurseg
tile classes. Predictions require human review before operational, regulatory,
or safety-critical decisions.

## Limitations

Performance is limited by the recorded data sources, sessions, labels, crop
scale, and class balance. It may not generalize to unseen sensors, seasons,
locations, illumination, or image orientations.

## Licence

The DINOv3 License applies to derived weights. A copy of the DINOv3 License
must accompany any published or redistributed weights. Uses prohibited by the
DINOv3 License are prohibited for these derived weights; consult that licence
before deployment or redistribution.
"""
    (context.run_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")


def finalize(context: TrainingContext, elapsed: float) -> Path:
    """Write metrics, final report, and model card, then return the run path."""
    if context.latest_result is None:
        context.latest_result = evaluate(
            context.model,
            context.val_set,
            context.config,
            context.device,
            len(context.labels),
        )
    _write_metrics(context.run_dir, context.rows or [])
    _write_report(
        context.run_dir,
        context.rows or [],
        context.latest_result,
        context.classes,
        int(context.train_set.manifest.get("dataset_version", 0)),
        elapsed,
    )
    _write_model_card(context, context.latest_result)
    return context.run_dir


def run_training(
    root: Path,
    config_path: Path,
    config: TrainConfig,
    resume: bool = False,
    steps: int | None = None,
    dry_run: bool = False,
) -> Path:
    """Run local training and return the output run directory."""
    context = setup(root, config_path, config, resume, steps, dry_run)
    began = time.monotonic()
    while context.step < context.total_steps:
        loss = train_step(context)
        if loss is None:
            continue
        row = _metric_row(context, loss)
        assert context.rows is not None
        context.rows.append(row)
        elapsed_per_step = (time.monotonic() - began) / max(context.step, 1)
        remaining = elapsed_per_step * (context.total_steps - context.step)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3 if context.device.type == "cuda" else 0.0
        print(
            f"\rstep {context.step}/{context.total_steps} epoch {context.train_set.epoch} "
            f"loss {row['loss']:.4f} lr {row['lr']:.2e} {elapsed_per_step:.2f}s/step "
            f"ETA {remaining:.0f}s VRAM {peak_memory:.2f}GiB",
            end="",
            flush=True,
        )
        if _evaluation_due(context, dry_run):
            result = evaluate_and_checkpoint(context)
            row["val_miou"] = float(context.latest_result["miou"])
            _save_checkpoint(context, context.run_dir / "last")
            if result:
                break
    print()
    return finalize(context, time.monotonic() - began)
