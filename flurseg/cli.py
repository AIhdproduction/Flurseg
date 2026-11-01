"""Command-line entry points with lazy optional dependency checks."""

from __future__ import annotations

import argparse
import importlib
import io
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from flurseg.paths import RepoPaths

Importer = Callable[[str], ModuleType]


def _sam3_checkpoint(paths: RepoPaths, cfg: Any) -> Path:
    """Resolve a configured checkpoint, retaining the main repository as a bench fallback."""
    configured = Path(cfg.pipeline.sam3.checkpoint_path)
    if configured.is_absolute():
        return configured
    bench_candidate = paths.root / configured
    if bench_candidate.is_file():
        return bench_candidate
    return Path(__file__).resolve().parents[1] / configured


def build_parser() -> argparse.ArgumentParser:
    """Build the extensible Flurseg command-line parser."""
    parser = argparse.ArgumentParser(prog="flurseg", description="Local segmentation dataset preparation.")
    parser.add_argument("--root", type=Path, default=argparse.SUPPRESS, help="Repository root directory.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    check_environment = subcommands.add_parser("check-env", help="Report required package and device status.")
    check_environment.add_argument(
        "--root",
        type=Path,
        default=argparse.SUPPRESS,
        help="Repository root directory.",
    )
    prelabel = subcommands.add_parser("prelabel", help="Create pseudo labels for new input images.")
    prelabel.add_argument("--backend", choices=("sam3", "fake"), default="sam3")
    prelabel.add_argument("--limit", type=int, default=None)
    prelabel.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    reprocess = subcommands.add_parser("reprocess", help="Reapply post-processing from stored scores.")
    reprocess.add_argument("--ids", nargs="*", default=None)
    reprocess.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    smoke = subcommands.add_parser("gpu-smoke", help="Run the owner-approved SAM 3 GPU smoke check.")
    smoke.add_argument("--image", type=Path, required=True)
    smoke.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    review = subcommands.add_parser("review", help="Open the local mask review application.")
    review.add_argument("--port", type=int, default=None)
    review.add_argument("--no-browser", action="store_true")
    review.add_argument("--backend", choices=("sam3", "fake", "none"), default="sam3")
    review.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    build_dataset_cmd = subcommands.add_parser(
        "build-dataset", help="Split pool images, extract tiles, and export dataset manifest."
    )
    build_dataset_cmd.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    train_cmd = subcommands.add_parser("train", help="Train EoMT-DINOv3 on the built dataset.")
    train_cmd.add_argument("--config", type=Path, default=Path("config/train.yaml"))
    train_cmd.add_argument("--resume", action="store_true")
    train_cmd.add_argument("--steps", type=int, default=None)
    train_cmd.add_argument("--dry-run", action="store_true")
    train_cmd.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    predict_cmd = subcommands.add_parser("predict", help="Predict a semantic mask using a saved training run.")
    predict_cmd.add_argument("--run", required=True, type=Path)
    predict_cmd.add_argument("--image", required=True, type=Path)
    predict_cmd.add_argument("--out", required=True, type=Path)
    predict_cmd.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    report_cmd = subcommands.add_parser("report", help="Render self-contained HTML quality report.")
    report_cmd.add_argument(
        "--open", action="store_true", help="Open the generated HTML report in the default browser."
    )
    report_cmd.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    fetch_samples_cmd = subcommands.add_parser(
        "fetch-samples", help="Fetch public sample dataset items and write lockfile."
    )
    fetch_samples_cmd.add_argument(
        "--dry-run", action="store_true", help="Plan sample downloads without fetching files."
    )
    fetch_samples_cmd.add_argument(
        "--only", choices=("ortho", "drone", "phone"), default=None, help="Filter sample source."
    )
    fetch_samples_cmd.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt before downloading."
    )
    fetch_samples_cmd.add_argument("--root", type=Path, default=argparse.SUPPRESS)
    return parser


def _load_module(name: str, importer: Importer) -> tuple[ModuleType | None, str | None]:
    try:
        return importer(name), None
    except Exception as error:  # noqa: BLE001 - optional dependencies can fail during import in many ways.
        return None, f"{type(error).__name__}: {error}"


def check_environment(paths: RepoPaths, importer: Importer = importlib.import_module) -> int:
    """Print environment state and return non-zero only for missing hard requirements."""
    has_error = False
    print(f"Python: {sys.version.split()[0]}")

    torch, torch_error = _load_module("torch", importer)
    if torch is None:
        has_error = True
        print(f"ERROR: torch import failed ({torch_error})")
    else:
        print(f"Torch: {getattr(torch, '__version__', 'unknown')}")
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception as error:  # noqa: BLE001 - driver initialization errors are reported to the user.
            cuda_available = False
            print(f"WARNING: CUDA status could not be queried ({type(error).__name__}: {error})")
        if cuda_available:
            try:
                device_name = torch.cuda.get_device_name(0)
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                print(f"CUDA: available ({device_name}, {total_vram_gb:.1f} GiB)")
            except Exception as error:  # noqa: BLE001 - device metadata is optional diagnostic information.
                print(f"WARNING: CUDA device details could not be queried ({type(error).__name__}: {error})")
        else:
            print("WARNING: CUDA is unavailable; the fake backend can still run tests.")

    ultralytics, ultralytics_error = _load_module("ultralytics", importer)
    if ultralytics is None:
        has_error = True
        print(f"ERROR: ultralytics import failed ({ultralytics_error})")
    else:
        print(f"Ultralytics: {getattr(ultralytics, '__version__', 'unknown')}")
        sam_module, sam_error = _load_module("ultralytics.models.sam", importer)
        if sam_module is None or not hasattr(sam_module, "SAM3SemanticPredictor"):
            has_error = True
            details = sam_error or "SAM3SemanticPredictor is not exported"
            print(f"ERROR: SAM3SemanticPredictor import failed ({details})")
        else:
            print("SAM3SemanticPredictor: available")

    for module_name in ("timm", "clip", "pillow_heif", "rasterio"):
        module, error = _load_module(module_name, importer)
        if module is None:
            has_error = True
            print(f"ERROR: {module_name} import failed ({error})")
        else:
            print(f"{module_name}: available")

    if paths.sam3_checkpoint.is_file():
        size_mib = paths.sam3_checkpoint.stat().st_size / 1024**2
        print(f"SAM 3 checkpoint: found ({size_mib:.1f} MiB)")
    else:
        print(f"WARNING: SAM 3 checkpoint is missing at {paths.sam3_checkpoint}.")

    return 1 if has_error else 0


def _format_size(size_bytes: int | None, is_estimate: bool = False) -> str:
    """Format a byte count into a human-readable size string."""
    if size_bytes is None:
        return "unknown"
    prefix = "~" if is_estimate else ""
    if size_bytes < 1024:
        return f"{prefix}{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{prefix}{size_bytes / 1024:.1f} KB"
    return f"{prefix}{size_bytes / (1024 * 1024):.1f} MB"


def _get_record_size(record: dict[str, Any]) -> tuple[int, bool]:
    """Extract effective size in bytes and estimate flag for a sample record."""
    is_estimate = bool(record.get("is_estimate"))
    for key in ("estimate_size", "estimate", "estimate_bytes", "window_size"):
        val = record.get(key)
        if isinstance(val, (int, float)):
            return int(val), True
    size_val = record.get("size")
    if isinstance(size_val, (int, float)):
        return int(size_val), is_estimate
    remote_size = record.get("remote_size")
    if isinstance(remote_size, (int, float)):
        return int(remote_size), is_estimate
    return 0, is_estimate


def _is_record_skipped_or_existing(record: dict[str, Any]) -> bool:
    """Check if record is already present locally or skipped."""
    status_val = str(record.get("status") or "").strip().lower()
    if status_val in ("exists", "skipped"):
        return True
    if status_val == "dry_run" or not status_val:
        dest = record.get("path")
        if dest is not None:
            try:
                dest_path = Path(dest) if isinstance(dest, (str, Path)) else dest
                if hasattr(dest_path, "is_file") and dest_path.is_file():
                    return True
            except (OSError, ValueError):
                pass
    return False


def _compute_samples_total(records: list[dict[str, Any]]) -> tuple[int, float]:
    """Compute file count and total MB for pending sample downloads."""
    total_bytes = 0
    n_files = 0
    for record in records:
        if _is_record_skipped_or_existing(record):
            continue
        size_bytes, _ = _get_record_size(record)
        total_bytes += size_bytes
        n_files += 1
    total_mb = total_bytes / (1024 * 1024)
    return n_files, total_mb


def _print_samples_table(records: list[dict[str, Any]]) -> None:
    """Display sample information formatted as an aligned terminal table."""
    headers = ("Source", "Session", "File", "Size", "Status", "Licence")
    rows: list[tuple[str, str, str, str, str, str]] = []
    max_filename_width = 60
    has_estimate = False

    for record in records:
        source = str(record.get("source", ""))
        session = str(record.get("session", ""))
        raw_filename = str(record.get("filename") or record.get("file") or "")
        if len(raw_filename) > max_filename_width:
            filename = raw_filename[: max_filename_width - 3] + "..."
        else:
            filename = raw_filename

        size_val, is_estimate = _get_record_size(record)
        if is_estimate:
            has_estimate = True
        status_val = str(record.get("status") or "")
        size_str = _format_size(size_val, is_estimate=is_estimate)
        if status_val == "skipped":
            display_status = "skipped (exists)"
        elif status_val == "failed":
            display_status = f"failed: {record.get('error') or 'unknown error'}"
        elif status_val == "dry_run":
            dest = record.get("path")
            if dest and hasattr(dest, "is_file") and dest.is_file():
                display_status = "skipped (exists)"
            else:
                display_status = "planned"
        else:
            display_status = status_val or "unknown"
        licence = str(record.get("licence") or record.get("license") or "unknown")
        rows.append((source, session, filename, size_str, display_status, licence))

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "  ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "  ".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(f"{val:<{col_widths[i]}}" for i, val in enumerate(row)))

    print()
    if has_estimate:
        print(
            "~ = estimate: ortho files are read as a window of a Cloud-Optimized GeoTIFF; "
            "only the window is transferred."
        )
    n_files, total_mb = _compute_samples_total(records)
    print(f"Total: {n_files} files, ~{total_mb:.0f} MB")


def _resolve_run_paths(run: Path) -> tuple[Path, Path]:
    """Resolve run directory and checkpoint path from a run or checkpoint directory."""
    run_dir = run.parent if run.name in {"best", "last"} else run
    checkpoint = run if run.name in {"best", "last"} else run / "best"
    return run_dir, checkpoint


def main(argv: list[str] | None = None) -> int:
    """Run a Flurseg command and return its process exit status."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="replace")
            except (AttributeError, io.UnsupportedOperation, ValueError):
                pass
    arguments = build_parser().parse_args(argv)
    paths = RepoPaths.from_root(getattr(arguments, "root", None))
    if arguments.command == "check-env":
        return check_environment(paths)
    if arguments.command == "prelabel":
        from flurseg.config import load_config
        from flurseg.pipeline import prelabel
        from flurseg.segment.fake_backend import FakeBackend
        from flurseg.segment.sam3_backend import Sam3Backend

        cfg = load_config(paths)
        if arguments.backend == "fake":
            factory = FakeBackend
        else:
            factory = lambda: Sam3Backend(
                _sam3_checkpoint(paths, cfg),
                cfg.pipeline.sam3,
                min_instance_score=cfg.pipeline.exemplars.find_similar.min_instance_score,
            )
        prelabel(paths, cfg, factory, limit=arguments.limit)
        return 0
    if arguments.command == "reprocess":
        from flurseg.config import load_config
        from flurseg.pipeline import reprocess

        reprocess(paths, load_config(paths), arguments.ids)
        return 0
    if arguments.command == "gpu-smoke":
        return gpu_smoke(paths, arguments.image)
    if arguments.command == "review":
        from flurseg.config import load_config
        from flurseg.review.app import run

        cfg = load_config(paths)
        if arguments.backend == "fake":
            from flurseg.segment.fake_backend import FakeBackend

            factory = FakeBackend
        elif arguments.backend == "sam3":
            from flurseg.segment.sam3_backend import Sam3Backend

            factory = lambda: Sam3Backend(
                _sam3_checkpoint(paths, cfg),
                cfg.pipeline.sam3,
                min_instance_score=cfg.pipeline.exemplars.find_similar.min_instance_score,
            )
        else:
            factory = None
        run(paths, cfg, factory, port=arguments.port, open_browser=not arguments.no_browser)
        return 0
    if arguments.command == "build-dataset":
        from flurseg.config import load_config
        from flurseg.dataset.build import build_dataset
        from flurseg.dataset.validate import MaskValidationError

        cfg = load_config(paths)
        try:
            info = build_dataset(paths, cfg)
        except (MaskValidationError, ValueError) as error:
            print(f"ERROR: Dataset build failed: {error}")
            return 1
        tile_counts = info.get("tile_counts", {})
        train_count = tile_counts.get("train", 0)
        val_count = tile_counts.get("val", 0)
        test_count = tile_counts.get("test", 0)
        print(f"Dataset split tile counts: train={train_count}, val={val_count}, test={test_count}")
        dataset_json_path = paths.dataset_dir / "dataset.json"
        print(f"Dataset manifest: {dataset_json_path}")
        return 0
    if arguments.command == "train":
        from flurseg.train.config import load_train_config
        from flurseg.train.engine import run_training

        config_path = arguments.config if arguments.config.is_absolute() else paths.root / arguments.config
        try:
            run_dir = run_training(paths.root, config_path, load_train_config(config_path), arguments.resume, arguments.steps, arguments.dry_run)
        except (OSError, ValueError, RuntimeError) as error:
            print(f"ERROR: Training failed: {error}")
            return 1
        print(f"Training run: {run_dir}")
        try:
            import torch

            best_miou = torch.load(
                run_dir / "last" / "training_state.pt", map_location="cpu", weights_only=False
            )["best_miou"]
            print(f"Best mIoU: {best_miou:.4f}")
        except (OSError, KeyError, RuntimeError):
            pass
        return 0
    if arguments.command == "predict":
        import torch
        from transformers import EomtDinov3ForUniversalSegmentation

        from flurseg.train.config import load_train_config
        from flurseg.train.predict import load_run_metadata, predict_image

        run = arguments.run if arguments.run.is_absolute() else paths.root / arguments.run
        run_dir, checkpoint = _resolve_run_paths(run)
        cfg = load_train_config(run_dir / "config.yaml")
        model = EomtDinov3ForUniversalSegmentation.from_pretrained(checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        mask, overlay = predict_image(
            model,
            arguments.image,
            arguments.out,
            cfg.model.image_mean,
            cfg.model.image_std,
            cfg.inference.window_size,
            cfg.inference.stride,
            load_run_metadata(run_dir),
        )
        print(f"Mask: {mask}\nOverlay: {overlay}")
        return 0
    if arguments.command == "report":
        from flurseg.config import load_config
        from flurseg.report.render import generate_report

        cfg = load_config(paths)
        report_path = generate_report(paths, cfg)
        print(f"Report generated: {report_path}")
        if getattr(arguments, "open", False):
            import webbrowser

            webbrowser.open(str(report_path))
        return 0
    if arguments.command == "fetch-samples":
        from flurseg.samples.fetch import fetch_samples

        if arguments.dry_run:
            results = fetch_samples(paths, dry_run=True, only=arguments.only)
            _print_samples_table(results)
            return 0

        planned = fetch_samples(paths, dry_run=True, only=arguments.only)
        _print_samples_table(planned)
        n_files, total_mb = _compute_samples_total(planned)

        if not getattr(arguments, "yes", False):
            try:
                answer = input(f"Download {n_files} files (~{total_mb:.0f} MB)? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nDownload aborted.")
                return 1
            if answer not in ("y", "yes"):
                print("Download aborted.")
                return 0

        results = fetch_samples(paths, dry_run=False, only=arguments.only)
        _print_samples_table(results)
        downloaded = sum(record.get("status") == "downloaded" for record in results)
        skipped = sum(record.get("status") == "skipped" for record in results)
        failed = sum(record.get("status") == "failed" for record in results)
        print(f"Downloaded {downloaded}, skipped {skipped}, failed {failed}")
        return 1 if failed else 0
    raise RuntimeError(f"unsupported command: {arguments.command}")


def gpu_smoke(paths: RepoPaths, image_path: Path) -> int:
    """Run the opt-in real-model smoke test only when CUDA and checkpoint are present."""
    import time

    import torch

    if not torch.cuda.is_available():
        print("SKIP: CUDA is unavailable; GPU smoke test was not run.")
        return 0
    if not image_path.is_file():
        print(f"ERROR: image does not exist: {image_path}")
        return 2
    from flurseg.config import load_config
    from flurseg.io.images import load_image
    from flurseg.segment.sam3_backend import Sam3Backend

    cfg = load_config(paths)
    checkpoint_path = _sam3_checkpoint(paths, cfg)
    if not checkpoint_path.is_file():
        print("SKIP: SAM 3 checkpoint is missing; GPU smoke test was not run.")
        return 0
    image = load_image(image_path)
    tile_size = int(cfg.pipeline.sam3.imgsz)
    if image.shape[0] < tile_size or image.shape[1] < tile_size:
        print(f"ERROR: image must be at least {tile_size}x{tile_size} pixels for the GPU smoke test.")
        return 2
    tile = image[:tile_size, :tile_size]
    prompts = [prompt for prompt, _ in cfg.prompt_list()]
    if not prompts:
        print("ERROR: classes configuration contains no SAM prompts.")
        return 2

    def valid_scores(scores: Any, expected_count: int) -> bool:
        return (
            isinstance(scores, np.ndarray)
            and scores.shape == (expected_count, tile_size, tile_size)
            and scores.dtype == np.float32
            and np.isfinite(scores).all()
            and ((scores >= 0).all() and (scores <= 1).all())
        )

    import numpy as np

    began = time.monotonic()
    backend = Sam3Backend(
        checkpoint_path,
        cfg.pipeline.sam3,
        min_instance_score=cfg.pipeline.exemplars.find_similar.min_instance_score,
    )
    try:
        print("Prompt chunk benchmark (requested size, time, peak VRAM):")
        for chunk_size in (1, 2, 4, 6, 8):
            torch.cuda.reset_peak_memory_stats()
            chunk_began = time.monotonic()
            try:
                chunk_scores = backend._raw_scores(tile, prompts, prompt_chunk_size=chunk_size)
                chunk_elapsed = time.monotonic() - chunk_began
                chunk_peak = min(
                    torch.cuda.max_memory_allocated(), torch.cuda.get_device_properties(0).total_memory
                ) / 1024**3
                if not valid_scores(chunk_scores, len(prompts)):
                    print(f"  {chunk_size}: invalid score values")
                    return 1
                print(f"  {chunk_size}: {chunk_elapsed:.1f}s, {chunk_peak:.2f} GiB")
            except torch.cuda.OutOfMemoryError:
                print(f"  {chunk_size}: CUDA out of memory")
                torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        auto_scores = backend.score_tile(tile, prompts)
        auto_elapsed = time.monotonic() - began
        auto_peak = min(
            torch.cuda.max_memory_allocated(), torch.cuda.get_device_properties(0).total_memory
        ) / 1024**3
        if not valid_scores(auto_scores, len(prompts)):
            print("ERROR: SAM 3 returned invalid score values.")
            return 1
        print(f"Configured precision full tile: {auto_elapsed:.1f}s, peak VRAM {auto_peak:.2f} GiB")
        for prompt, score in zip(prompts, auto_scores, strict=True):
            print(f"Prompt max score [{prompt}]: {float(score.max()):.6f}")

        center_x, center_y = tile_size // 2, tile_size // 2
        radius = max(1, tile_size // 8)
        box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
        box_scores = backend.score_boxes(tile, [box], "tree" if "tree" in prompts else prompts[0])
        if (
            not isinstance(box_scores, np.ndarray)
            or box_scores.shape != (tile_size, tile_size)
            or box_scores.dtype != np.float32
            or not np.isfinite(box_scores).all()
            or not ((box_scores >= 0).all() and (box_scores <= 1).all())
        ):
            print("ERROR: SAM 3 returned an invalid box-prompt score map.")
            return 1
        print(f"Box prompt max score: {float(box_scores.max()):.6f}")
    finally:
        backend.close()

    fp32_config = cfg.pipeline.sam3.model_copy(update={"precision": "fp32"})
    fp32_backend = Sam3Backend(
        checkpoint_path,
        fp32_config,
        min_instance_score=cfg.pipeline.exemplars.find_similar.min_instance_score,
    )
    try:
        torch.cuda.reset_peak_memory_stats()
        fp32_began = time.monotonic()
        fp32_scores = fp32_backend.score_tile(tile, prompts)
        fp32_elapsed = time.monotonic() - fp32_began
        fp32_peak = min(
            torch.cuda.max_memory_allocated(), torch.cuda.get_device_properties(0).total_memory
        ) / 1024**3
        if not valid_scores(fp32_scores, len(prompts)):
            print("ERROR: SAM 3 returned invalid FP32 score values.")
            return 1
    finally:
        fp32_backend.close()

    mean_absolute_difference = float(np.abs(fp32_scores - auto_scores).mean())
    argmax_agreement = float((np.argmax(fp32_scores, axis=0) == np.argmax(auto_scores, axis=0)).mean() * 100)
    print(f"FP32 full tile: {fp32_elapsed:.1f}s, peak VRAM {fp32_peak:.2f} GiB")
    print(
        f"Precision comparison: mean absolute difference {mean_absolute_difference:.8f}, "
        f"argmax agreement {argmax_agreement:.2f}%"
    )
    print(f"GPU smoke passed: {time.monotonic() - began:.1f}s total")
    return 0
