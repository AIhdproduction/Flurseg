"""Lazy, dense-score SAM 3 integration.

This module never downloads weights and does not run inference during import.  Actual model
execution is exclusively initiated by an explicit caller such as ``prelabel --backend sam3``.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from flurseg.segment.gpu import (
    reduced_chunk_size,
    release_cuda_memory,
    reset_peak_vram_stats,
    retry_after_oom,
)

logger = logging.getLogger(__name__)


def _ultralytics_version() -> str:
    try:
        import ultralytics

        return str(ultralytics.__version__)
    except Exception:  # noqa: BLE001 - preserve a useful error when optional dependencies are unavailable.
        return "unavailable"


def _require_torch() -> Any:
    import torch

    return torch


def _fuse_box_instance_masks(
    predictions: dict[str, Any],
    *,
    min_instance_score: float,
    output_size: tuple[int, int],
    query_chunk_size: int,
) -> Any:
    """Return visual-exemplar scores from retained instance masks only."""
    if not 0 <= min_instance_score <= 1:
        raise ValueError("min_instance_score must be between 0 and 1")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")

    from torch.nn import functional

    from flurseg.segment.heads import validate_heads

    classes, queries, _, _ = validate_heads(predictions)
    logits = predictions["pred_logits"].squeeze(-1).sigmoid()
    presence = predictions["presence_logit_dec"].reshape(classes).sigmoid()
    instance_scores = logits * presence[:, None]
    masks = predictions["pred_masks"]
    best_instance = masks.new_zeros((classes, *masks.shape[-2:]))
    for start in range(0, queries, query_chunk_size):
        end = min(queries, start + query_chunk_size)
        scores = instance_scores[:, start:end]
        values = masks[:, start:end].sigmoid() * scores[..., None, None]
        values = values.masked_fill((scores < min_instance_score)[..., None, None], 0)
        best_instance = best_instance.maximum(values.max(dim=1).values)
    return functional.interpolate(
        best_instance[:, None].float(), size=output_size, mode="bilinear", align_corners=False
    )[:, 0]


def _make_predictor_type() -> type[Any]:
    """Build the raw-head predictor subclass only when SAM 3 is actually requested."""
    from ultralytics.models.sam import SAM3SemanticPredictor

    from flurseg.segment.heads import fuse_dense_heads, validate_heads

    class DenseSAM3Predictor(SAM3SemanticPredictor):
        def postprocess(self, preds: dict[str, Any], img: Any, orig_imgs: Any) -> list[Any]:
            try:
                validate_heads(preds)
                if isinstance(orig_imgs, list):
                    height, width = orig_imgs[0].shape[:2]
                else:
                    height, width = orig_imgs.shape[-2:]
                dense = fuse_dense_heads(
                    preds,
                    query_threshold=float(self.query_threshold),
                    output_size=(height, width),
                    query_chunk_size=int(self.query_fusion_chunk_size),
                )
            except (KeyError, ValueError, AttributeError) as error:
                raise RuntimeError(
                    f"SAM 3 raw-head contract mismatch with ultralytics {_ultralytics_version()}: {error}"
                ) from error
            return [dense]

    return DenseSAM3Predictor


class Sam3Backend:
    """SAM 3 dense prompt scorer with feature reuse across prompt chunks."""

    name = "sam3"

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: Any,
        *,
        predictor_factory: Any | None = None,
        clock: Callable[[], float] | None = None,
        min_instance_score: float = 0.5,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.config = config
        self._predictor_factory = predictor_factory
        self._clock = clock or time.monotonic
        self.min_instance_score = min_instance_score
        self._predictor: Any | None = None
        self._torch: Any | None = None
        self._text_embedding_cache: dict[tuple[str, ...], Any] = {}
        self._safe_prompt_chunk_size: int | None = None
        self._adaptive_warning_emitted = False

    @property
    def predictor(self) -> Any:
        """Create the predictor lazily so tests and fake-only workflows need no checkpoint."""
        if self._predictor is None:
            if not self.checkpoint_path.is_file():
                raise FileNotFoundError(f"SAM 3 checkpoint is missing: {self.checkpoint_path}")
            torch = _require_torch()
            if not torch.cuda.is_available():
                raise RuntimeError("SAM 3 requires CUDA; use the fake backend for CPU workflows")
            precision = self._precision(torch)
            factory = self._predictor_factory or _make_predictor_type()
            self._predictor = factory(
                overrides={
                    "conf": float(self.config.conf),
                    "task": "segment",
                    "mode": "predict",
                    "model": str(self.checkpoint_path),
                    "imgsz": int(self.config.imgsz),
                    "quantize": 16 if precision == "fp16" else 32,
                    "verbose": False,
                    "save": False,
                }
            )
            # These are Flurseg controls, not Ultralytics configuration keys.  Passing
            # them through overrides makes current Ultralytics reject construction.
            self._predictor.query_threshold = float(self.config.query_threshold)
            self._predictor.query_fusion_chunk_size = int(self.config.query_fusion_chunk_size)
            self._torch = torch
        return self._predictor

    def _precision(self, torch: Any) -> str:
        configured = str(self.config.precision)
        if configured != "auto":
            return configured
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"

    @staticmethod
    def _bgr(tile: np.ndarray) -> np.ndarray:
        if tile.ndim != 3 or tile.shape[2] != 3 or tile.dtype != np.uint8:
            raise ValueError("tile must be an HxWx3 uint8 RGB array")
        return np.ascontiguousarray(tile[..., ::-1])

    def _autocast(self) -> contextlib.AbstractContextManager[None]:
        torch = self._torch or _require_torch()
        if self._precision(torch) == "bf16":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def _set_cached_text_embeddings(self, prompts: Sequence[str]) -> None:
        """Restore text features explicitly because predictor.reset_prompts clears only them."""
        model = getattr(self.predictor, "model", None)
        if model is None:
            return
        key = tuple(prompts)
        if key not in self._text_embedding_cache:
            self._text_embedding_cache[key] = model.backbone.forward_text(list(key))
        model.text_embeddings = self._text_embedding_cache[key]
        model.names = list(key)

    def _adapt_chunk_size(
        self,
        chunk_size: int,
        elapsed_seconds: float,
        previous_seconds: list[float],
    ) -> int:
        torch = self._torch
        assert torch is not None
        reduced, reason = reduced_chunk_size(
            current_chunk_size=chunk_size,
            max_vram_gb=float(self.config.max_vram_gb),
            slow_chunk_min_seconds=float(self.config.slow_chunk_min_seconds),
            elapsed_seconds=elapsed_seconds,
            previous_seconds=previous_seconds,
            torch_module=torch,
        )
        if reason is not None:
            self._safe_prompt_chunk_size = reduced
            if not self._adaptive_warning_emitted:
                logger.warning(
                    "SAM 3 reducing prompt chunk size from %d to %d: %s",
                    chunk_size,
                    reduced,
                    reason,
                )
                self._adaptive_warning_emitted = True
        return reduced

    def _raw_scores(
        self,
        image: np.ndarray,
        prompts: Sequence[str],
        boxes: Sequence[tuple[int, int, int, int]] | None = None,
        *,
        prompt_chunk_size: int | None = None,
    ) -> np.ndarray:
        predictor = self.predictor
        torch = self._torch
        assert torch is not None
        bgr = self._bgr(image)
        inference_mode = getattr(torch, "inference_mode", contextlib.nullcontext)
        with inference_mode():
            predictor.set_image(bgr)
            try:
                # set_image is the installed public feature-cache API.  We call the verified
                # raw-head method below because normal predictor(...) postprocessing discards heads.
                features = predictor.features
                if features is None:
                    raise RuntimeError("SAM 3 predictor did not expose cached image features")
                source_shape = bgr.shape[:2]
                chunks: list[np.ndarray] = []
                configured_chunk_size = prompt_chunk_size or int(self.config.prompt_chunk_size)
                chunk_size = min(configured_chunk_size, self._safe_prompt_chunk_size or configured_chunk_size)
                prompt_list = list(prompts)
                offset = 0
                chunk_seconds: list[float] = []
                while offset < len(prompt_list):
                    chunk = prompt_list if boxes is not None else prompt_list[offset : offset + chunk_size]
                    reset_peak_vram_stats(torch)
                    chunk_started = self._clock()
                    with self._autocast():
                        self._set_cached_text_embeddings(chunk)
                        if boxes is None:
                            raw = predictor._inference_features(features, text=chunk)
                        else:
                            prepared = predictor._prepare_geometric_prompts(source_shape, boxes, None)
                            raw = predictor._inference_features(features, *prepared, text=chunk[0])
                        dense = predictor.postprocess(raw, None, [bgr])[0]
                    chunks.append(dense.detach().float().cpu().numpy())
                    elapsed = self._clock() - chunk_started
                    next_size = self._adapt_chunk_size(chunk_size, elapsed, chunk_seconds)
                    chunk_seconds.append(elapsed)
                    if boxes is not None:
                        break
                    offset += len(chunk)
                    chunk_size = next_size
                return np.concatenate(chunks, axis=0).astype(np.float32)
            finally:
                predictor.reset_image()

    def score_tile(self, tile: np.ndarray, prompts: Sequence[str]) -> np.ndarray:
        """Return one dense probability map per text prompt with one OOM retry."""
        if not prompts:
            return np.empty((0, *tile.shape[:2]), dtype=np.float32)
        original = int(self.config.prompt_chunk_size)

        def score(chunk_size: int) -> np.ndarray:
            return self._raw_scores(tile, prompts, prompt_chunk_size=chunk_size)

        result = retry_after_oom(score, original, self.predictor and self._torch)
        self._release_cuda_memory()
        return result

    def _raw_box_instance_scores(
        self, image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], prompt: str
    ) -> np.ndarray:
        """Run geometric prompts and aggregate only retained instance masks."""
        predictor = self.predictor
        torch = self._torch
        assert torch is not None
        bgr = self._bgr(image)
        inference_mode = getattr(torch, "inference_mode", contextlib.nullcontext)
        with inference_mode():
            predictor.set_image(bgr)
            try:
                features = predictor.features
                if features is None:
                    raise RuntimeError("SAM 3 predictor did not expose cached image features")
                with self._autocast():
                    self._set_cached_text_embeddings([prompt])
                    prepared = predictor._prepare_geometric_prompts(bgr.shape[:2], boxes, None)
                    raw = predictor._inference_features(features, *prepared, text=prompt)
                    try:
                        dense = _fuse_box_instance_masks(
                            raw,
                            min_instance_score=float(self.min_instance_score),
                            output_size=bgr.shape[:2],
                            query_chunk_size=int(self.config.query_fusion_chunk_size),
                        )
                    except (KeyError, ValueError, AttributeError) as error:
                        raise RuntimeError(
                            f"SAM 3 raw-head contract mismatch with ultralytics {_ultralytics_version()}: {error}"
                        ) from error
                return dense.detach().float().cpu().numpy().astype(np.float32)
            finally:
                predictor.reset_image()

    def score_boxes(self, image: np.ndarray, boxes: Sequence[tuple[int, int, int, int]], prompt: str) -> np.ndarray:
        """Return a map from same-image visual exemplars using instance masks only."""
        if not boxes:
            return np.zeros(image.shape[:2], dtype=np.float32)
        result = self._raw_box_instance_scores(image, boxes, prompt)[0]
        self._release_cuda_memory()
        return result

    def _release_cuda_memory(self) -> None:
        if self._torch is None:
            return
        release_cuda_memory(self._torch)

    def close(self) -> None:
        """Release model references and cached CUDA allocations."""
        if self._torch is not None:
            release_cuda_memory(self._torch)
        self._predictor = None
        self._torch = None
        self._text_embedding_cache.clear()
