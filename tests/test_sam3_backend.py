from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flurseg.segment.gpu import reduced_chunk_size
from flurseg.segment.sam3_backend import Sam3Backend, _make_predictor_type


def test_sam3_backend_rgb_to_bgr_without_loading_a_model() -> None:
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    assert Sam3Backend._bgr(rgb).tolist() == [[[3, 2, 1]]]


def test_dense_predictor_constructs_with_only_valid_ultralytics_overrides() -> None:
    """Construction intentionally does not load weights or initialise CUDA."""
    predictor_type = _make_predictor_type()
    predictor = predictor_type(
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": "models/sam3.pt",
            "imgsz": 1008,
            "quantize": 32,
            "verbose": False,
            "save": False,
        }
    )
    predictor.query_threshold = 0.25
    predictor.query_fusion_chunk_size = 128
    assert predictor.query_threshold == 0.25


class _FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_bf16_supported() -> bool:
        return True

    @staticmethod
    def empty_cache() -> None:
        return None

    @staticmethod
    def max_memory_allocated() -> int:
        return 0

    @staticmethod
    def reset_peak_memory_stats() -> None:
        return None


class _FakeTorch:
    cuda = _FakeCuda()


class _ArrayTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self.array


class _ChunkPredictor:
    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self.features = object()
        self.set_image_calls = 0
        self.reset_prompt_calls = 0
        self.reset_image_calls = 0
        self.model = _FakeModel()

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1

    def _inference_features(self, features, *, text):
        assert features is self.features
        assert self.model.names == text
        assert self.model.text_embeddings == {"prompts": tuple(text)}
        return len(text)

    def postprocess(self, raw, image, originals):
        return [_ArrayTensor(np.full((raw, self.height, self.width), raw, np.float32))]

    def reset_prompts(self) -> None:
        self.reset_prompt_calls += 1

    def reset_image(self) -> None:
        self.reset_image_calls += 1


class _FakeBackbone:
    def __init__(self) -> None:
        self.forward_text_calls: list[tuple[str, ...]] = []

    def forward_text(self, prompts: list[str]) -> dict[str, tuple[str, ...]]:
        self.forward_text_calls.append(tuple(prompts))
        return {"prompts": tuple(prompts)}


class _FakeModel:
    def __init__(self) -> None:
        self.backbone = _FakeBackbone()
        self.text_embeddings: dict[str, tuple[str, ...]] = {}
        self.names: list[str] = []


class _ReadOnlyShapePredictor:
    """Mimic current Ultralytics behavior after set_image leaves batch unset."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width
        self.features = object()
        self.batch = None
        self.prepared_shape: tuple[int, int] | None = None
        self.reset_prompt_calls = 0
        self.reset_image_calls = 0

    @property
    def src_shape(self) -> tuple[int, int]:
        return self.batch[1][0].shape[:2]

    def set_image(self, image: np.ndarray) -> None:
        self.batch = None

    def _prepare_geometric_prompts(self, source_shape, boxes, labels):
        self.prepared_shape = tuple(source_shape)
        return boxes, labels

    def _inference_features(self, features, bboxes, labels, *, text):
        assert features is self.features
        return {
            "pred_logits": torch.full((1, 1, 1), 10.0),
            "pred_masks": torch.zeros((1, 1, self.height, self.width)),
            "semantic_seg": torch.zeros((1, 1, self.height, self.width)),
            "presence_logit_dec": torch.full((1,), 10.0),
            "pred_boxes": torch.zeros((1, 1, 4)),
        }

    def postprocess(self, raw, image, originals):
        return [_ArrayTensor(np.full((raw, self.height, self.width), 0.5, np.float32))]

    def reset_prompts(self) -> None:
        self.reset_prompt_calls += 1

    def reset_image(self) -> None:
        self.reset_image_calls += 1


def _backend(*, chunk_size: int = 16, precision: str = "fp16", clock=None) -> Sam3Backend:
    config = SimpleNamespace(
        prompt_chunk_size=chunk_size,
        precision=precision,
        max_vram_gb=99.0,
        slow_chunk_min_seconds=2.0,
        min_instance_score=0.5,
        query_fusion_chunk_size=128,
    )
    return Sam3Backend("unused.pt", config, clock=clock)


@pytest.mark.parametrize(
    ("precision", "bf16_supported", "expected_quantize"),
    [
        ("fp16", True, 16),
        ("fp32", True, 32),
        ("bf16", True, 32),
        ("auto", True, 32),
        ("auto", False, 16),
    ],
)
def test_predictor_factory_uses_quantize_without_deprecated_half(
    precision: str, bf16_supported: bool, expected_quantize: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_overrides: dict[str, object] = {}

    def factory(*, overrides: dict[str, object]) -> SimpleNamespace:
        captured_overrides.update(overrides)
        return SimpleNamespace()

    config = SimpleNamespace(
        conf=0.25,
        imgsz=1008,
        precision=precision,
        query_threshold=0.25,
        query_fusion_chunk_size=128,
    )
    backend = Sam3Backend(__file__, config, predictor_factory=factory)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: bf16_supported,
        )
    )
    monkeypatch.setattr("flurseg.segment.sam3_backend._require_torch", lambda: fake_torch)

    assert backend.predictor is not None
    assert captured_overrides["quantize"] == expected_quantize
    assert "half" not in captured_overrides


def test_sam3_backend_converts_rgb_to_bgr_exactly() -> None:
    rgb = np.array([[[4, 19, 230], [7, 80, 2]]], dtype=np.uint8)
    assert np.array_equal(Sam3Backend._bgr(rgb), np.array([[[230, 19, 4], [2, 80, 7]]], dtype=np.uint8))


def test_prompt_chunks_cache_text_embeddings_across_tiles() -> None:
    clock_values = iter(range(18))
    backend = _backend(chunk_size=16, clock=lambda: float(next(clock_values)))
    predictor = _ChunkPredictor(4, 5)
    backend._predictor, backend._torch = predictor, _FakeTorch()
    prompts = [f"prompt-{index}" for index in range(40)]
    for _ in range(3):
        scores = backend.score_tile(np.zeros((4, 5, 3), np.uint8), prompts)
        assert scores.shape == (40, 4, 5)
    assert predictor.model.backbone.forward_text_calls == [
        tuple(prompts[:16]), tuple(prompts[16:32]), tuple(prompts[32:]),
    ]
    assert predictor.set_image_calls == 3 and predictor.reset_prompt_calls == 0 and predictor.reset_image_calls == 3


def test_slow_chunk_minimum_duration_ignores_short_relative_outlier() -> None:
    reduced, reason = reduced_chunk_size(
        current_chunk_size=16,
        max_vram_gb=99.0,
        slow_chunk_min_seconds=2.0,
        elapsed_seconds=1.0,
        previous_seconds=[0.1],
        torch_module=_FakeTorch(),
    )

    assert (reduced, reason) == (16, None)


def test_box_prompts_use_explicit_source_shape_without_assigning_predictor_property() -> None:
    backend = _backend()
    predictor = _ReadOnlyShapePredictor(4, 5)
    backend._predictor, backend._torch = predictor, _FakeTorch()

    scores = backend.score_boxes(np.zeros((4, 5, 3), np.uint8), [(1, 1, 4, 3)], "tree")

    assert scores.shape == (4, 5)
    assert predictor.batch is None
    assert predictor.prepared_shape == (4, 5)
    assert predictor.reset_prompt_calls == 0 and predictor.reset_image_calls == 1


class _InstanceOnlyPredictor(_ReadOnlyShapePredictor):
    def _inference_features(self, features, bboxes, labels, *, text):
        assert features is self.features
        return {
            "pred_logits": torch.tensor([[[10.0], [-10.0]]]),
            "pred_masks": torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]], [[10.0, 10.0], [10.0, 10.0]]]]),
            "semantic_seg": torch.full((1, 1, 2, 2), 20.0),
            "presence_logit_dec": torch.tensor([10.0]),
            "pred_boxes": torch.zeros((1, 2, 4)),
        }


def test_box_prompts_ignore_noisy_semantic_head_and_use_retained_instances() -> None:
    backend = _backend()
    predictor = _InstanceOnlyPredictor(2, 2)
    backend._predictor, backend._torch = predictor, _FakeTorch()

    scores = backend.score_boxes(np.zeros((2, 2, 3), np.uint8), [(0, 0, 2, 2)], "vehicle")

    assert scores[0, 0] > 0.99 and scores[1, 1] > 0.99
    assert scores[0, 1] < 0.01 and scores[1, 0] < 0.01


def test_oom_retries_once_with_half_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(chunk_size=16)
    backend._predictor, backend._torch = object(), _FakeTorch()
    calls: list[int] = []

    def raw(image, prompts, *, prompt_chunk_size=None):
        calls.append(prompt_chunk_size)
        if len(calls) == 1:
            raise _FakeCuda.OutOfMemoryError("full")
        return np.zeros((len(prompts), *image.shape[:2]), np.float32)

    monkeypatch.setattr(backend, "_raw_scores", raw)
    assert backend.score_tile(np.zeros((3, 4, 3), np.uint8), ["a", "b"]).shape == (2, 3, 4)
    assert calls == [16, 8]


def test_ooms_raise_after_retrying_down_to_one_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _backend(chunk_size=16)
    backend._predictor, backend._torch = object(), _FakeTorch()
    calls: list[int] = []

    def raw(*args, prompt_chunk_size=None, **kwargs):
        calls.append(prompt_chunk_size)
        raise _FakeCuda.OutOfMemoryError("oom")

    monkeypatch.setattr(backend, "_raw_scores", raw)
    with pytest.raises(_FakeCuda.OutOfMemoryError):
        backend.score_tile(np.zeros((3, 4, 3), np.uint8), ["a"])
    assert calls == [16, 8, 4, 2, 1]


@pytest.mark.parametrize(("supported", "expected"), [(True, "bf16"), (False, "fp16")])
def test_auto_precision_uses_bf16_only_when_supported(supported: bool, expected: str) -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_bf16_supported=lambda: supported))
    assert _backend(precision="auto")._precision(torch) == expected
