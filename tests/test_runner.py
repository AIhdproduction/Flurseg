from __future__ import annotations

from collections.abc import Sequence

import numpy as np

import flurseg.segment.runner
from flurseg.config import load_config
from flurseg.paths import RepoPaths
from flurseg.segment.fake_backend import FakeBackend
from flurseg.segment.runner import _prompt_data, score_image
from flurseg.segment.scales import resize_image
from flurseg.segment.tiling import TileAccumulator, iter_tiles
from flurseg.segment.tta import apply, invert


def test_runner_produces_scaled_scores_and_zero_equivariant_disagreement(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = score_image(np.full((30, 40, 3), 120, np.uint8), FakeBackend(), cfg)
    assert set(result.scores) == {item.name for item in cfg.classes.classes if item.trainable}
    assert result.tta_disagreement.max() == 0


def test_runner_records_per_class_prompt_winners_and_tta_disagreement(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))

    class IndexedBackend:
        def score_tile(self, tile, prompts):
            base = np.arange(len(prompts), dtype=np.float32)[:, None, None] / max(1, len(prompts))
            return np.broadcast_to(base, (len(prompts), *tile.shape[:2])).copy()

    result = score_image(np.full((17, 19, 3), 120, np.uint8), IndexedBackend(), cfg)
    for definition in cfg.classes.classes:
        if definition.trainable:
            assert result.winner_per_class[definition.name].max() == len(definition.prompts) - 1


def test_runner_reports_nonzero_disagreement_for_non_equivariant_backend(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    cfg.pipeline.tta.variants = ["identity", "rot90", "hflip", "vflip"]
    result = score_image(np.full((30, 40, 3), 120, np.uint8), FakeBackend(equivariant=False, disagreement=0.5), cfg)
    assert result.tta_disagreement.max() > 0


def test_per_class_score_is_the_maximum_over_its_prompts(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))

    class IndexedBackend:
        def score_tile(self, tile, prompts):
            values = np.arange(len(prompts), dtype=np.float32)[:, None, None] / len(prompts)
            return np.broadcast_to(values, (len(prompts), *tile.shape[:2])).copy()

    scale = next(item for item in cfg.pipeline.scales if "tree" in item.classes)
    scale_prompts = [
        prompt
        for definition in cfg.classes.classes
        if definition.name in scale.classes
        for prompt in definition.prompts
    ]
    tree_prompts = set(cfg.class_by_name("tree").prompts)
    tree_indices = [i for i, prompt in enumerate(scale_prompts) if prompt in tree_prompts]
    expected = max(tree_indices) / len(scale_prompts)

    result = score_image(np.full((20, 20, 3), 120, np.uint8), IndexedBackend(), cfg)
    assert np.allclose(result.scores["tree"], expected)


def test_per_class_winner_indices_use_the_three_prompt_class_local_index(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))

    class IndexedBackend:
        def score_tile(self, tile, prompts):
            values = np.arange(len(prompts), dtype=np.float32)[:, None, None]
            return np.broadcast_to(values, (len(prompts), *tile.shape[:2])).copy()

    result = score_image(np.full((20, 20, 3), 120, np.uint8), IndexedBackend(), cfg)
    assert np.all(result.winner_per_class["tree"] == 2)


def test_image_smaller_than_a_tile_keeps_expected_output_shapes(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    result = score_image(np.full((500, 300, 3), 120, np.uint8), FakeBackend(), cfg)
    assert result.prompt_winner.shape == (500, 300) and result.tta_disagreement.shape == (500, 300)
    assert all(score.shape == (500, 300) or score.shape == (250, 150) for score in result.scores.values())


def test_runner_accumulators_are_per_class(monkeypatch, tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    created: list[flurseg.segment.runner.TileAccumulator] = []
    real_cls = flurseg.segment.runner.TileAccumulator

    class RecordingTileAccumulator(real_cls):
        def __init__(self, classes: int, width: int, height: int, reduction: str = "max") -> None:
            super().__init__(classes, width, height, reduction=reduction)
            created.append(self)

    monkeypatch.setattr(flurseg.segment.runner, "TileAccumulator", RecordingTileAccumulator)

    image = np.full((600, 600, 3), 120, dtype=np.uint8)
    score_image(image, FakeBackend(), cfg)

    assert len(created) > 0
    for acc in created:
        scale = next(
            s
            for s in cfg.pipeline.scales
            if (acc.width, acc.height) == resize_image(image, float(s.factor)).shape[1::-1]
        )
        assert acc.classes <= len(scale.classes)


def test_runner_matches_reference_implementation(tmp_root) -> None:
    cfg = load_config(RepoPaths.from_root(tmp_root))
    tiling_cfg = cfg.pipeline.tiling.model_copy(update={"tile_size": 256, "min_overlap": 0.25})
    pipeline_cfg = cfg.pipeline.model_copy(update={"tiling": tiling_cfg})
    cfg_copy = cfg.model_copy(update={"pipeline": pipeline_cfg})

    rng = np.random.default_rng(12345)
    image = rng.integers(0, 256, size=(700, 500, 3), dtype=np.uint8)

    class SmoothNonEquivariantBackend:
        def score_tile(self, tile: np.ndarray, prompts: Sequence[str]) -> np.ndarray:
            th, tw = tile.shape[:2]
            pixels = tile.astype(np.float32) / 255.0
            r = pixels[..., 0]
            g = pixels[..., 1]
            b = pixels[..., 2]
            smooth_color = 0.01 * (np.sin(r * 3.14159) + np.cos(g * 3.14159) + b)
            y_grad = np.linspace(0.0, 0.005, th, dtype=np.float32)[:, None]
            x_grad = np.linspace(0.0, 0.003, tw, dtype=np.float32)[None, :]
            base = smooth_color + y_grad + x_grad
            out = np.empty((len(prompts), th, tw), dtype=np.float32)
            for idx in range(len(prompts)):
                out[idx] = (base + idx * 0.05) / (1.0 + len(prompts) * 0.05)
            return out

    backend = SmoothNonEquivariantBackend()
    result = score_image(image, backend, cfg_copy)

    ref_scores: dict[str, np.ndarray] = {}
    for assignment in cfg_copy.pipeline.scales:
        scaled = resize_image(image, float(assignment.factor))
        height, width = scaled.shape[:2]
        prompts, prompt_class_ids, _ = _prompt_data(cfg_copy, list(assignment.classes))
        if not prompts:
            continue
        mosaic = cfg_copy.pipeline.exemplars.mosaic
        effective_height = (
            max(1, round(cfg_copy.pipeline.tiling.tile_size * (1 - mosaic.strip_fraction)))
            if mosaic.enabled
            else cfg_copy.pipeline.tiling.tile_size
        )
        tiles = list(
            iter_tiles(width, height, cfg_copy.pipeline.tiling.tile_size, cfg_copy.pipeline.tiling.min_overlap)
        )
        if mosaic.enabled:
            tiles = list(iter_tiles(width, height, effective_height, cfg_copy.pipeline.tiling.min_overlap))
            tiles = [
                type(tile)(tile.x, tile.y, min(tile.width, cfg_copy.pipeline.tiling.tile_size), tile.height)
                for tile in tiles
            ]

        variants: list[np.ndarray] = []
        for variant in cfg_copy.pipeline.tta.variants:
            accumulator = TileAccumulator(len(prompts), width, height, reduction="mean")
            for tile in tiles:
                data = scaled[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
                transformed = apply(variant, data)
                values = invert(variant, backend.score_tile(transformed, prompts))
                accumulator.add(tile, values)
            variants.append(accumulator.scores())

        mean_prompt_maps = np.mean(variants, axis=0, dtype=np.float32)
        for class_id in set(prompt_class_ids):
            class_name = cfg_copy.class_by_id(class_id).name
            prompt_indices = [i for i, cid in enumerate(prompt_class_ids) if cid == class_id]
            ref_scores[class_name] = mean_prompt_maps[prompt_indices].max(axis=0)

    assert set(result.scores.keys()) == set(ref_scores.keys())
    for class_name, ref_score in ref_scores.items():
        np.testing.assert_allclose(result.scores[class_name], ref_score, atol=1e-5)
