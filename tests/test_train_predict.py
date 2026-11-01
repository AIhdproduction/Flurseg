"""Regression tests for sliding-window training inference."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from flurseg.train.predict import predict_image, sliding_window_scores


class ConstantQueryModel(torch.nn.Module):
    """Return spatially constant EoMT-style query logits for inference tests."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        """Return two class scores independent of the input tile."""
        batch = pixel_values.shape[0]
        masks = self.anchor + torch.zeros((batch, 1, 1, 1), device=pixel_values.device)
        classes = self.anchor + torch.tensor(
            [[[2.0, 0.0, -2.0]]],
            device=pixel_values.device,
        ).expand(batch, -1, -1)
        return SimpleNamespace(
            masks_queries_logits=masks,
            class_queries_logits=classes,
        )


def test_hann_blending_preserves_constant_scores() -> None:
    """Normalized Hann blending keeps every output pixel at the same score."""
    model = ConstantQueryModel()
    scores = sliding_window_scores(
        model,
        torch.zeros((3, 91, 77)),
        window_size=32,
        stride=13,
    )
    expected = scores[:, :1, :1].expand_as(scores)
    torch.testing.assert_close(scores, expected, rtol=0, atol=1e-6)


def test_predict_image_preserves_a_1000_by_700_mask_shape(tmp_path: Path) -> None:
    """Image prediction writes a mask with precisely the source width and height."""
    image_path = tmp_path / "source.png"
    image = np.full((700, 1000, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    mask_path, overlay_path = predict_image(
        ConstantQueryModel(),
        image_path,
        tmp_path / "predictions",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        window_size=128,
        stride=64,
        classes=[
            {"id": 0, "color": "#000000"},
            {"id": 1, "color": "#ffffff"},
        ],
    )
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    assert mask.shape == (700, 1000)
    assert overlay_path.is_file()


def test_resolve_run_paths_accepts_run_dir_and_subdirectories(tmp_path: Path) -> None:
    """CLI predict path resolution accepts a run directory or its checkpoint subdirectories."""
    from flurseg.cli import _resolve_run_paths

    run_dir = tmp_path / "run_sample"
    run_dir.mkdir()
    best_dir = run_dir / "best"
    best_dir.mkdir()
    last_dir = run_dir / "last"
    last_dir.mkdir()

    resolved_run, resolved_checkpoint = _resolve_run_paths(run_dir)
    assert resolved_run == run_dir
    assert resolved_checkpoint == best_dir

    resolved_run_best, resolved_checkpoint_best = _resolve_run_paths(best_dir)
    assert resolved_run_best == run_dir
    assert resolved_checkpoint_best == best_dir

    resolved_run_last, resolved_checkpoint_last = _resolve_run_paths(last_dir)
    assert resolved_run_last == run_dir
    assert resolved_checkpoint_last == last_dir
