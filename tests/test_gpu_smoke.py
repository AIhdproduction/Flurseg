from __future__ import annotations

import pytest

from flurseg.paths import RepoPaths


@pytest.mark.gpu
def test_gpu_smoke_skips_without_cuda_or_checkpoint(tmp_root) -> None:
    import torch

    paths = RepoPaths.from_root(tmp_root)
    if not torch.cuda.is_available() or not paths.sam3_checkpoint.exists():
        pytest.skip("CUDA or the owner-supplied SAM 3 checkpoint is unavailable.")
    pytest.skip("Owner-approved checkpoint smoke requires a supplied image.")
