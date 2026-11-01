"""Small CUDA-memory safeguards shared by the SAM 3 backend and smoke command."""

from __future__ import annotations

import gc
import statistics
from collections.abc import Callable


def peak_vram_gb(torch_module: object) -> float:
    """Return the current process CUDA peak allocation in GiB."""
    peak_bytes = float(torch_module.cuda.max_memory_allocated())
    get_properties = getattr(torch_module.cuda, "get_device_properties", None)
    if get_properties is not None:
        peak_bytes = min(peak_bytes, float(get_properties(0).total_memory))
    return peak_bytes / 1024**3


def reset_peak_vram_stats(torch_module: object) -> None:
    """Reset CUDA peak accounting before one independently measured prompt chunk."""
    reset = getattr(torch_module.cuda, "reset_peak_memory_stats", None)
    if reset is not None:
        reset()


def reduced_chunk_size(
    *,
    current_chunk_size: int,
    max_vram_gb: float,
    slow_chunk_min_seconds: float,
    elapsed_seconds: float,
    previous_seconds: list[float],
    torch_module: object,
) -> tuple[int, str | None]:
    """Return a safer size when a prompt chunk exceeds memory or time limits."""
    peak = peak_vram_gb(torch_module)
    reason: str | None = None
    if peak > max_vram_gb:
        reason = f"peak VRAM {peak:.1f} GiB exceeds the {max_vram_gb:.1f} GiB budget"
    elif (
        previous_seconds
        and elapsed_seconds >= slow_chunk_min_seconds
        and elapsed_seconds > 5 * statistics.median(previous_seconds)
    ):
        reason = (
            f"chunk time {elapsed_seconds:.2f}s exceeds five times the prior median "
            f"of {statistics.median(previous_seconds):.2f}s"
        )
    if reason is None or current_chunk_size <= 1:
        return current_chunk_size, None
    return max(1, current_chunk_size // 2), reason


def release_cuda_memory(torch_module: object) -> None:
    """Release cached allocations between images without touching user files."""
    gc.collect()
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def retry_after_oom[T](operation: Callable[[int], T], chunk_size: int, torch_module: object) -> T:
    """Retry with progressively smaller prompt chunks after CUDA out-of-memory errors."""
    current_chunk_size = chunk_size
    while True:
        try:
            return operation(current_chunk_size)
        except torch_module.cuda.OutOfMemoryError:
            if current_chunk_size <= 1:
                raise
            release_cuda_memory(torch_module)
            current_chunk_size = max(1, current_chunk_size // 2)
