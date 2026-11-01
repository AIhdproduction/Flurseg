"""Small console-friendly progress reporter."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ProgressReporter:
    """Report completed work, per-item time, average, and estimated time remaining."""

    total: int
    emit: Callable[[str], None] = print
    clock: Callable[[], float] = time.monotonic
    completed: int = 0
    _started: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self) -> None:
        """Capture a shared start time from the injectable monotonic clock."""
        self._started = self.clock()
        self._last = self._started

    def advance(self) -> str:
        """Record one completion, print, and return a plain-text status line."""
        now = self.clock()
        self.completed += 1
        last = now - self._last
        elapsed = now - self._started
        average = elapsed / self.completed
        eta = max(0.0, (self.total - self.completed) * average)
        self._last = now
        message = (
            f"{self.completed}/{self.total} | last {last:.1f}s | average {average:.1f}s | ETA {eta:.1f}s"
        )
        self.emit(message)
        return message
