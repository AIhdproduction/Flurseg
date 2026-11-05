"""Small console-friendly progress reporter."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from flurseg.estimate import format_duration


@dataclass
class ProgressReporter:
    """Report completed work, per-item time, average, and estimated time remaining."""

    total: int
    emit: Callable[[str], None] = print
    clock: Callable[[], float] = time.monotonic
    total_weight: float | None = None
    human: bool = False
    now: Callable[[], datetime] = datetime.now
    completed: int = 0
    completed_weight: float = 0.0
    _started: float = field(init=False)
    _last: float = field(init=False)

    def __post_init__(self) -> None:
        """Capture a shared start time from the injectable monotonic clock."""
        if self.total_weight is None:
            self.total_weight = float(self.total)
        if self.total_weight < 0:
            raise ValueError("total_weight must be non-negative")
        self._started = self.clock()
        self._last = self._started

    def advance(self, weight: float = 1.0) -> str:
        """Record one completion, print, and return a plain-text status line."""
        if weight < 0:
            raise ValueError("weight must be non-negative")
        now = self.clock()
        self.completed += 1
        self.completed_weight += weight
        last = now - self._last
        elapsed = now - self._started
        average = elapsed / self.completed
        eta = 0.0 if self.completed_weight <= 0 else max(
            0.0, (self.total_weight - self.completed_weight) * elapsed / self.completed_weight
        )
        self._last = now
        if self.human:
            percentage = 100 * self.completed_weight / self.total_weight if self.total_weight else 100.0
            done_time = self.now() + timedelta(seconds=eta)
            message = (
                f"[{self.completed:>3}/{self.total} | {percentage:>4.1f} %] "
                f"last {format_duration(last)} | avg {format_duration(average)} | "
                f"remaining ≈ {format_duration(eta)} | done ≈ {done_time:%a %d.%m. %H:%M}"
            )
            self.emit(message)
            return message
        message = (
            f"{self.completed}/{self.total} | last {last:.1f}s | average {average:.1f}s | ETA {eta:.1f}s"
        )
        self.emit(message)
        return message
