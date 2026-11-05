"""Config-driven public imagery fetcher for additional training data."""

from typing import Any

__all__ = ["fetch_extra"]


def __getattr__(name: str) -> Any:
    """Load the runner lazily so the shared licence policy has no import cycle."""
    if name == "fetch_extra":
        from flurseg.samples.extra.runner import fetch_extra

        return fetch_extra
    raise AttributeError(name)
