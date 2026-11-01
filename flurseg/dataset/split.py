"""Deterministic, persistent session splits for training datasets."""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flurseg.config import DatasetSplit

logger = logging.getLogger(__name__)
_SPLITS = ("train", "val", "test")


@dataclass
class SessionGroup:
    """Aggregated pool images belonging to one ``(source, session)`` pair."""

    source: str
    session: str
    is_test_session: bool
    image_ids: list[str] = field(default_factory=list)
    records: list[Any] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.session

    @property
    def registry_key(self) -> str:
        return f"{self.source}/{self.session}"

    @property
    def count(self) -> int:
        return len(self.image_ids)


class SplitResult(dict[str, list[Any]]):
    """Split records plus stable assignment metadata."""

    def __init__(self, splits: dict[str, list[Any]], session_to_split: dict[str, str], source_session_to_split: dict[tuple[str, str], str], split_to_sessions: dict[str, list[str]], image_ids_per_split: dict[str, list[str]], new_sessions: list[tuple[str, str]] | None = None) -> None:
        super().__init__(splits)
        self.splits = splits
        self.session_to_split = session_to_split
        self.source_session_to_split = source_session_to_split
        self.split_to_sessions = split_to_sessions
        self.split_per_session = session_to_split
        self.image_ids = image_ids_per_split
        self.source_image_ids_per_split = image_ids_per_split
        self.new_sessions = new_sessions or []


def group_images_by_session(items: Sequence[Any]) -> list[SessionGroup]:
    """Group pool image records, objects, or mappings by source and session."""
    groups: dict[tuple[str, str], SessionGroup] = {}
    for item in items:
        if isinstance(item, Mapping):
            source, session = str(item.get("source", "default")), str(item.get("session", "default"))
            image_id, is_test = str(item.get("id") or item.get("image_id") or ""), bool(item.get("is_test_session", False))
        else:
            source, session = str(getattr(item, "source", "default")), str(getattr(item, "session", "default"))
            image_id, is_test = str(getattr(item, "id", getattr(item, "image_id", ""))), bool(getattr(item, "is_test_session", False))
        group = groups.setdefault((source, session), SessionGroup(source, session, is_test))
        group.is_test_session = group.is_test_session or is_test
        if image_id:
            group.image_ids.append(image_id)
        group.records.append(item)
    for group in groups.values():
        if not group.image_ids:
            group.image_ids = [f"{group.source}_{group.session}_{index:04d}" for index in range(len(group.records))]
    return sorted(groups.values(), key=lambda group: group.key)


def _ratios(value: tuple[float, float, float] | Mapping[str, float] | DatasetSplit) -> dict[str, float]:
    if isinstance(value, DatasetSplit):
        return {"train": value.train, "val": value.val, "test": value.test}
    if isinstance(value, Mapping):
        return {split: float(value.get(split, default)) for split, default in zip(_SPLITS, (0.70, 0.15, 0.15), strict=True)}
    return dict(zip(_SPLITS, value, strict=True))


def load_split_registry(path: Path, seed: int) -> dict[str, Any]:
    """Load a registry, returning an empty state when it has not been created yet."""
    if not path.is_file():
        return {"version": 0, "seed": seed, "sessions": {}}
    with path.open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    if not isinstance(registry, dict) or not isinstance(registry.get("sessions"), dict):
        raise TypeError(f"Invalid split registry: {path}")
    registry.setdefault("version", 0)
    registry.setdefault("seed", seed)
    return registry


def save_split_registry(path: Path, registry: Mapping[str, Any]) -> None:
    """Persist a registry outside the disposable dataset directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _choose_split(
    group: SessionGroup,
    counts: Mapping[str, int],
    targets: Mapping[str, float],
    sources_by_split: Mapping[str, set[str]],
    rng: random.Random,
) -> str:
    """Choose the largest deficit, preferring a split new to the source."""
    deficits = {split: targets[split] - counts[split] for split in _SPLITS}
    largest_deficit = max(deficits.values())
    candidates = [
        split for split in _SPLITS if largest_deficit - deficits[split] <= 1.0
    ]
    diverse = [split for split in candidates if group.source not in sources_by_split[split]]
    return rng.choice(diverse or candidates)


def _fill_empty_evaluation_splits(
    assignments: dict[tuple[str, str], str],
    groups: Sequence[SessionGroup],
    known: Mapping[str, Any],
    overrides: Mapping[str, str],
) -> None:
    """Use the smallest movable new sessions to ensure evaluation splits exist."""
    non_test_forced = [group for group in groups if not group.is_test_session]
    if len(non_test_forced) < 3:
        return
    protected: set[tuple[str, str]] = set()
    for empty_split in ("test", "val"):
        if any(assignments[group.key] == empty_split for group in groups):
            continue
        candidates = [
            group
            for group in non_test_forced
            if group.registry_key not in known
            and group.registry_key not in overrides
            and group.key not in protected
            and assignments[group.key] != empty_split
        ]
        if not candidates:
            continue
        group = min(candidates, key=lambda item: (item.count, item.registry_key))
        assignments[group.key] = empty_split
        protected.add(group.key)


def split_sessions(images_or_sessions: Sequence[Any], seed: int = 42, split_ratios: tuple[float, float, float] | Mapping[str, float] | DatasetSplit = (0.70, 0.15, 0.15), *, registry: Mapping[str, Any] | None = None, overrides: Mapping[str, str] | None = None) -> SplitResult:
    """Assign sessions while keeping registry assignments stable across rebuilds."""
    groups = group_images_by_session(images_or_sessions)
    if not groups:
        empty = {split: [] for split in _SPLITS}
        return SplitResult(empty, {}, {}, {split: [] for split in _SPLITS}, empty.copy())
    ratios, known, overrides = _ratios(split_ratios), dict((registry or {}).get("sessions", {})), dict(overrides or {})
    rng, assignments = random.Random(seed), {}
    counts = {split: 0 for split in _SPLITS}
    targets = {split: sum(group.count for group in groups) * ratios[split] for split in _SPLITS}
    sources_by_split = {split: set() for split in _SPLITS}
    for group in groups:
        previous = known.get(group.registry_key, {})
        assigned = overrides.get(group.registry_key) or previous.get("split")
        assigned = assigned if assigned in _SPLITS else None
        if group.is_test_session and assigned != "test":
            if assigned is not None:
                logger.warning("Session %s was reassigned from %s to test", group.registry_key, assigned)
            assigned = "test"
        if assigned is not None:
            assignments[group.key] = assigned
            counts[assigned] += group.count
            sources_by_split[assigned].add(group.source)
    new_groups = sorted(
        (group for group in groups if group.key not in assignments),
        key=lambda group: (-group.count, group.registry_key),
    )
    for group in new_groups:
        split = _choose_split(group, counts, targets, sources_by_split, rng)
        assignments[group.key], counts[split] = split, counts[split] + group.count
        sources_by_split[split].add(group.source)
    _fill_empty_evaluation_splits(assignments, groups, known, overrides)
    new_sessions = [group.key for group in groups if group.registry_key not in known]
    splits, split_to_sessions, image_ids = ({split: [] for split in _SPLITS} for _ in range(3))
    session_to_split: dict[str, str] = {}
    for group in groups:
        split = assignments[group.key]
        splits[split].extend(group.records)
        split_to_sessions[split].append(group.session)
        image_ids[split].extend(group.image_ids)
        session_to_split[group.session] = split
    return SplitResult(splits, session_to_split, assignments, split_to_sessions, image_ids, new_sessions)


def update_split_registry(registry: dict[str, Any], result: SplitResult, groups: Sequence[SessionGroup], seed: int) -> dict[str, Any]:
    """Return the next persistent registry state after a dataset build."""
    registry, sessions, now = dict(registry), dict(registry.get("sessions", {})), datetime.now(UTC).isoformat()
    for group in groups:
        previous = dict(sessions.get(group.registry_key, {}))
        sessions[group.registry_key] = {"split": result.source_session_to_split[group.key], "assigned_at": previous.get("assigned_at", now), "images": group.count}
    registry.update({"version": int(registry.get("version", 0)) + 1, "seed": seed, "sessions": sessions})
    return registry


split_pool_images = split_sessions
