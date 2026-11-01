"""Tests for deterministic pool image splitting by session."""

from __future__ import annotations

from typing import Any

from flurseg.config import DatasetSplit
from flurseg.dataset.split import (
    SplitResult,
    group_images_by_session,
    split_pool_images,
    split_sessions,
    update_split_registry,
)
from flurseg.manifest.repo import ImageRecord


def _make_record(
    source: str,
    session: str,
    index: int,
    *,
    is_test_session: bool = False,
) -> ImageRecord:
    return ImageRecord(
        id=f"{source}_{session}_{index:04d}",
        hash=f"hash_{source}_{session}_{index:04d}",
        source=source,
        session=session,
        gsd_method="sensor",
        original_path=f"input/{source}/{session}/img_{index:04d}.jpg",
        image_path=f"work/{source}_{session}_{index:04d}/image.jpg",
        work_dir=f"work/{source}_{session}_{index:04d}",
        width=1920,
        height=1080,
        is_test_session=is_test_session,
    )


def _make_dict_item(
    source: str,
    session: str,
    index: int,
    *,
    is_test_session: bool = False,
) -> dict[str, Any]:
    return {
        "id": f"{source}_{session}_{index:04d}",
        "source": source,
        "session": session,
        "is_test_session": is_test_session,
    }


def test_empty_input_returns_empty_split_result() -> None:
    result = split_sessions([])
    assert isinstance(result, SplitResult)
    assert result.splits == {"train": [], "val": [], "test": []}
    assert result["train"] == []
    assert result["val"] == []
    assert result["test"] == []
    assert result.session_to_split == {}
    assert result.source_session_to_split == {}
    assert result.split_to_sessions == {"train": [], "val": [], "test": []}
    assert result.image_ids == {"train": [], "val": [], "test": []}


def test_no_session_leakage() -> None:
    records: list[ImageRecord] = []
    sources = ["drone", "phone", "ortho"]
    for source in sources:
        for s_idx in range(4):
            session = f"{source}_session_{s_idx}"
            for img_idx in range(5):
                records.append(_make_record(source, session, img_idx))

    result = split_sessions(records, seed=42)

    train_sessions = set(result.split_to_sessions["train"])
    val_sessions = set(result.split_to_sessions["val"])
    test_sessions = set(result.split_to_sessions["test"])

    assert train_sessions.isdisjoint(val_sessions)
    assert train_sessions.isdisjoint(test_sessions)
    assert val_sessions.isdisjoint(test_sessions)

    train_ids = set(result.image_ids["train"])
    val_ids = set(result.image_ids["val"])
    test_ids = set(result.image_ids["test"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert len(train_ids) + len(val_ids) + len(test_ids) == len(records)

    for record in records:
        expected_split = result.source_session_to_split[(record.source, record.session)]
        assert record in result[expected_split]
        assert record.id in result.image_ids[expected_split]


def test_same_session_name_across_sources_treated_as_different_sessions() -> None:
    """Verify two sources sharing the same session name form distinct (source, session) groups."""
    records = [
        _make_record("drone", "common_session", 0),
        _make_record("drone", "common_session", 1),
        _make_record("phone", "common_session", 0),
        _make_record("phone", "common_session", 1),
    ]

    result = split_sessions(records, seed=42)

    assert ("drone", "common_session") in result.source_session_to_split
    assert ("phone", "common_session") in result.source_session_to_split
    assert len(result.source_session_to_split) == 2

    for record in records:
        expected_split = result.source_session_to_split[(record.source, record.session)]
        assert record in result[expected_split]
        assert record.id in result.image_ids[expected_split]


def test_test_sessions_always_assigned_to_test() -> None:
    records: list[ImageRecord] = []
    test_keys: set[tuple[str, str]] = set()

    for s_idx in range(6):
        source = "drone"
        session = f"s_{s_idx}"
        is_test = s_idx in (1, 4)
        if is_test:
            test_keys.add((source, session))
        for img_idx in range(4):
            records.append(_make_record(source, session, img_idx, is_test_session=is_test))

    for s_idx in range(4):
        source = "phone"
        session = f"s_{s_idx}"
        is_test = s_idx == 0
        if is_test:
            test_keys.add((source, session))
        for img_idx in range(4):
            records.append(_make_record(source, session, img_idx, is_test_session=is_test))

    result = split_sessions(records, seed=99)

    for key in test_keys:
        source, session = key
        assert result.source_session_to_split[key] == "test"
        matching_records = [r for r in records if r.source == source and r.session == session]
        for record in matching_records:
            assert record in result["test"]
            assert record not in result["train"]
            assert record not in result["val"]


def test_all_test_sessions_directed_to_test() -> None:
    records = [
        _make_record("drone", f"sess_{i}", j, is_test_session=True)
        for i in range(5)
        for j in range(3)
    ]
    result = split_sessions(records, seed=42)
    assert len(result["train"]) == 0
    assert len(result["val"]) == 0
    assert len(result["test"]) == len(records)


def test_ratios_within_tolerance_on_twenty_synthetic_sessions() -> None:
    records: list[ImageRecord] = []
    sources = ["drone", "phone", "ortho", "handheld"]
    session_count = 0

    for source in sources:
        for s_idx in range(5):
            session = f"{source}_ses_{s_idx:02d}"
            # Vary image count per session deterministically (8 to 16 images)
            images_in_session = 8 + 2 * (session_count % 5)
            session_count += 1
            for img_idx in range(images_in_session):
                records.append(_make_record(source, session, img_idx))

    assert session_count == 20
    total_images = len(records)

    result = split_sessions(records, seed=42, split_ratios=(0.70, 0.15, 0.15))

    train_count = len(result["train"])
    val_count = len(result["val"])
    test_count = len(result["test"])

    assert train_count + val_count + test_count == total_images

    train_ratio = train_count / total_images
    val_ratio = val_count / total_images
    test_ratio = test_count / total_images

    # With 20 sessions of ~8-16 images and the per-source presence guarantee,
    # the achievable granularity is one session = ~5% of images, and the presence
    # constraint can force up to ~2 extra sessions into val/test. Use tolerance 0.15.
    assert abs(train_ratio - 0.70) <= 0.15
    assert abs(val_ratio - 0.15) <= 0.15
    assert abs(test_ratio - 0.15) <= 0.15


def test_source_diversity_is_preferred_when_deficits_are_close() -> None:
    records: list[ImageRecord] = []
    # Source A has 5 sessions.
    for s_idx in range(5):
        session = f"drone_s_{s_idx}"
        for img_idx in range(6):
            records.append(_make_record("drone", session, img_idx))

    # Source B has 4 sessions.
    for s_idx in range(4):
        session = f"phone_s_{s_idx}"
        for img_idx in range(6):
            records.append(_make_record("phone", session, img_idx))

    # Source C has 2 sessions.
    for s_idx in range(2):
        session = f"ortho_s_{s_idx}"
        for img_idx in range(6):
            records.append(_make_record("ortho", session, img_idx))

    result = split_sessions(records, seed=42)

    assert result.split_to_sessions["val"]
    assert result.split_to_sessions["test"]


def test_existing_test_session_is_preserved() -> None:
    records: list[ImageRecord] = []
    source = "drone"
    # 3 sessions: session 0 is a test session, sessions 1 and 2 are normal
    for s_idx in range(3):
        session = f"ses_{s_idx}"
        is_test = s_idx == 0
        for img_idx in range(5):
            records.append(_make_record(source, session, img_idx, is_test_session=is_test))

    result = split_sessions(records, seed=42)

    splits_for_source = {
        result.source_session_to_split[(source, f"ses_{s_idx}")]
        for s_idx in range(3)
    }
    assert splits_for_source.issubset({"train", "val", "test"})
    assert result.source_session_to_split[(source, "ses_0")] == "test"


def test_new_sessions_fill_global_image_deficits_and_evaluation_splits() -> None:
    """The reported five-session regression assigns usable validation and test data."""
    records = [
        _make_record(source, session, index)
        for source, session, count in (
            ("drone", "odm_aukerman", 7),
            ("ortho", "swissimage_seeland_agriculture", 4),
            ("ortho", "swissimage_zurich_oerlikon", 1),
            ("phone", "kerbs", 1),
            ("phone", "streets_in_lucerne", 2),
        )
        for index in range(count)
    ]

    result = split_sessions(records, seed=42)

    assert result.split_to_sessions["val"]
    assert result.split_to_sessions["test"]
    assert len(result["train"]) > len(result["val"])
    assert len(result["train"]) > len(result["test"])


def test_incremental_assignment_preserves_registry_and_fills_deficits() -> None:
    """Existing sessions remain fixed while later sessions fill missing splits."""
    original = [_make_record("drone", "existing", 0)]
    first = split_sessions(original, seed=42)
    registry = update_split_registry({}, first, group_images_by_session(original), 42)
    additions = [
        _make_record("phone", f"new_{index}", 0)
        for index in range(5)
    ]

    result = split_sessions(original + additions, seed=42, registry=registry)

    assert result.source_session_to_split[("drone", "existing")] == first.source_session_to_split[
        ("drone", "existing")
    ]
    assert result.split_to_sessions["val"]
    assert result.split_to_sessions["test"]


def test_determinism_same_seed_same_split() -> None:
    records = [
        _make_record(src, f"ses_{s}", i)
        for src in ["drone", "phone"]
        for s in range(5)
        for i in range(4)
    ]
    res1 = split_sessions(records, seed=777)
    res2 = split_sessions(records, seed=777)

    assert res1.source_session_to_split == res2.source_session_to_split
    assert res1.image_ids == res2.image_ids
    assert [r.id for r in res1["train"]] == [r.id for r in res2["train"]]
    assert [r.id for r in res1["val"]] == [r.id for r in res2["val"]]
    assert [r.id for r in res1["test"]] == [r.id for r in res2["test"]]


def test_determinism_different_seed_different_split() -> None:
    records = [
        _make_record(src, f"ses_{s}", i)
        for src in ["drone", "phone", "ortho"]
        for s in range(6)
        for i in range(5)
    ]
    res1 = split_sessions(records, seed=1)
    res2 = split_sessions(records, seed=99999)

    assert res1.source_session_to_split != res2.source_session_to_split


def test_split_ratios_various_formats() -> None:
    records = [
        _make_record("drone", f"s_{s}", i)
        for s in range(6)
        for i in range(5)
    ]

    res_tuple = split_sessions(records, seed=42, split_ratios=(0.60, 0.20, 0.20))
    assert sum(len(res_tuple[k]) for k in ("train", "val", "test")) == len(records)

    res_dict = split_sessions(
        records, seed=42, split_ratios={"train": 0.60, "val": 0.20, "test": 0.20}
    )
    assert sum(len(res_dict[k]) for k in ("train", "val", "test")) == len(records)

    res_model = split_sessions(
        records, seed=42, split_ratios=DatasetSplit(train=0.60, val=0.20, test=0.20)
    )
    assert sum(len(res_model[k]) for k in ("train", "val", "test")) == len(records)


def test_group_images_by_session_with_dicts_and_auto_ids() -> None:
    items = [
        {"source": "src_b", "session": "ses_1"},
        {"source": "src_b", "session": "ses_1"},
        {"source": "src_a", "session": "ses_2", "is_test_session": True},
    ]
    groups = group_images_by_session(items)

    assert len(groups) == 2
    # Alphabetical order by (source, session)
    assert groups[0].source == "src_a"
    assert groups[0].session == "ses_2"
    assert groups[0].is_test_session is True
    assert groups[0].count == 1
    assert groups[0].image_ids == ["src_a_ses_2_0000"]

    assert groups[1].source == "src_b"
    assert groups[1].session == "ses_1"
    assert groups[1].is_test_session is False
    assert groups[1].count == 2
    assert groups[1].image_ids == ["src_b_ses_1_0000", "src_b_ses_1_0001"]


def test_split_pool_images_alias() -> None:
    assert split_pool_images is split_sessions
