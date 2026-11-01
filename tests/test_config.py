"""Validation coverage for the strict classes and pipeline configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flurseg.config import load_config
from flurseg.paths import RepoPaths


def _load_yaml(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _save_yaml(path: Path, content: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")


def _classes(tmp_root: Path) -> dict[str, object]:
    return _load_yaml(tmp_root / "config" / "classes.yaml")


def _pipeline(tmp_root: Path) -> dict[str, object]:
    return _load_yaml(tmp_root / "config" / "pipeline.yaml")


def _save_classes(tmp_root: Path, content: dict[str, object]) -> None:
    _save_yaml(tmp_root / "config" / "classes.yaml", content)


def _save_pipeline(tmp_root: Path, content: dict[str, object]) -> None:
    _save_yaml(tmp_root / "config" / "pipeline.yaml", content)


def _assert_invalid(tmp_root: Path) -> None:
    with pytest.raises((ValidationError, ValueError)):
        load_config(RepoPaths.from_root(tmp_root))


def test_real_configuration_loads(tmp_root: Path) -> None:
    config = load_config(RepoPaths.from_root(tmp_root))
    assert config.class_by_id(4).name == "tree"
    assert config.class_by_name("vehicle").id == 6
    assert ("tree", 4) in config.prompt_list()
    assert config.priority_rank("vehicle") == 0
    assert config.pipeline.exemplars.find_similar.min_instance_score == 0.5


def test_rejects_duplicate_class_ids(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][1]["id"] = 0
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_duplicate_class_names(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][1]["name"] = "other"
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_out_of_range_class_id(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][0]["id"] = 11
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_incomplete_priority(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["priority"].pop()
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_duplicate_prompts(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][0]["prompts"].append("person")
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_empty_prompts(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][0]["prompts"][0] = ""
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_comma_joined_prompts(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][0]["prompts"][0] = "person, pole"
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_invalid_color_format(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][0]["display_color"] = "red"
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_colors_that_are_too_similar(tmp_root: Path) -> None:
    content = _classes(tmp_root)
    content["classes"][1]["display_color"] = "#9E9E9E"
    _save_classes(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_unknown_scale_class(tmp_root: Path) -> None:
    content = _pipeline(tmp_root)
    content["scales"][0]["classes"][0] = "unknown_class"
    _save_pipeline(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_unknown_priority_override_class(tmp_root: Path) -> None:
    content = _pipeline(tmp_root)
    content["fusion"]["priority_overrides"][0]["winner"] = "unknown_class"
    _save_pipeline(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_threshold_outside_unit_interval(tmp_root: Path) -> None:
    content = _pipeline(tmp_root)
    content["sam3"]["conf"] = 1.01
    _save_pipeline(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_invalid_find_similar_instance_score(tmp_root: Path) -> None:
    content = _pipeline(tmp_root)
    content["exemplars"]["find_similar"]["min_instance_score"] = 1.01
    _save_pipeline(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_overlap_below_minimum(tmp_root: Path) -> None:
    content = _pipeline(tmp_root)
    content["tiling"]["min_overlap"] = 0.24
    _save_pipeline(tmp_root, content)
    _assert_invalid(tmp_root)


def test_rejects_unknown_tta_variant(tmp_root: Path) -> None:
    content = _pipeline(tmp_root)
    content["tta"]["variants"].append("diagonal")
    _save_pipeline(tmp_root, content)
    _assert_invalid(tmp_root)
