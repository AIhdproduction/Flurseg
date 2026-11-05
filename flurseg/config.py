"""Strict loading and validation for Flurseg YAML configuration."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flurseg.paths import RepoPaths

_ALLOWED_CLASS_IDS = set(range(11)) | {255}
_COLOR_RE = re.compile(r"#[0-9A-Fa-f]{6}")
_ALLOWED_TTA_VARIANTS = {"identity", "rot90", "hflip", "vflip"}


class StrictModel(BaseModel):
    """Base model that rejects unknown configuration keys."""

    model_config = ConfigDict(extra="forbid")


class ClassDefinition(StrictModel):
    """One semantic label and its SAM 3 prompts."""

    id: int
    name: str = Field(min_length=1)
    prompts: list[str]
    includes: str
    excludes: str
    cyclone_code: int | str
    display_color: str
    trainable: bool

    @field_validator("prompts")
    @classmethod
    def validate_prompts(cls, prompts: list[str]) -> list[str]:
        if any(not prompt.strip() for prompt in prompts):
            raise ValueError("prompts must not be empty")
        if any("," in prompt for prompt in prompts):
            raise ValueError("prompts must be separate items and must not contain commas")
        if len(set(prompts)) != len(prompts):
            raise ValueError("prompts must not contain duplicates")
        return prompts

    @field_validator("display_color")
    @classmethod
    def validate_color(cls, color: str) -> str:
        if not _COLOR_RE.fullmatch(color):
            raise ValueError("display_color must use #RRGGBB")
        return color.upper()


class ClassesConfig(StrictModel):
    """Class catalogue, draw priority, and review rules."""

    classes: list[ClassDefinition]
    priority: list[str]
    labeling_rules: list[str]

    @model_validator(mode="after")
    def validate_catalogue(self) -> ClassesConfig:
        ids = [item.id for item in self.classes]
        names = [item.name for item in self.classes]
        if any(class_id not in _ALLOWED_CLASS_IDS for class_id in ids):
            raise ValueError("class IDs must be in 0..10 or be ignore ID 255")
        if len(set(ids)) != len(ids):
            raise ValueError("class IDs must be unique")
        if len(set(names)) != len(names):
            raise ValueError("class names must be unique")
        if set(ids) != _ALLOWED_CLASS_IDS:
            raise ValueError("classes must define IDs 0..10 and ignore ID 255 exactly once")

        all_prompts = [prompt for item in self.classes for prompt in item.prompts]
        if len(set(all_prompts)) != len(all_prompts):
            raise ValueError("prompts must be unique across classes")

        trainable_names = {item.name for item in self.classes if item.trainable}
        if len(self.priority) != len(set(self.priority)) or set(self.priority) != trainable_names:
            raise ValueError("priority must contain every trainable class exactly once")
        if any(not rule.strip() for rule in self.labeling_rules):
            raise ValueError("labeling_rules must not contain empty entries")
        self._validate_color_distance()
        return self

    def _validate_color_distance(self) -> None:
        for index, first in enumerate(self.classes):
            for second in self.classes[index + 1 :]:
                first_rgb = tuple(int(first.display_color[offset : offset + 2], 16) for offset in (1, 3, 5))
                second_rgb = tuple(int(second.display_color[offset : offset + 2], 16) for offset in (1, 3, 5))
                distance = math.dist(first_rgb, second_rgb)
                if distance < 60:
                    raise ValueError(
                        f"class colors for {first.name} and {second.name} are too similar ({distance:.1f})"
                    )


class Sam3Config(StrictModel):
    checkpoint_path: str = Field(min_length=1)
    imgsz: int = Field(gt=0)
    precision: str
    conf: float = Field(ge=0, le=1)
    query_threshold: float = Field(ge=0, le=1)
    query_fusion_chunk_size: int = Field(default=128, gt=0)
    prompt_chunk_size: int = Field(gt=0)
    max_vram_gb: float = Field(gt=0)
    slow_chunk_min_seconds: float = Field(default=2.0, gt=0)

    @field_validator("precision")
    @classmethod
    def validate_precision(cls, precision: str) -> str:
        if precision not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("precision must be auto, fp32, fp16, or bf16")
        return precision


class TilingConfig(StrictModel):
    tile_size: int = Field(gt=0)
    min_overlap: float = Field(ge=0.05, lt=1)


class ScaleAssignment(StrictModel):
    factor: float = Field(gt=0)
    classes: list[str] = Field(min_length=1)

    @field_validator("classes")
    @classmethod
    def validate_classes(cls, classes: list[str]) -> list[str]:
        if len(classes) != len(set(classes)):
            raise ValueError("scale classes must not contain duplicates")
        return classes


class TtaConfig(StrictModel):
    variants: list[str] = Field(min_length=1)

    @field_validator("variants")
    @classmethod
    def validate_variants(cls, variants: list[str]) -> list[str]:
        unknown = set(variants) - _ALLOWED_TTA_VARIANTS
        if unknown:
            raise ValueError(f"unknown TTA variants: {', '.join(sorted(unknown))}")
        if len(variants) != len(set(variants)):
            raise ValueError("TTA variants must not contain duplicates")
        return variants


class MosaicConfig(StrictModel):
    enabled: bool
    strip_fraction: float = Field(gt=0, lt=1)
    max_per_class: int = Field(gt=0)


class FindSimilarConfig(StrictModel):
    enabled: bool
    min_instance_score: float = Field(default=0.5, ge=0, le=1)


class ExemplarsConfig(StrictModel):
    mosaic: MosaicConfig
    find_similar: FindSimilarConfig


class AnonymizationConfig(StrictModel):
    enabled_for: list[str] = Field(min_length=1)
    prompts: list[str] = Field(min_length=1)
    blur_kernel: int = Field(gt=0)
    blur_sigma: float = Field(ge=0)
    min_score: float = Field(ge=0, le=1)

    @field_validator("blur_kernel")
    @classmethod
    def validate_blur_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("blur_kernel must be odd")
        return value


class IngestConfig(StrictModel):
    """Canonical output options used during source normalization."""

    jpeg_quality: int = Field(default=95, ge=1, le=100)


class OrthoConfig(StrictModel):
    tile_size: int = Field(gt=0)
    max_nodata_fraction: float = Field(ge=0, le=1)
    black_threshold: int = Field(ge=0, le=255)
    gsd_m: float = Field(gt=0)


class PriorityOverride(StrictModel):
    winner: str = Field(min_length=1)
    loser: str = Field(min_length=1)
    min_winner_score: float = Field(ge=0, le=1)


class FusionConfig(StrictModel):
    priority_overrides: list[PriorityOverride]
    presence_threshold: float = Field(default=0.5, ge=0, le=1)
    presence_threshold_per_class: dict[str, float] = Field(default_factory=dict)
    priority_margin: float = Field(default=0.15, ge=0, le=1)
    no_prompt_score: float = Field(ge=0, le=1)


class EdgeSnapConfig(StrictModel):
    rule_id: Literal[1]
    enabled: bool
    slic_n_segments_per_mp: int = Field(gt=0)
    compactness: float = Field(gt=0)
    protected_classes: list[str]
    canny_low_threshold: int = Field(default=50, ge=0, le=255)
    canny_high_threshold: int = Field(default=150, ge=0, le=255)
    max_edge_fraction: float = Field(default=0.35, ge=0, le=1)


class MorphologyConfig(StrictModel):
    rule_id: Literal[2]
    enabled: bool
    min_island_m2: dict[str, float]
    min_component_px: dict[str, int]
    max_hole_m2: float = Field(ge=0)

    @field_validator("min_island_m2")
    @classmethod
    def validate_island_areas(cls, values: dict[str, float]) -> dict[str, float]:
        if any(area < 0 for area in values.values()):
            raise ValueError("min_island_m2 values must be non-negative")
        return values

    @field_validator("min_component_px")
    @classmethod
    def validate_component_sizes(cls, values: dict[str, int]) -> dict[str, int]:
        if any(area < 0 for area in values.values()):
            raise ValueError("min_component_px values must be non-negative")
        return values


class ShapeConfig(StrictModel):
    rule_id: Literal[3]
    enabled: bool
    utility_cover_diameter_m: tuple[float, float]
    utility_cover_min_compactness: float = Field(ge=0, le=1)
    vehicle_area_m2: tuple[float, float]

    @model_validator(mode="after")
    def validate_ranges(self) -> ShapeConfig:
        if self.utility_cover_diameter_m[0] <= 0 or self.utility_cover_diameter_m[0] > self.utility_cover_diameter_m[1]:
            raise ValueError("utility_cover_diameter_m must be an ascending positive range")
        if self.vehicle_area_m2[0] <= 0 or self.vehicle_area_m2[0] > self.vehicle_area_m2[1]:
            raise ValueError("vehicle_area_m2 must be an ascending positive range")
        return self


class ContextConfig(StrictModel):
    rule_id: Literal[4]
    enabled: bool
    vehicle_ring_width_px: int = Field(default=3, gt=0)
    vehicle_on_building_ratio: float = Field(default=0.6, ge=0, le=1)
    utility_cover_ring_width_px: int = Field(default=5, gt=0)
    utility_cover_ground_ratio: float = Field(default=0.5, ge=0, le=1)
    marking_paved_ring_ratio: float = Field(default=0.3, ge=0, le=1)
    marking_ring_width_px: int = Field(default=3, gt=0)
    curb_max_distance_px: int = Field(default=5, ge=0)


class ColorConfig(StrictModel):
    rule_id: Literal[5]
    enabled: bool
    exg_threshold: float = Field(default=0.15, ge=-1, le=2)
    exg_low_threshold: float = Field(default=-0.02, ge=-1, le=2)
    min_brightness: float = Field(default=70, ge=0, le=255)
    min_target_score: float = Field(default=0.25, ge=0, le=1)
    min_region_px: int = Field(default=3000, ge=1)


class OrthoRefinementConfig(StrictModel):
    rule_id: Literal[6]
    enabled: bool
    min_small_object_m: float = Field(gt=0)
    min_marking_width_m: float = Field(gt=0)


class RefinementConfig(StrictModel):
    edge_snap: EdgeSnapConfig
    morphology: MorphologyConfig
    shape: ShapeConfig
    context: ContextConfig
    color: ColorConfig
    ortho: OrthoRefinementConfig


class UncertaintyWeights(StrictModel):
    top_score: float = Field(ge=0)
    margin: float = Field(ge=0)
    priority_conflict: float = Field(default=0.3, ge=0)
    tta: float = Field(ge=0)
    rule_change: float = Field(ge=0)


class UncertaintyConfig(StrictModel):
    weights: UncertaintyWeights
    threshold: float = Field(ge=0, le=1)
    min_region_px: int = Field(default=64, ge=1)


class RoutingConfig(StrictModel):
    auto_accept_max_uncertain_fraction: float = Field(default=0.06, ge=0, le=1)
    auto_accept_max_ignore_fraction: float = Field(default=0.15, ge=0, le=1)
    spot_check_fraction: float = Field(default=0.04, ge=0, le=1)
    seed: int


class ReviewConfig(StrictModel):
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    open_browser: bool
    coarse_superpixels_per_mp: int = Field(default=150, gt=0)


class DatasetSplit(StrictModel):
    train: float = Field(ge=0, le=1)
    val: float = Field(ge=0, le=1)
    test: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_total(self) -> DatasetSplit:
        if not math.isclose(self.train + self.val + self.test, 1.0, abs_tol=1e-9):
            raise ValueError("dataset split fractions must sum to 1")
        return self


class DatasetConfig(StrictModel):
    tile_size: int = Field(gt=0)
    max_ignore_fraction: float = Field(ge=0, le=1)
    split: DatasetSplit
    seed: int


class ReportConfig(StrictModel):
    spot_check_error_limit: float = Field(ge=0, le=1)


class PipelineConfig(StrictModel):
    sam3: Sam3Config
    tiling: TilingConfig
    scales: list[ScaleAssignment] = Field(min_length=1)
    tta: TtaConfig
    exemplars: ExemplarsConfig
    anonymization: AnonymizationConfig
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    ortho: OrthoConfig
    fusion: FusionConfig
    refinement: RefinementConfig
    uncertainty: UncertaintyConfig
    routing: RoutingConfig
    review: ReviewConfig
    dataset: DatasetConfig
    report: ReportConfig

    @model_validator(mode="after")
    def validate_coarse_superpixels(self) -> PipelineConfig:
        if self.review.coarse_superpixels_per_mp >= self.refinement.edge_snap.slic_n_segments_per_mp:
            raise ValueError(
                "review.coarse_superpixels_per_mp must be less than refinement.edge_snap.slic_n_segments_per_mp"
            )
        return self


class FlurConfig(StrictModel):
    """Validated combination of the two repository configuration files."""

    classes: ClassesConfig
    pipeline: PipelineConfig

    def class_by_id(self, class_id: int) -> ClassDefinition:
        """Return a class definition by its numeric mask value."""
        for class_definition in self.classes.classes:
            if class_definition.id == class_id:
                return class_definition
        raise KeyError(f"unknown class ID: {class_id}")

    def class_by_name(self, name: str) -> ClassDefinition:
        """Return a class definition by its stable name."""
        for class_definition in self.classes.classes:
            if class_definition.name == name:
                return class_definition
        raise KeyError(f"unknown class name: {name}")

    def prompt_list(self) -> list[tuple[str, int]]:
        """Return the flat SAM prompt-to-class-ID mapping."""
        return [(prompt, item.id) for item in self.classes.classes for prompt in item.prompts]

    def priority_rank(self, class_name: str) -> int:
        """Return a lower-is-higher priority rank for a trainable class."""
        try:
            return self.classes.priority.index(class_name)
        except ValueError as error:
            raise KeyError(f"class has no trainable priority: {class_name}") from error


def _read_yaml(path: Path) -> object:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _validate_references(classes: ClassesConfig, pipeline: PipelineConfig) -> None:
    known_classes = {item.name for item in classes.classes}
    references = [
        reference
        for scale in pipeline.scales
        for reference in scale.classes
    ]
    references.extend(
        reference
        for override in pipeline.fusion.priority_overrides
        for reference in (override.winner, override.loser)
    )
    references.extend(pipeline.fusion.presence_threshold_per_class)
    if unknown := sorted(set(references) - known_classes):
        raise ValueError(f"pipeline references unknown classes: {', '.join(unknown)}")


def load_config(paths: RepoPaths) -> FlurConfig:
    """Load, strictly validate, and cross-check repository YAML configuration."""
    classes = ClassesConfig.model_validate(_read_yaml(paths.classes_config))
    pipeline = PipelineConfig.model_validate(_read_yaml(paths.pipeline_config))
    _validate_references(classes, pipeline)
    return FlurConfig(classes=classes, pipeline=pipeline)
