"""Pydantic configuration models for ``config/extra_sources.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from flurseg.ingest.session import SessionMetadata
from flurseg.samples.extra.licences import is_acceptable_licence

SourceName = Literal["drone", "phone", "ortho"]
AreaType = Literal["construction_site", "street", "residential", "urban", "agriculture", "forest_water"]


class SessionYaml(BaseModel):
    """Session fields controlled by a source entry."""

    model_config = ConfigDict(extra="forbid")
    camera: str | None = None
    date: str | None = None
    area_type: AreaType
    gsd_m: float | None = Field(default=None, gt=0)
    test: bool = False

    def metadata(self, *, gsd_m: float, notes: str) -> SessionMetadata:
        """Create validated ingest metadata with the actual output GSD."""
        return SessionMetadata(
            camera=self.camera,
            date=self.date,
            area_type=self.area_type,
            gsd_m=gsd_m,
            test=self.test,
            notes=notes,
        )


class CommonEntry(BaseModel):
    """Fields shared by every additional source type."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    source: SourceName
    session: str = Field(pattern=r"^ext_[a-z0-9][a-z0-9_-]*$")
    licence: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    session_yaml: SessionYaml
    enabled: bool = True
    min_interval_s: float = Field(default=0.5, ge=0.5)
    max_per_entry: int = Field(default=6, ge=1)

class WindowEntry(CommonEntry):
    """Shared model for sources accepting WGS84 centres or a sampled bbox."""

    windows: list[tuple[float, float]] | None = None
    bbox: tuple[float, float, float, float] | None = None
    count: int | None = Field(default=None, ge=1)
    gsd_m: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_windows(self) -> WindowEntry:
        if bool(self.windows) == bool(self.bbox):
            raise ValueError("provide exactly one of windows or bbox")
        if self.bbox and (self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]):
            raise ValueError("bbox must be west,south,east,north")
        if self.bbox and self.count is None:
            raise ValueError("bbox requires count")
        return self


class WmsEntry(WindowEntry):
    type: Literal["wms"]
    url: str
    layer: str
    crs: str = Field(pattern=r"^EPSG:\d+$")
    axis_order: Literal["en", "ne"] = "en"
    format: str = "image/jpeg"
    max_request_px: int = Field(default=2048, ge=1, le=2048)


class XyzEntry(CommonEntry):
    type: Literal["xyz"]
    url: str
    zoom: int = Field(ge=0, le=24)
    windows: list[tuple[float, float]] | None = None
    bbox: tuple[float, float, float, float] | None = None
    count: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_windows(self) -> XyzEntry:
        if bool(self.windows) == bool(self.bbox):
            raise ValueError("provide exactly one of windows or bbox")
        if self.bbox and self.count is None:
            raise ValueError("bbox requires count")
        return self


class ArcgisEntry(WindowEntry):
    type: Literal["arcgis_export"]
    url: str
    crs: str = Field(default="EPSG:3857", pattern=r"^EPSG:\d+$")
    format: str = "jpg"


class StacStaticEntry(CommonEntry):
    type: Literal["stac_static"]
    url: str
    bboxes: list[tuple[float, float, float, float]] = Field(min_length=1)
    target_gsd_m: float | None = Field(default=None, gt=0)


class StacSearchEntry(CommonEntry):
    type: Literal["stac_search"]
    url: str
    collection: str = "openaerialmap"
    bboxes: list[tuple[float, float, float, float]] = Field(min_length=1)
    max_items: int = Field(default=8, ge=1)
    windows_per_item: int = Field(default=2, ge=1)
    max_gsd_m: float | None = Field(default=None, gt=0)
    min_gsd_m: float | None = Field(default=None, gt=0)


class PanoramaxEntry(CommonEntry):
    type: Literal["panoramax"]
    bbox: tuple[float, float, float, float]
    instance: str = "https://api.panoramax.xyz/api"
    mode: Literal["search", "collections_prefilter"] = "search"
    flat_only: bool = False
    max_collection_pages: int = Field(default=30, ge=1)
    min_spacing_m: float = Field(default=30.0, gt=0)
    views: int = Field(default=4, ge=1, le=36)
    fov_deg: float = Field(default=90.0, gt=1, lt=180)
    pitch_deg: float = Field(default=-8.0, ge=-89, le=89)
    crop_size: tuple[int, int] = (1600, 1200)

    @model_validator(mode="after")
    def validate_bbox(self) -> PanoramaxEntry:
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            raise ValueError("bbox must be west,south,east,north")
        return self


class AmsterdamPanoramaEntry(CommonEntry):
    type: Literal["amsterdam_panorama"]
    near: tuple[float, float]
    radius_m: float = Field(default=500.0, gt=0)
    min_spacing_m: float = Field(default=25.0, gt=0)
    views: int = Field(default=4, ge=1, le=36)
    fov_deg: float = Field(default=90.0, gt=1, lt=180)
    pitch_deg: float = Field(default=-8.0, ge=-89, le=89)
    crop_size: tuple[int, int] = (1600, 1200)


class InternetArchiveEntry(CommonEntry):
    type: Literal["internet_archive"]
    search_query: str | None = None
    creator: str | None = None
    identifier_glob: str | None = None
    max_videos: int = Field(default=25, ge=1)
    max_frames_per_video: int = Field(default=40, ge=1)
    frame_interval_s: float = Field(default=2.0, gt=0)
    prefer: Literal["original", "derivative"] = "derivative"
    max_total_gb: float = Field(default=2.0, gt=0)
    max_download_gb: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def validate_search(self) -> InternetArchiveEntry:
        if not self.search_query and not self.identifier_glob:
            raise ValueError("provide search_query or identifier_glob")
        return self


class ArchiveDatasetEntry(CommonEntry):
    type: Literal["archive_dataset"]
    url: str | None = None
    urls: list[str] = Field(default_factory=list)
    figshare_article: int | None = None
    zenodo_record: int | None = None
    hf_dataset: str | None = None
    expected_licence: str
    subset: str | None = None
    max_images: int = Field(default=300, ge=1)
    max_per_group: int | None = Field(default=None, ge=1)
    min_width: int = Field(default=1, ge=1)
    max_download_gb: float = Field(default=3.0, gt=0)

    @model_validator(mode="after")
    def validate_location(self) -> ArchiveDatasetEntry:
        choices = int(bool(self.url or self.urls)) + int(self.figshare_article is not None) + int(
            self.zenodo_record is not None
        ) + int(self.hf_dataset is not None)
        if choices != 1:
            raise ValueError("provide exactly one archive location")
        return self


class VegbilderBbox(BaseModel):
    """One named WGS84 Vegbilder sampling region."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_bbox(self) -> VegbilderBbox:
        if self.bbox[0] >= self.bbox[2] or self.bbox[1] >= self.bbox[3]:
            raise ValueError("bbox must be west,south,east,north")
        return self


class VegbilderWfsEntry(CommonEntry):
    """Planar road imagery from Statens vegvesen's Vegbilder WFS."""

    type: Literal["vegbilder_wfs"]
    url: str
    layer_year: list[int] = Field(min_length=1)
    bboxes: list[VegbilderBbox] = Field(min_length=1)
    per_bbox: int = Field(default=40, ge=1)
    min_spacing_m: float = Field(default=80.0, gt=0)
    heading_tolerance: float | None = Field(default=None, gt=0, le=180)


class CommonsCategoriesEntry(CommonEntry):
    type: Literal["commons_categories"]
    categories: list[str] = Field(min_length=1)
    depth: int = Field(default=1, ge=0, le=4)
    min_width: int = Field(default=1600, ge=1)


class CommonsVideoFramesEntry(CommonEntry):
    type: Literal["commons_video_frames"]
    categories: list[str] = Field(min_length=1)
    exclude_categories: list[str] = Field(default_factory=list)
    depth: int = Field(default=1, ge=0, le=4)
    frame_interval_s: float = Field(default=2.0, gt=0)
    max_videos: int = Field(default=30, ge=1)
    max_frames_per_video: int = Field(default=40, ge=1)


Entry = Annotated[
    WmsEntry
    | XyzEntry
    | ArcgisEntry
    | StacStaticEntry
    | StacSearchEntry
    | PanoramaxEntry
    | AmsterdamPanoramaEntry
    | CommonsCategoriesEntry
    | CommonsVideoFramesEntry
    | InternetArchiveEntry
    | ArchiveDatasetEntry
    | VegbilderWfsEntry,
    Field(discriminator="type"),
]


class ExtraSourcesConfig(BaseModel):
    """Top-level source configuration."""

    model_config = ConfigDict(extra="forbid")
    sources: list[Entry] = Field(min_length=1)
    licence_overrides: tuple[str, ...] = ()

    @model_validator(mode="after")
    def unique_ids(self) -> ExtraSourcesConfig:
        ids = [entry.id for entry in self.sources]
        duplicates = sorted({entry_id for entry_id in ids if ids.count(entry_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate source id(s): {', '.join(duplicates)}")
        for entry in self.sources:
            if entry.enabled and not is_acceptable_licence(entry.licence, self.licence_overrides):
                raise ValueError(f"licence is not accepted by the training-data policy: {entry.licence}")
        return self


def load_extra_config(path: Path) -> ExtraSourcesConfig:
    """Read and validate an extra-source YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ExtraSourcesConfig.model_validate(raw)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ValueError(f"invalid extra sources configuration {path}: {error}") from error
