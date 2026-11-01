"""Ground-sampling-distance resolution for each source type."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from flurseg.ingest.session import SessionMetadata


@dataclass(frozen=True, slots=True)
class GsdResult:
    """Resolved ground sampling distance and the evidence used."""

    gsd_m: float | None
    method: str


def resolve_gsd(
    source: str, session: SessionMetadata, metadata: Mapping[str, float | str], image_width_px: int,
    *, ortho_gsd_m: float = 0.10,
) -> GsdResult:
    """Resolve GSD according to session, EXIF/XMP, source, then unknown priority."""
    if source == "phone":
        return GsdResult(None, "unknown")
    if session.gsd_m is not None:
        return GsdResult(session.gsd_m, "session_yaml")
    if source == "ortho":
        return GsdResult(ortho_gsd_m, "fixed_ortho")
    altitude = metadata.get("relative_altitude_m")
    focal = metadata.get("focal_length_mm")
    if not isinstance(altitude, (int, float)) or not isinstance(focal, (int, float)) or focal <= 0:
        return GsdResult(None, "unknown")
    sensor_width: float | None = None
    focal_35 = metadata.get("focal_length_35mm")
    if isinstance(focal_35, (int, float)) and focal_35 > 0:
        sensor_width = 36 * float(focal) / float(focal_35)
    else:
        resolution = metadata.get("focal_plane_x_resolution")
        unit = metadata.get("focal_plane_resolution_unit")
        unit_mm = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}.get(int(unit)) if isinstance(unit, (int, float)) else None
        if isinstance(resolution, (int, float)) and resolution > 0 and unit_mm:
            sensor_width = image_width_px / float(resolution) * unit_mm
    if sensor_width is None:
        return GsdResult(None, "unknown")
    return GsdResult(sensor_width * float(altitude) / (float(focal) * image_width_px), "exif_xmp")
