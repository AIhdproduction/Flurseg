"""GSD precedence and formula coverage."""

from __future__ import annotations

import pytest

from flurseg.ingest.gsd import resolve_gsd
from flurseg.ingest.session import SessionMetadata


def test_session_yaml_gsd_beats_exif_metadata() -> None:
    metadata = {"relative_altitude_m": 50.0, "focal_length_mm": 24.0, "focal_length_35mm": 24.0}
    result = resolve_gsd("drone", SessionMetadata(gsd_m=0.0125), metadata, 5472)
    assert result.gsd_m == 0.0125
    assert result.method == "session_yaml"


def test_exif_xmp_gsd_uses_35mm_sensor_width_formula() -> None:
    metadata = {"relative_altitude_m": 50.0, "focal_length_mm": 24.0, "focal_length_35mm": 24.0}
    result = resolve_gsd("drone", SessionMetadata(), metadata, 5472)
    assert result.gsd_m == pytest.approx(36 * 50 / (24 * 5472))
    assert result.method == "exif_xmp"


def test_exif_xmp_gsd_uses_focal_plane_resolution_formula() -> None:
    metadata = {
        "relative_altitude_m": 50.0,
        "focal_length_mm": 24.0,
        "focal_plane_x_resolution": 1000.0,
        "focal_plane_resolution_unit": 4.0,
    }
    result = resolve_gsd("drone", SessionMetadata(), metadata, 5472)
    sensor_width = 5472 / 1000
    assert result.gsd_m == pytest.approx(sensor_width * 50 / (24 * 5472))
    assert result.method == "exif_xmp"


def test_phone_gsd_is_always_unknown() -> None:
    result = resolve_gsd("phone", SessionMetadata(gsd_m=0.01), {}, 100)
    assert (result.gsd_m, result.method) == (None, "unknown")


def test_ortho_gsd_is_fixed() -> None:
    result = resolve_gsd("ortho", SessionMetadata(), {}, 100)
    assert (result.gsd_m, result.method) == (0.10, "fixed_ortho")


def test_drone_without_metadata_is_unknown() -> None:
    result = resolve_gsd("drone", SessionMetadata(), {}, 100)
    assert (result.gsd_m, result.method) == (None, "unknown")
