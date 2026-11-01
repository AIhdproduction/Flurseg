"""Source-folder candidate discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from flurseg.paths import RepoPaths

_EXTENSIONS = {
    "drone": {".jpg", ".jpeg", ".png", ".tif"},
    "phone": {".jpg", ".jpeg", ".png", ".heic", ".heif"},
    "ortho": {".tif", ".tiff"},
}


@dataclass(frozen=True, slots=True)
class Candidate:
    """One supported raw input together with its source/session identity."""

    path: Path
    source: str
    session: str


def discover(paths: RepoPaths) -> list[Candidate]:
    """Return all supported candidates in stable source/session/path order."""
    candidates: list[Candidate] = []
    for source, extensions in _EXTENSIONS.items():
        source_directory = paths.input_dir / source
        if not source_directory.exists():
            continue
        for session_directory in sorted((path for path in source_directory.iterdir() if path.is_dir()), key=lambda path: path.name):
            for path in sorted(session_directory.iterdir(), key=lambda item: item.name.lower()):
                if path.name.lower() == "session.yaml" or not path.is_file() or path.suffix.lower() not in extensions:
                    continue
                candidates.append(Candidate(path=path, source=source, session=session_directory.name))
    return candidates
