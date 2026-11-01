"""Streaming SHA-256 coverage."""

from __future__ import annotations

import hashlib
from pathlib import Path

from flurseg.io.hashing import sha256_file


def test_sha256_matches_known_fixed_value(tmp_path: Path) -> None:
    path = tmp_path / "fixed.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_sha256_streams_across_chunk_boundaries(tmp_path: Path) -> None:
    content = b"prefix" + bytes(range(256)) * 257 + b"suffix"
    path = tmp_path / "large.bin"
    path.write_bytes(content)
    assert sha256_file(path, chunk_size=257) == hashlib.sha256(content).hexdigest()
