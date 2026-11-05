"""Repository-relative paths and safe runtime-directory creation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepoPaths:
    """Resolved paths owned by one Flurseg repository."""

    root: Path

    @classmethod
    def from_root(cls, root: Path | str | None = None) -> RepoPaths:
        """Resolve an explicit root, FLURSEG_ROOT, or the package's repository root."""
        selected_root = root if root is not None else os.environ.get("FLURSEG_ROOT")
        if selected_root is None:
            selected_root = Path(__file__).resolve().parents[1]
        return cls(Path(selected_root).expanduser().resolve())

    @property
    def config_dir(self) -> Path:
        return self.root / "config"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    @property
    def package_dir(self) -> Path:
        return self.root / "flurseg"

    @property
    def tests_dir(self) -> Path:
        return self.root / "tests"

    @property
    def classes_config(self) -> Path:
        return self.config_dir / "classes.yaml"

    @property
    def pipeline_config(self) -> Path:
        return self.config_dir / "pipeline.yaml"

    @property
    def samples_config(self) -> Path:
        return self.config_dir / "samples.yaml"

    @property
    def extra_sources_config(self) -> Path:
        return self.config_dir / "extra_sources.yaml"

    @property
    def exemplars_dir(self) -> Path:
        return self.config_dir / "exemplars"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def sam3_checkpoint(self) -> Path:
        return self.models_dir / "sam3.pt"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def drone_input_dir(self) -> Path:
        return self.input_dir / "drone"

    @property
    def phone_input_dir(self) -> Path:
        return self.input_dir / "phone"

    @property
    def ortho_input_dir(self) -> Path:
        return self.input_dir / "ortho"

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    @property
    def quarantine_dir(self) -> Path:
        """Return the local quarantine directory for reversible file operations."""
        return self.root / "quarantine"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def review_dir(self) -> Path:
        return self.root / "review"

    @property
    def train_dir(self) -> Path:
        return self.root / "train"

    @property
    def pool_dir(self) -> Path:
        return self.train_dir / "pool"

    @property
    def pool_images_dir(self) -> Path:
        return self.pool_dir / "images"

    @property
    def pool_masks_dir(self) -> Path:
        return self.pool_dir / "masks"

    @property
    def dataset_dir(self) -> Path:
        return self.train_dir / "dataset"

    @property
    def runs_dir(self) -> Path:
        return self.train_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def log_file(self) -> Path:
        return self.logs_dir / "flurseg.log"

    @property
    def database(self) -> Path:
        return self.root / "flurseg.db"

    @property
    def setup_script(self) -> Path:
        return self.root / "setup.bat"

    @property
    def prelabel_script(self) -> Path:
        return self.root / "prelabel.bat"

    @property
    def review_script(self) -> Path:
        return self.root / "review.bat"

    @property
    def build_dataset_script(self) -> Path:
        return self.root / "build_dataset.bat"

    @property
    def train_script(self) -> Path:
        return self.root / "train.bat"

    @property
    def report_script(self) -> Path:
        return self.root / "report.bat"

    @property
    def selftest_script(self) -> Path:
        return self.root / "selftest.bat"

    @property
    def fetch_samples_script(self) -> Path:
        return self.root / "fetch_samples.bat"

    @property
    def fetch_extra_script(self) -> Path:
        return self.root / "fetch_extra.bat"

    @property
    def dedup_script(self) -> Path:
        return self.root / "dedup.bat"

    @property
    def license_file(self) -> Path:
        return self.root / "LICENSE"

    @property
    def notice_file(self) -> Path:
        return self.root / "NOTICE.md"

    @property
    def readme_file(self) -> Path:
        return self.root / "README.md"

    @property
    def requirements_file(self) -> Path:
        return self.root / "requirements.txt"

    @property
    def pyproject_file(self) -> Path:
        return self.root / "pyproject.toml"

    @property
    def architecture_doc(self) -> Path:
        return self.docs_dir / "architecture.md"

    @property
    def runtime_dirs(self) -> tuple[Path, ...]:
        """Directories created only when Flurseg needs runtime storage."""
        return (
            self.models_dir,
            self.input_dir,
            self.drone_input_dir,
            self.phone_input_dir,
            self.ortho_input_dir,
            self.archive_dir,
            self.quarantine_dir,
            self.work_dir,
            self.review_dir,
            self.train_dir,
            self.pool_dir,
            self.pool_images_dir,
            self.pool_masks_dir,
            self.dataset_dir,
            self.runs_dir,
            self.reports_dir,
            self.logs_dir,
        )

    def ensure_runtime_dirs(self) -> None:
        """Create the repository's runtime directories without touching any other path."""
        for directory in self.runtime_dirs:
            directory.mkdir(parents=True, exist_ok=True)


def _strip_extended_prefix(path_str: str) -> str:
    """Strip extended-length prefixes on Windows (\\\\?\\ and \\\\?\\UNC\\)."""
    if path_str.startswith(("\\\\?\\UNC\\", "//?/UNC/", "\\\\?\\unc\\", "//?/unc/")):
        return "\\\\" + path_str[8:]
    if path_str.startswith(("\\\\?\\", "//?/")):
        return path_str[4:]
    return path_str


def _normalize_path(path: Path | str, *, base: Path | str | None = None) -> Path:
    """Normalize a path resolving deepest existing ancestor and stripping extended prefixes."""
    path_obj = Path(path)
    clean_str = _strip_extended_prefix(str(path_obj))
    path_obj = Path(clean_str)

    if not path_obj.is_absolute():
        anchor = Path(base) if base is not None else Path.cwd()
        path_obj = anchor / path_obj

    current = path_obj
    remaining: list[str] = []
    while True:
        try:
            if current.exists() or current.is_symlink():
                break
        except OSError:
            pass
        parent = current.parent
        if parent == current:
            break
        remaining.append(current.name)
        current = parent

    resolved_ancestor_str = _strip_extended_prefix(
        os.path.realpath(str(current), strict=False)
    )
    resolved_path = Path(resolved_ancestor_str)
    for part in reversed(remaining):
        resolved_path = resolved_path / part

    normalized_str = _strip_extended_prefix(os.path.normpath(str(resolved_path)))
    return Path(normalized_str)


def ensure_inside(path: Path | str, root: Path | str) -> Path:
    """Ensure *path* is inside *root*, returning the normalised absolute Path or raising ValueError."""
    norm_root = _normalize_path(root)
    norm_path = _normalize_path(path, base=norm_root)

    case_path = os.path.normcase(str(norm_path))
    case_root = os.path.normcase(str(norm_root))

    try:
        common = os.path.commonpath([case_path, case_root])
    except ValueError as error:
        raise ValueError(f"path {path!s} is outside root {root!s}: {error}") from error

    if common != case_root:
        raise ValueError(f"path {path!s} is outside root {root!s}")

    return norm_path


def is_inside(path: Path | str, root: Path | str) -> bool:
    """Return True if *path* is inside *root*, False otherwise."""
    try:
        ensure_inside(path, root)
        return True
    except (ValueError, OSError):
        return False
