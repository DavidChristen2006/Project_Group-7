"""Repository-relative paths shared by the Streamlit app and research scripts.

Data files are intentionally excluded from source control. Downloaded users
should place them under ``data/``; execution never depends on shell CWD.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def data_path(*parts: str) -> Path:
    """Return the absolute, repository-relative path within ``data/``.

    Args:
        *parts: Filesystem segments beneath the data directory.

    Returns:
        Resolved path within the local data directory.

    Raises:
        ValueError: If the requested segments attempt directory traversal.
    """
    path = DATA_DIR.joinpath(*parts).resolve()
    if not path.is_relative_to(DATA_DIR.resolve()):
        raise ValueError("Data paths must remain inside the data directory")
    return path


def output_dir(*parts: str) -> Path:
    """Create and return a directory for generated research artifacts.

    Args:
        *parts: Subdirectory segments within ``data/processed/``.

    Returns:
        Existing, absolute output directory.
    """
    target = data_path("processed", *parts)
    target.mkdir(parents=True, exist_ok=True)
    return target
