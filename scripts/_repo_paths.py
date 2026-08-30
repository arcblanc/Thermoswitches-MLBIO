"""Repo root resolution for nested ``scripts/<category>/`` entry points."""

from __future__ import annotations

from pathlib import Path

# scripts/_repo_paths.py -> parents[1] = repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def repo_root_from_script(script_file: str | Path) -> Path:
    """Return repo root when called from ``scripts/<category>/script.py``."""
    return Path(script_file).resolve().parents[2]
