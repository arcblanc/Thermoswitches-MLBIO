from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_path(path):
    """Resolve a repo-relative path to an absolute path under PROJECT_ROOT."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path
