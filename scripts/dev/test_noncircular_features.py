"""Backward-compatible runner for noncircular feature tests.

Prefer: ``make test`` or ``uv run pytest tests/test_noncircular_features.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS = Path(__file__).resolve().parents[2] / "tests" / "test_noncircular_features.py"

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(_TESTS), *sys.argv[1:]]))
