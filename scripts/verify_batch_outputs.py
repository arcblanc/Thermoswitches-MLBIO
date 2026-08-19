import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from data_engineering.paths import resolve_path

FUSED_CSV = "data/processed/fused_features.csv"
VIENNA_CSV = "data/processed/viennarna/features.csv"
NUPACK_CSV = "data/processed/nupack/features.csv"


def _line_count(path: str | Path) -> int:
    """Return the number of lines in path, or 0 if it does not exist."""
    path = resolve_path(path)
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for _ in handle)


def _read_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV into a DataFrame, or empty if missing or zero-sized."""
    path = resolve_path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def verify(expected_rows: int = 10) -> bool:
    """Check fused/Vienna/NUPACK CSVs have expected rows and unique join keys."""
    fused = _read_csv(FUSED_CSV)
    vienna = _read_csv(VIENNA_CSV)
    nupack = _read_csv(NUPACK_CSV)

    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        """Record and print one verification check result."""
        checks.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    fused_lines = _line_count(FUSED_CSV)
    vienna_lines = _line_count(VIENNA_CSV)
    nupack_lines = _line_count(NUPACK_CSV)

    check(
        "fused row count",
        len(fused) == expected_rows,
        f"{len(fused)} rows (expected {expected_rows})",
    )
    check("vienna row count", len(vienna) == expected_rows, f"{len(vienna)} rows")
    check("nupack row count", len(nupack) == expected_rows, f"{len(nupack)} rows")
    check(
        "fused line count",
        fused_lines == expected_rows + 1,
        f"{fused_lines} lines (header + rows)",
    )
    check(
        "vienna line count", vienna_lines == expected_rows + 1, f"{vienna_lines} lines"
    )
    check(
        "nupack line count", nupack_lines == expected_rows + 1, f"{nupack_lines} lines"
    )

    if not fused.empty:
        dupes = fused.duplicated(subset=JOIN_COLUMNS, keep=False).sum()
        check("no duplicate join keys", dupes == 0, f"{dupes} duplicates")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"\nVerification failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll batch output checks passed.")
    return True


def verify_append_after_rerun(initial_rows: int = 6, total_rows: int = 10) -> None:
    """After a partial run (6 rows) and resumed run (10 total), fused CSV should have 10 rows."""
    fused = _read_csv(FUSED_CSV)
    ok = len(fused) == total_rows
    print(
        f"[{'PASS' if ok else 'FAIL'}] append/resume total rows: {len(fused)} (expected {total_rows})"
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    expected = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    verify(expected_rows=expected)
