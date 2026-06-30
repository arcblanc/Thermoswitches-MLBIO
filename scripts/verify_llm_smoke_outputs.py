import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_engineering.paths import resolve_path

FASTA_PATH = "data/processed/de_novo/smoke/generated.fasta"
EMBED_DIR = "data/processed/validation_embedding/smoke"
RNA_PATTERN = re.compile(r"^[AUGC]+$")


def parse_fasta(path):
    records = []
    header = None
    parts = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def verify(expected_records=2, hidden_size=768):
    fasta_path = resolve_path(FASTA_PATH)
    embed_dir = resolve_path(EMBED_DIR)
    manifest_path = embed_dir / "manifest.jsonl"

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    records = parse_fasta(fasta_path) if fasta_path.exists() else []
    check("fasta record count", len(records) == expected_records, f"{len(records)} records")

    for header, sequence in records:
        seq = sequence.replace("T", "U").upper()
        ok = bool(RNA_PATTERN.match(seq))
        check(f"{header} valid RNA", ok, seq[:40] + ("..." if len(seq) > 40 else ""))

    npy_files = sorted(embed_dir.glob("*.npy"))
    check("npy file count", len(npy_files) >= expected_records, f"{len(npy_files)} files")
    for npy_path in npy_files[:expected_records]:
        array = np.load(npy_path)
        check(
            f"{npy_path.name} hidden dim",
            array.ndim == 3 and array.shape[-1] == hidden_size,
            str(array.shape),
        )

    manifest_lines = 0
    if manifest_path.exists():
        manifest_lines = sum(1 for _ in manifest_path.open())
    check("manifest.jsonl lines", manifest_lines == expected_records, f"{manifest_lines} lines")

    failed = [name for name, ok, _ in checks if not ok]
    if failed:
        print(f"\nVerification failed: {', '.join(failed)}")
        sys.exit(1)
    print("\nAll LLM smoke output checks passed.")


if __name__ == "__main__":
    expected = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    verify(expected_records=expected)
