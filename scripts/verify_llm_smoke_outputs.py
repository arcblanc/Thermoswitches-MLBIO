import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from data_engineering.paths import resolve_path

RNA_PATTERN = re.compile(r"^[AUGC]+$")
DEFAULT_FASTA = "data/processed/de_novo/smoke/generated.fasta"
DEFAULT_EMBED_DIR = "data/processed/validation_embedding/smoke"


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


def verify(
    expected_records=2,
    hidden_size=768,
    fasta_path=DEFAULT_FASTA,
    embed_dir=DEFAULT_EMBED_DIR,
):
    fasta_path = resolve_path(fasta_path)
    embed_dir = resolve_path(embed_dir)
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
        return False
    print("\nAll LLM smoke output checks passed.")
    return True


def _build_parser():
    parser = argparse.ArgumentParser(description="Verify GenerRNA + BiRNA-BERT outputs.")
    parser.add_argument("--expected", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--embed-dir", default=DEFAULT_EMBED_DIR)
    return parser


def main():
    args = _build_parser().parse_args()
    ok = verify(
        expected_records=args.expected,
        hidden_size=args.hidden_size,
        fasta_path=args.fasta,
        embed_dir=args.embed_dir,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
