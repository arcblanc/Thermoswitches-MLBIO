"""Decontaminate RefSeq UTR candidates with Infernal cmscan (--cut_ga).

Discards sequences with significant hits to thermoswitch / riboswitch Rfam CMs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path

TOOLS_BIN = Path(__file__).resolve().parents[2] / ".tools" / "bin"


def _resolve_tool(name: str) -> str:
    """Return a local tools-bin path for name, else the bare command."""
    local = TOOLS_BIN / name
    if local.exists():
        return str(local)
    return name


def download_rfam_cms(
    positives_csv: str = "data/processed/cdhitoutput/positives_deduped.csv",
    cm_dir: str = "data/raw/rfam_cms",
) -> Path:
    """Fetch Rfam.cm.gz (full) once; subset later via cmfetch if needed."""
    cm_dir_p = resolve_path(cm_dir)
    cm_dir_p.mkdir(parents=True, exist_ok=True)
    cm_path = cm_dir_p / "Rfam.cm"
    if cm_path.exists() and cm_path.stat().st_size > 1_000_000:
        return cm_path
    url = "https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz"
    gz = cm_dir_p / "Rfam.cm.gz"
    print(f"Downloading {url}")
    subprocess.run(["curl", "-fsSL", "-o", str(gz), url], check=True)
    subprocess.run(["gunzip", "-f", str(gz)], check=True)
    if not cm_path.exists():
        raise FileNotFoundError(cm_path)
    return cm_path


def build_target_cm(
    rfam_cm: Path,
    positives_csv: str,
    out_cm: Path,
) -> Path:
    """Extract thermoswitch + riboswitch-related CMs into a smaller file."""
    pos = pd.read_csv(resolve_path(positives_csv))
    families = sorted(set(pos["rfam_acc"].astype(str).tolist()))
    # Broader clans / known IDs to scrub (cmstat names vary; fetch by accession when possible)
    extra = [
        "RF00080",  # yybP-ykoY
        "RF00162",  # SAM
        "RF00059",  # TPP
        "RF00174",  # Cobalamin
        "RF00050",  # FMN
        "RF00504",  # glycine
        "RF00167",  # Purine
        "RF01055",  # ROK
        "RF01727",  # SAM-I/IV
    ]
    wanted = sorted(set(families + extra))
    cmfetch = _resolve_tool("cmfetch")
    # Build concatenated CM file
    out_cm.parent.mkdir(parents=True, exist_ok=True)
    with out_cm.open("w") as out:
        for acc in wanted:
            try:
                result = subprocess.run(
                    [cmfetch, str(rfam_cm), acc],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                out.write(result.stdout)
            except subprocess.CalledProcessError:
                print(f"  warn: cmfetch missed {acc}")
    if out_cm.stat().st_size < 100:
        # fallback: use full Rfam.cm
        print("WARN: subset CM empty/small; using full Rfam.cm")
        return rfam_cm
    return out_cm


def press_cm(cm_path: Path) -> Path:
    """Press a covariance model with cmpress if not already pressed."""
    cmpress = _resolve_tool("cmpress")
    # cmpress refuses if already pressed
    for suffix in (".i1m", ".i1i", ".i1f", ".i1p"):
        if (Path(str(cm_path) + suffix)).exists():
            return cm_path
    subprocess.run([cmpress, str(cm_path)], check=True)
    return cm_path


def run_cmscan(
    cm_path: Path,
    query_fasta: Path,
    tblout: Path,
    cpu: int = 4,
) -> Path:
    """Run Infernal cmscan against a query FASTA and write tblout."""
    cmscan = _resolve_tool("cmscan")
    tblout.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        cmscan,
        "--cut_ga",
        "--rfam",
        "--nohmmonly",
        "--cpu",
        str(cpu),
        "--tblout",
        str(tblout),
        str(cm_path),
        str(query_fasta),
    ]
    print(" $", " ".join(cmd))
    # stdout can be huge; discard
    with open(os.devnull, "w") as devnull:
        subprocess.run(cmd, check=True, stdout=devnull)
    return tblout


def parse_tblout_hits(tblout: Path, evalue_max: float = 1e-3) -> set[str]:
    """Return query names with significant hits (GA pass already in file; also E filter)."""
    hits = set()
    if not tblout.exists():
        return hits
    with tblout.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 16:
                continue
            # Infernal tblout: target name, accession, query name, ... E-value at field 15 (0-index 14) for cmscan
            # Format: (1) target name (2) accession (3) query name (4) accession (5) mdl (6) mdl from ...
            # E-value is column 16 (index 15) in standard --tblout
            try:
                query = parts[2]
                evalue = float(parts[15])
            except (IndexError, ValueError):
                continue
            if evalue < evalue_max:
                hits.add(query)
            else:
                # still count GA-passing rows present in tblout when --cut_ga used
                hits.add(query)
    return hits


def fasta_headers(path: Path) -> list[str]:
    """Return FASTA header strings without the leading '>'."""
    headers = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers


def decontaminate(
    candidates_csv: str = "data/processed/refseq_utr/candidates.csv",
    candidates_fasta: str = "data/processed/refseq_utr/candidates.fasta",
    output_csv: str = "data/processed/refseq_utr/candidates_clean.csv",
    output_fasta: str = "data/processed/refseq_utr/candidates_clean.fasta",
    report_json: str = "data/processed/refseq_utr/cmscan_report.json",
    tblout: str = "data/processed/refseq_utr/cmscan_hits.tbl",
    positives_csv: str = "data/processed/cdhitoutput/positives_deduped.csv",
    evalue_max: float = 1e-3,
    cpu: int = 4,
) -> dict:
    """Drop candidate UTRs with significant Rfam cmscan hits."""
    rfam_cm = download_rfam_cms(positives_csv=positives_csv)
    subset = resolve_path("data/raw/rfam_cms/thermoswitch_riboswitch.cm")
    cm_path = build_target_cm(rfam_cm, positives_csv, subset)
    press_cm(cm_path)

    # Rewrite FASTA headers to simple unique IDs for cmscan parsing
    df = pd.read_csv(resolve_path(candidates_csv))
    for col in ("seq_start", "seq_end"):
        df[col] = df[col].astype(int)
    id_map = {}
    scan_fa = resolve_path("data/processed/refseq_utr/candidates_scan.fasta")
    with scan_fa.open("w") as handle:
        for i, row in df.iterrows():
            qid = f"cand_{i}"
            id_map[qid] = i
            handle.write(f">{qid}\n{row['sequence']}\n")

    tbl = resolve_path(tblout)
    run_cmscan(cm_path, scan_fa, tbl, cpu=cpu)
    hit_qids = parse_tblout_hits(tbl, evalue_max=evalue_max)
    hit_idx = {id_map[q] for q in hit_qids if q in id_map}

    clean = df.loc[~df.index.isin(hit_idx)].copy()
    out_csv = resolve_path(output_csv)
    out_fa = resolve_path(output_fasta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out_csv, index=False)
    with out_fa.open("w") as handle:
        for _, row in clean.iterrows():
            header = (
                f"{row.get('rfam_acc', 'REFSEQ')}|{row.get('rfam_id', 'refseq_utr')}|"
                f"{row['rfamseq_acc']}|{int(row['seq_start'])}-{int(row['seq_end'])}|label=0"
            )
            handle.write(">" + header.lstrip(">") + "\n")
            handle.write(str(row["sequence"]) + "\n")

    report = {
        "n_input": int(len(df)),
        "n_hits_removed": int(len(hit_idx)),
        "n_clean": int(len(clean)),
        "evalue_max": evalue_max,
        "cm_path": str(cm_path),
        "tblout": str(tbl),
        "output_csv": str(resolve_path(output_csv)),
        "output_fasta": str(resolve_path(output_fasta)),
    }
    resolve_path(report_json).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Parse CLI arguments and run cmscan decontamination."""
    p = argparse.ArgumentParser()
    p.add_argument("--cpu", type=int, default=4)
    p.add_argument("--evalue-max", type=float, default=1e-3)
    args = p.parse_args()
    decontaminate(cpu=args.cpu, evalue_max=args.evalue_max)


if __name__ == "__main__":
    main()
