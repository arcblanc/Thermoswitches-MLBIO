"""Extract housekeeping 5' UTRs [200–600 nt] from RefSeq complete genomes.

Operon-aware: upstream window is clipped at the prior CDS / contig edge, so
polycistronic spacers often fall below 200 nt and are discarded by design.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path
from thermo_sim.thermo_common import gc_content, normalize_sequence

HOUSEKEEPING_PATTERNS = re.compile(
    r"(ribosomal|rpl[A-Z0-9]|rps[A-Z0-9]|rpo[ABCDEZ]|gapA|gyr[AB]|tuf|fusA|"
    r"inf[ABC]|ef[-_ ]?tu|atp[A-Z]|eno|pgk|mdh|recA|dna[KN])",
    re.IGNORECASE,
)
HEATSHOCK_PATTERNS = re.compile(
    r"(heat[- ]?shock|chaperon|groES|groEL|dnaK|dnaJ|ibpA|clpB|hs[lp]|thermoswitch|"
    r"thermometer|ROSE|cspA)",
    re.IGNORECASE,
)

MIN_UTR = 200
MAX_UTR = 600


def _parse_gff_attrs(attr_field: str) -> dict[str, str]:
    out = {}
    for part in attr_field.strip().split(";"):
        if not part or "=" not in part:
            continue
        key, val = part.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def _load_assembly_dirs(root: Path) -> list[tuple[str, Path, Path]]:
    """Return list of (assembly_accession, fasta_path, gff_path)."""
    assemblies = []
    for d in sorted(root.rglob("GCF_*")) + sorted(root.rglob("GCA_*")):
        if not d.is_dir():
            continue
        fasta = next(d.glob("*.fna"), None) or next(d.glob("*genomic.fna"), None)
        gff = next(d.glob("*.gff"), None) or next(d.glob("*.gff3"), None)
        if fasta is None:
            # NCBI datasets layout: nested ncbi_dataset/data/GC*_*/
            fasta = next(d.glob("**/*genomic.fna"), None) or next(d.glob("**/*.fna"), None)
        if gff is None:
            gff = next(d.glob("**/*.gff"), None) or next(d.glob("**/*.gff3"), None)
        if fasta and gff:
            assemblies.append((d.name, fasta, gff))
    # Deduplicate by accession
    seen = set()
    uniq = []
    for acc, fa, gff in assemblies:
        if acc in seen:
            continue
        seen.add(acc)
        uniq.append((acc, fa, gff))
    return uniq


def _parse_cds(gff_path: Path) -> dict[str, list[dict]]:
    by_seq: dict[str, list[dict]] = defaultdict(list)
    with gff_path.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS":
                continue
            seqid, _, _, start, end, _, strand, _, attrs = parts[:9]
            a = _parse_gff_attrs(attrs)
            product = a.get("product", a.get("Name", ""))
            gene = a.get("gene", a.get("locus_tag", ""))
            by_seq[seqid].append(
                {
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "product": product,
                    "gene": gene,
                    "locus_tag": a.get("locus_tag", gene),
                }
            )
    for seqid in by_seq:
        by_seq[seqid].sort(key=lambda x: (x["start"], x["end"]))
    return by_seq


def _is_housekeeping(cds: dict) -> bool:
    text = f"{cds.get('product', '')} {cds.get('gene', '')}"
    if HEATSHOCK_PATTERNS.search(text):
        return False
    return bool(HOUSEKEEPING_PATTERNS.search(text))


def _upstream_interval(cds: dict, neighbors: list[dict], contig_len: int) -> tuple[int, int] | None:
    """Return 0-based [start, end) genomic interval for UTR, or None if empty."""
    if cds["strand"] == "+":
        cds_start = cds["start"]  # 1-based inclusive
        prev_end = 0
        for other in neighbors:
            if other is cds:
                break
            if other["end"] < cds_start:
                prev_end = max(prev_end, other["end"])
        # 1-based inclusive UTR: (prev_end+1) .. (cds_start-1)
        left = prev_end + 1
        right = cds_start - 1
        if right < left:
            return None
        # convert to 0-based half-open
        return left - 1, right
    # minus strand: UTR is immediately "upstream" which is higher coords
    cds_end = cds["end"]
    next_start = contig_len + 1
    found = False
    for other in neighbors:
        if other is cds:
            found = True
            continue
        if found and other["start"] > cds_end:
            next_start = other["start"]
            break
    left = cds_end + 1
    right = next_start - 1
    if right < left:
        return None
    return left - 1, right


def extract_from_assembly(
    assembly_accession: str,
    fasta_path: Path,
    gff_path: Path,
    tax_string: str,
    min_utr: int = MIN_UTR,
    max_utr: int = MAX_UTR,
) -> tuple[list[dict], dict]:
    records = {rec.id.split()[0]: rec for rec in SeqIO.parse(str(fasta_path), "fasta")}
    # Also index without version quirks
    for rec in list(records.values()):
        records[rec.id] = rec

    cds_by_seq = _parse_cds(gff_path)
    stats = {
        "assembly": assembly_accession,
        "n_cds": 0,
        "n_housekeeping": 0,
        "n_too_short": 0,
        "n_too_long": 0,
        "n_kept": 0,
        "n_missing_seq": 0,
    }
    rows = []
    for seqid, cds_list in cds_by_seq.items():
        rec = records.get(seqid)
        if rec is None:
            # try prefix match
            rec = next((records[k] for k in records if k.startswith(seqid) or seqid.startswith(k)), None)
        if rec is None:
            stats["n_missing_seq"] += 1
            continue
        seq = str(rec.seq).upper()
        contig_len = len(seq)
        for cds in cds_list:
            stats["n_cds"] += 1
            if not _is_housekeeping(cds):
                continue
            stats["n_housekeeping"] += 1
            interval = _upstream_interval(cds, cds_list, contig_len)
            if interval is None:
                stats["n_too_short"] += 1
                continue
            a, b = interval
            utr = seq[a:b]
            if cds["strand"] == "-":
                utr = str(Seq(utr).reverse_complement())
            utr = normalize_sequence(utr)
            n = len(utr)
            if n < min_utr:
                stats["n_too_short"] += 1
                continue
            if n > max_utr:
                # RNA is always 5'→3' with CDS at the 3' end; keep CDS-proximal bases
                utr = utr[-max_utr:]
                n = len(utr)
            if not set(utr) <= set("AUGC"):
                continue
            stats["n_kept"] += 1
            # 1-based inclusive genomic coords of the retained (CDS-proximal) window
            if cds["strand"] == "+":
                seq_start, seq_end = b - n + 1, b
            else:
                seq_start, seq_end = a + 1, a + n
            rows.append(
                {
                    "rfam_acc": f"REFSEQ:{assembly_accession}",
                    "rfam_id": "refseq_housekeeping_utr",
                    "description": cds.get("product", ""),
                    "type": "refseq_utr",
                    "rfamseq_acc": f"{assembly_accession}:{seqid}",
                    "seq_start": int(seq_start),
                    "seq_end": int(seq_end),
                    "tax_string": tax_string,
                    "label": 0,
                    "sequence": utr,
                    "seq_length": n,
                    "gc_content": gc_content(utr),
                    "assembly_accession": assembly_accession,
                    "gene": cds.get("gene", ""),
                    "locus_tag": cds.get("locus_tag", ""),
                    "source": "refseq_utr",
                }
            )
    return rows, stats


def run_extract(
    genome_root: str = "data/raw/refseq_genomes",
    output_csv: str = "data/processed/refseq_utr/candidates.csv",
    output_fasta: str = "data/processed/refseq_utr/candidates.fasta",
    report_json: str = "data/processed/refseq_utr/extract_report.json",
    min_utr: int = MIN_UTR,
    max_utr: int = MAX_UTR,
    target_candidates: int = 3000,
) -> dict:
    root = resolve_path(genome_root)
    assemblies = _load_assembly_dirs(root)
    all_rows = []
    per_assembly = []
    for acc, fasta, gff in assemblies:
        taxon = "Bacteria"
        for part in Path(acc).parts:
            pass
        # infer taxon folder from path
        tax_string = "Bacteria"
        for parent in fasta.parents:
            if parent.name in {"Pseudomonadota", "Bacillota"}:
                tax_string = f"Bacteria; {parent.name}"
                break
        rows, stats = extract_from_assembly(acc, fasta, gff, tax_string, min_utr, max_utr)
        all_rows.extend(rows)
        per_assembly.append(stats)
        print(f"{acc}: kept={stats['n_kept']} hk={stats['n_housekeeping']} cds={stats['n_cds']}")
        if len(all_rows) >= target_candidates * 3:
            # enough headroom after cmscan losses
            break

    out_csv = resolve_path(output_csv)
    out_fa = resolve_path(output_fasta)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.DataFrame(all_rows)
    if len(df):
        df = df.drop_duplicates(subset=["rfamseq_acc", "seq_start", "seq_end"])
    df.to_csv(out_csv, index=False)
    with out_fa.open("w") as handle:
        for _, row in df.iterrows():
            header = (
                f">{row['rfam_acc']}|{row['rfam_id']}|{row['rfamseq_acc']}|"
                f"{row['seq_start']}-{row['seq_end']}|label=0"
            )
            handle.write(header + "\n")
            handle.write(row["sequence"] + "\n")

    report = {
        "n_assemblies": len(per_assembly),
        "n_candidates": int(len(df)),
        "target_candidates": target_candidates,
        "min_utr": min_utr,
        "max_utr": max_utr,
        "n_too_short_total": int(sum(s["n_too_short"] for s in per_assembly)),
        "n_housekeeping_total": int(sum(s["n_housekeeping"] for s in per_assembly)),
        "per_assembly": per_assembly,
        "output_csv": str(out_csv),
        "output_fasta": str(out_fa),
        "ready_for_cmscan": bool(len(df) >= target_candidates),
    }
    report_path = resolve_path(report_json)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in report if k != "per_assembly"}, indent=2))
    if not report["ready_for_cmscan"]:
        print(
            f"WARNING: only {len(df)} candidates (< {target_candidates}). "
            "Download more assemblies before rematch.",
            file=sys.stderr,
        )
    return report


def main():
    p = argparse.ArgumentParser(description="Extract RefSeq housekeeping 5' UTRs")
    p.add_argument("--genome-root", default="data/raw/refseq_genomes")
    p.add_argument("--output-csv", default="data/processed/refseq_utr/candidates.csv")
    p.add_argument("--output-fasta", default="data/processed/refseq_utr/candidates.fasta")
    p.add_argument("--min-utr", type=int, default=MIN_UTR)
    p.add_argument("--max-utr", type=int, default=MAX_UTR)
    p.add_argument("--target-candidates", type=int, default=3000)
    args = p.parse_args()
    run_extract(
        genome_root=args.genome_root,
        output_csv=args.output_csv,
        output_fasta=args.output_fasta,
        min_utr=args.min_utr,
        max_utr=args.max_utr,
        target_candidates=args.target_candidates,
    )


if __name__ == "__main__":
    main()
