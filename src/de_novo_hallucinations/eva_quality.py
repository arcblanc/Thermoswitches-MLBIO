"""Quality gates for EVA-generated RNA.

Supports:
- Hard batch evaluation (`evaluate_chunk` / `gate_chunk`) for diagnostics / tests
- Per-sequence soft-drop (`evaluate_single_sequence` / `soft_filter_sequences`)
  so one bad sequence does not discard an entire GPU chunk.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

RNA_PATTERN = re.compile(r"^[AUGC]+$")


class QualityGateError(Exception):
    """Raised when a chunk fails one or more quality gates (hard path)."""

    def __init__(self, errors: list[str]) -> None:
        """Store gate error messages and set the exception text."""
        self.errors = errors
        joined = "; ".join(errors)
        super().__init__(joined)


@dataclass
class QualityGateConfig:
    min_len: int = 40
    max_len: int = 600
    mono_fraction_max: float = 0.85
    dimer_fraction_max: float = 0.85
    min_unique_3mer_ratio: float = 0.05
    max_identical_copies: int = 3
    max_near_duplicate_frac: float = 0.20
    near_duplicate_identity: float = 0.98


@dataclass
class GateResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self) -> None:
        """Raise QualityGateError when the gate result is not ok."""
        if not self.ok:
            raise QualityGateError(self.errors)


@dataclass
class SoftFilterResult:
    """Outcome of soft-dropping bad sequences from a chunk."""

    passed: list[str]
    dropped: list[tuple[int, str, str]]  # (index, sequence_or_empty, reason)
    drop_reason_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_passed(self) -> int:
        """Return the number of sequences that passed filters."""
        return len(self.passed)

    @property
    def n_dropped(self) -> int:
        """Return the number of sequences that were dropped."""
        return len(self.dropped)


def normalize_rna(sequence: str) -> str:
    """Normalize RNA by stripping, uppercasing, and converting T to U."""
    return sequence.replace("T", "U").upper().strip()


def check_invalid_biological_formatting(sequences: list[str]) -> list[str]:
    """Return errors for empty, non-AUGC, or whitespace-containing sequences."""
    errors = []
    for index, raw in enumerate(sequences):
        seq = normalize_rna(raw)
        label = f"seq[{index}]"
        if not seq:
            errors.append(f"Invalid Biological Formatting: {label} is empty")
            continue
        if not RNA_PATTERN.match(seq):
            invalid = sorted(set(seq) - set("AUGC"))
            errors.append(
                f"Invalid Biological Formatting: {label} has non-AUGC chars {invalid}"
            )
        if any(ch.isspace() for ch in raw.strip()):
            # Internal whitespace after strip of ends only matters if remaining
            if re.search(r"\s", raw.strip()):
                errors.append(
                    f"Invalid Biological Formatting: {label} contains whitespace"
                )
    return errors


def check_length_violations(
    sequences: list[str], *, min_len: int, max_len: int
) -> list[str]:
    """Return errors for sequences outside the allowed length bounds."""
    errors = []
    for index, raw in enumerate(sequences):
        seq = normalize_rna(raw)
        n = len(seq)
        if n < min_len or n > max_len:
            errors.append(
                f"Length Violations: seq[{index}] length={n} outside [{min_len}, {max_len}]"
            )
    return errors


def _mono_fraction(seq: str) -> float:
    """Return the fraction of the most common nucleotide."""
    if not seq:
        return 1.0
    counts = Counter(seq)
    return max(counts.values()) / len(seq)


def _dimer_fraction(seq: str) -> float:
    """Return the fraction of the most common overlapping dimer."""
    if len(seq) < 2:
        return 0.0
    dimers = [seq[i : i + 2] for i in range(len(seq) - 1)]
    counts = Counter(dimers)
    return max(counts.values()) / len(dimers)


def _unique_3mer_ratio(seq: str) -> float:
    """Return unique 3-mer count divided by the number of 3-mer windows."""
    if len(seq) < 3:
        return 0.0
    kmers = {seq[i : i + 3] for i in range(len(seq) - 2)}
    return len(kmers) / (len(seq) - 2)


def _pairwise_identity(a: str, b: str) -> float:
    """Approximate identity as prefix matches over the longer length."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Fast approx: identity over min length overlapping prefix (chunk-level gate).
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / max(len(a), len(b))


def check_repetitive_text_collapse(
    sequences: list[str], cfg: QualityGateConfig
) -> list[str]:
    """Return errors for low-complexity, identical, or near-duplicate sequences."""
    errors = []
    norms = [normalize_rna(s) for s in sequences]

    for index, seq in enumerate(norms):
        if not seq:
            continue
        mono = _mono_fraction(seq)
        if mono >= cfg.mono_fraction_max:
            errors.append(
                f"Repetitive Text Collapse: seq[{index}] mono_fraction={mono:.3f} "
                f">= {cfg.mono_fraction_max}"
            )
        dimer = _dimer_fraction(seq)
        if dimer >= cfg.dimer_fraction_max:
            errors.append(
                f"Repetitive Text Collapse: seq[{index}] dimer_fraction={dimer:.3f} "
                f">= {cfg.dimer_fraction_max}"
            )
        ratio = _unique_3mer_ratio(seq)
        if len(seq) >= 80 and ratio < cfg.min_unique_3mer_ratio:
            errors.append(
                f"Repetitive Text Collapse: seq[{index}] unique_3mer_ratio={ratio:.3f} "
                f"< {cfg.min_unique_3mer_ratio}"
            )

    counts = Counter(norms)
    for seq, count in counts.items():
        if seq and count >= cfg.max_identical_copies:
            errors.append(
                f"Repetitive Text Collapse: {count} identical copies of a sequence "
                f"(max allowed {cfg.max_identical_copies})"
            )

    if len(norms) >= 2:
        near_dup_pairs = 0
        compared = 0
        # Cap pairwise work for large chunks
        limit = min(len(norms), 128)
        for i in range(limit):
            for j in range(i + 1, limit):
                compared += 1
                if (
                    _pairwise_identity(norms[i], norms[j])
                    >= cfg.near_duplicate_identity
                ):
                    near_dup_pairs += 1
        if compared > 0:
            frac = near_dup_pairs / compared
            if frac > cfg.max_near_duplicate_frac:
                errors.append(
                    f"Repetitive Text Collapse: near-duplicate fraction={frac:.3f} "
                    f"> {cfg.max_near_duplicate_frac}"
                )

    return errors


def evaluate_chunk(
    sequences: list[str],
    cfg: QualityGateConfig | None = None,
) -> GateResult:
    """Evaluate a chunk against all three hard gates."""
    cfg = cfg or QualityGateConfig()
    errors: list[str] = []
    errors.extend(check_invalid_biological_formatting(sequences))
    # Length / collapse only on biologically formatted seqs when possible,
    # but still run length on normalized forms.
    errors.extend(
        check_length_violations(sequences, min_len=cfg.min_len, max_len=cfg.max_len)
    )
    errors.extend(check_repetitive_text_collapse(sequences, cfg))
    return GateResult(ok=not errors, errors=errors)


def gate_chunk(
    sequences: list[str],
    cfg: QualityGateConfig | None = None,
) -> GateResult:
    """Evaluate and raise QualityGateError on failure."""
    result = evaluate_chunk(sequences, cfg=cfg)
    result.raise_if_failed()
    return result


def evaluate_single_sequence(
    sequence: str,
    cfg: QualityGateConfig | None = None,
    *,
    index: int = 0,
) -> tuple[bool, str | None]:
    """Per-sequence gates (formatting, length, mono/dimer/3-mer).

    Returns (ok, reason_or_none). Reason prefixes match batch gate categories.
    """
    cfg = cfg or QualityGateConfig()
    label = f"seq[{index}]"
    raw = sequence
    seq = normalize_rna(raw)
    if not seq:
        return False, f"Invalid Biological Formatting: {label} is empty"
    if not RNA_PATTERN.match(seq):
        invalid = sorted(set(seq) - set("AUGC"))
        return False, (
            f"Invalid Biological Formatting: {label} has non-AUGC chars {invalid}"
        )
    if re.search(r"\s", raw.strip()):
        return False, f"Invalid Biological Formatting: {label} contains whitespace"
    n = len(seq)
    if n < cfg.min_len or n > cfg.max_len:
        return False, (
            f"Length Violations: {label} length={n} outside [{cfg.min_len}, {cfg.max_len}]"
        )
    mono = _mono_fraction(seq)
    if mono >= cfg.mono_fraction_max:
        return False, (
            f"Repetitive Text Collapse: {label} mono_fraction={mono:.3f} "
            f">= {cfg.mono_fraction_max}"
        )
    dimer = _dimer_fraction(seq)
    if dimer >= cfg.dimer_fraction_max:
        return False, (
            f"Repetitive Text Collapse: {label} dimer_fraction={dimer:.3f} "
            f">= {cfg.dimer_fraction_max}"
        )
    ratio = _unique_3mer_ratio(seq)
    if n >= 80 and ratio < cfg.min_unique_3mer_ratio:
        return False, (
            f"Repetitive Text Collapse: {label} unique_3mer_ratio={ratio:.3f} "
            f"< {cfg.min_unique_3mer_ratio}"
        )
    return True, None


def soft_filter_sequences(
    sequences: list[str],
    cfg: QualityGateConfig | None = None,
) -> SoftFilterResult:
    """Keep sequences that pass per-seq gates; drop near/identical dups within chunk.

    Does not raise — callers append `passed` and top-up generation for the deficit.
    """
    cfg = cfg or QualityGateConfig()
    passed: list[str] = []
    dropped: list[tuple[int, str, str]] = []
    reason_counts: dict[str, int] = {}
    identical_counts: Counter[str] = Counter()

    def _drop(index: int, seq: str, reason: str) -> None:
        """Record a dropped sequence and increment its reason count."""
        dropped.append((index, seq, reason))
        cat = reason.split(":", 1)[0].strip() or "unknown"
        reason_counts[cat] = reason_counts.get(cat, 0) + 1

    for index, raw in enumerate(sequences):
        ok, reason = evaluate_single_sequence(raw, cfg, index=index)
        if not ok:
            _drop(index, normalize_rna(raw), reason or "unknown")
            continue
        seq = normalize_rna(raw)
        identical_counts[seq] += 1
        if identical_counts[seq] >= cfg.max_identical_copies:
            _drop(
                index,
                seq,
                f"Repetitive Text Collapse: seq[{index}] identical copy "
                f"#{identical_counts[seq]} (max {cfg.max_identical_copies - 1} kept)",
            )
            continue
        # Exact duplicates are governed by max_identical_copies above.
        # Near-dup only applies to non-identical high-identity pairs.
        near_dup = False
        for kept in passed:
            if seq == kept:
                continue
            if _pairwise_identity(seq, kept) >= cfg.near_duplicate_identity:
                _drop(
                    index,
                    seq,
                    f"Repetitive Text Collapse: seq[{index}] near-duplicate of kept seq "
                    f"(identity>={cfg.near_duplicate_identity})",
                )
                near_dup = True
                break
        if near_dup:
            continue
        passed.append(seq)

    return SoftFilterResult(
        passed=passed, dropped=dropped, drop_reason_counts=reason_counts
    )
