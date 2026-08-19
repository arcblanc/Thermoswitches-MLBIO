"""Lightweight unit tests for EVA quality gates and panel prompts (no GPU / EVA binary)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from de_novo_hallucinations.eva_prompts import load_panel_hosts, panel_total, rna_type
from de_novo_hallucinations.eva_quality import (
    QualityGateConfig,
    QualityGateError,
    evaluate_chunk,
    evaluate_single_sequence,
    gate_chunk,
    soft_filter_sequences,
)


def test_valid_diverse_chunk() -> None:
    """Assert a diverse valid chunk passes quality gates."""
    diverse = [
        "AUGCCGAUUAGCUGACCUAGGAUCCGAUGCAUUGCCAAGGAUUCGAUCCGAUGCA",
        "GCAAUUCGGAUCCUAUGCCGAUUACGGAUCCGAUGCUUAAUCCCGGAUUCGAUGC",
    ]
    result = evaluate_chunk(diverse)
    assert result.ok, result.errors


def test_invalid_biological_formatting() -> None:
    """Assert N-containing sequence fails biological formatting."""
    result = evaluate_chunk(["AUGCNNN"])
    assert any("Invalid Biological Formatting" in e for e in result.errors)


def test_length_violations() -> None:
    """Assert a too-short sequence fails the length gate."""
    result = evaluate_chunk(["AUGC"], cfg=QualityGateConfig(min_len=40, max_len=600))
    assert any("Length Violations" in e for e in result.errors)


def test_repetitive_collapse() -> None:
    """Assert a homopolymer sequence fails the repetitive-collapse gate."""
    result = evaluate_chunk(["A" * 100], cfg=QualityGateConfig(min_len=40, max_len=600))
    assert any("Repetitive Text Collapse" in e for e in result.errors)


def test_gate_raises() -> None:
    """Assert gate_chunk raises QualityGateError on a failing chunk."""
    try:
        gate_chunk(["A" * 100], cfg=QualityGateConfig(min_len=40, max_len=600))
        raise AssertionError("expected QualityGateError")
    except QualityGateError:
        pass


def test_panel_full_and_smoke() -> None:
    """Assert full and smoke panel taxids, sizes, and default RNA type."""
    assert rna_type() == "mRNA"
    assert panel_total(load_panel_hosts(smoke=False)) == 10000
    assert [h.taxid for h in load_panel_hosts(smoke=False)] == [562, 28901, 1639]
    assert sum(h.n_seqs for h in load_panel_hosts(smoke=True)) == 16


def test_reject_srna(monkeypatch: object | None = None) -> None:
    """Assert EVA_RNA_TYPE=sRNA is rejected as an invalid RNA type."""
    os.environ["EVA_RNA_TYPE"] = "sRNA"
    try:
        try:
            rna_type()
            raise AssertionError("expected ValueError for sRNA")
        except ValueError:
            pass
    finally:
        os.environ["EVA_RNA_TYPE"] = "mRNA"


def _diverse_seq(seed: int = 0) -> str:
    """Deterministic diverse RNA >= 80 nt for soft-filter tests."""
    import random

    rng = random.Random(10_000 + seed)
    return "".join(rng.choice("AUGC") for _ in range(120))


def test_soft_filter_keeps_majority() -> None:
    """Hard batch gate would abort; soft-drop keeps passers (Step 2 failure mode)."""
    cfg = QualityGateConfig(min_len=40, max_len=600)
    good = [_diverse_seq(i) for i in range(10)]
    # Low unique-3mer collapse without mono/dimer trip (same failure class as pilot).
    collapsed = "AUA" * 40  # ~120 nt, unique_3mer_ratio ≈ 2/118 ≪ 0.05
    ok, reason = evaluate_single_sequence(collapsed, cfg)
    assert not ok and reason and "unique_3mer_ratio" in reason
    hard = evaluate_chunk(good + [collapsed], cfg=cfg)
    assert not hard.ok
    soft = soft_filter_sequences(good + [collapsed], cfg=cfg)
    assert soft.n_passed == 10
    assert soft.n_dropped == 1
    assert soft.drop_reason_counts.get("Repetitive Text Collapse") == 1


def test_soft_filter_drops_identical_copies() -> None:
    """Assert identical copies beyond max_identical_copies are dropped."""
    cfg = QualityGateConfig(min_len=40, max_len=600, max_identical_copies=3)
    seq = _diverse_seq(7)
    soft = soft_filter_sequences([seq, seq, seq, seq], cfg=cfg)
    # Keeps max_identical_copies - 1 (=2); drops the 3rd+ copies.
    assert soft.n_passed == 2
    assert soft.n_dropped == 2


if __name__ == "__main__":
    test_valid_diverse_chunk()
    test_invalid_biological_formatting()
    test_length_violations()
    test_repetitive_collapse()
    test_gate_raises()
    test_panel_full_and_smoke()
    test_reject_srna()
    test_soft_filter_keeps_majority()
    test_soft_filter_drops_identical_copies()
    print("all eva unit checks passed")
