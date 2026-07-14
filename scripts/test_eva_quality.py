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
    gate_chunk,
)


def test_valid_diverse_chunk():
    diverse = [
        "AUGCCGAUUAGCUGACCUAGGAUCCGAUGCAUUGCCAAGGAUUCGAUCCGAUGCA",
        "GCAAUUCGGAUCCUAUGCCGAUUACGGAUCCGAUGCUUAAUCCCGGAUUCGAUGC",
    ]
    result = evaluate_chunk(diverse)
    assert result.ok, result.errors


def test_invalid_biological_formatting():
    result = evaluate_chunk(["AUGCNNN"])
    assert any("Invalid Biological Formatting" in e for e in result.errors)


def test_length_violations():
    result = evaluate_chunk(["AUGC"], cfg=QualityGateConfig(min_len=40, max_len=600))
    assert any("Length Violations" in e for e in result.errors)


def test_repetitive_collapse():
    result = evaluate_chunk(["A" * 100], cfg=QualityGateConfig(min_len=40, max_len=600))
    assert any("Repetitive Text Collapse" in e for e in result.errors)


def test_gate_raises():
    try:
        gate_chunk(["A" * 100], cfg=QualityGateConfig(min_len=40, max_len=600))
        raise AssertionError("expected QualityGateError")
    except QualityGateError:
        pass


def test_panel_full_and_smoke():
    assert rna_type() == "mRNA"
    assert panel_total(load_panel_hosts(smoke=False)) == 10000
    assert [h.taxid for h in load_panel_hosts(smoke=False)] == [562, 28901, 1639]
    assert sum(h.n_seqs for h in load_panel_hosts(smoke=True)) == 16


def test_reject_srna(monkeypatch=None):
    os.environ["EVA_RNA_TYPE"] = "sRNA"
    try:
        try:
            rna_type()
            raise AssertionError("expected ValueError for sRNA")
        except ValueError:
            pass
    finally:
        os.environ["EVA_RNA_TYPE"] = "mRNA"


if __name__ == "__main__":
    test_valid_diverse_chunk()
    test_invalid_biological_formatting()
    test_length_violations()
    test_repetitive_collapse()
    test_gate_raises()
    test_panel_full_and_smoke()
    test_reject_srna()
    print("all eva unit checks passed")
