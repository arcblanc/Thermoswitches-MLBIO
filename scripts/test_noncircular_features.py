"""Unit tests for non-circular RF features and post-hoc gates (no Vienna/NUPACK)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from thermo_sim.noncircular_features import (
    DINUC_COLUMNS,
    SD_AUG_SENTINEL,
    TRINUC_COLUMNS,
    add_composition_features,
    kmer_frequencies,
    p_paired_rbs_37,
    sd_aug_features,
)
from thermo_sim.rf_posthoc import (
    SPEARMAN_MIN_N,
    confidence_bin,
    gate_delta_p_rbs,
    gate_hill,
    gate_tm,
    gate_zscore,
    spearman_consensus,
    visual_checklist_flags,
)


def test_kmer_keys_and_sum() -> None:
    """Assert dinuc/trinuc keys, sums, and AU count for a short sequence."""
    seq = "AUGCAUGC"
    dinuc = kmer_frequencies(seq, 2)
    trinuc = kmer_frequencies(seq, 3)
    assert list(dinuc) == DINUC_COLUMNS
    assert list(trinuc) == TRINUC_COLUMNS
    assert len(dinuc) == 16
    assert len(trinuc) == 64
    assert abs(sum(dinuc.values()) - 1.0) < 1e-9
    assert abs(sum(trinuc.values()) - 1.0) < 1e-9
    # AU appears twice in AUGCAUGC dinucs (positions 0 and 4)
    assert dinuc["dinuc_AU"] == 2 / 7


def test_sd_aug_spacing_known() -> None:
    """Assert SD–AUG spacing is 6 for a known AGGAGG spacer construct."""
    # AGGAGG + 6 nt spacer + AUG
    seq = "AGGAGGAAAAAAAUGCCCC"
    feats = sd_aug_features(seq)
    assert feats["sd_aug_missing"] == 0
    assert feats["sd_aug_spacing"] == 6


def test_sd_aug_missing_sentinel_keeps_row() -> None:
    """Assert missing AUG uses the sentinel and composition features keep the row."""
    seq = "GGGGCCCCGGGGCCCC"  # no AUG
    feats = sd_aug_features(seq)
    assert feats["sd_aug_spacing"] == SD_AUG_SENTINEL
    assert feats["sd_aug_missing"] == 1
    df = pd.DataFrame(
        {
            "sequence": [seq, "AGGAGGAAAAAAAUGC"],
            "label": [0, 1],
            "viennarna_P_open_RBS_37": [0.1, 0.2],
        }
    )
    out = add_composition_features(df)
    assert len(out) == 2
    assert out.loc[0, "sd_aug_missing"] == 1
    assert out.loc[0, "sd_aug_spacing"] == SD_AUG_SENTINEL
    assert out["sd_aug_spacing"].notna().all()


def test_p_paired() -> None:
    """Assert P_paired is 1 minus P_open, and None is passed through."""
    assert p_paired_rbs_37(0.25) == 0.75
    assert p_paired_rbs_37(None) is None


def test_posthoc_gates() -> None:
    """Assert confidence bins and ΔP_RBS, Hill, Tm, and Z-score gate thresholds."""
    assert confidence_bin(0.81) == "high"
    assert confidence_bin(0.50) == "mid"
    assert confidence_bin(0.10) == "low"
    assert confidence_bin(0.70) == "other"
    assert gate_delta_p_rbs(0.01)
    assert not gate_delta_p_rbs(0.0)
    assert gate_hill(1.1)
    assert not gate_hill(1.0)
    assert gate_tm(43.0)
    assert not gate_tm(41.0)
    assert gate_zscore(-2.0)
    assert not gate_zscore(-1.9)


def test_spearman_underpowered_high_bin() -> None:
    """Assert Spearman is withheld below min_n and computed when unconstrained."""
    n = 10
    df = pd.DataFrame(
        {
            "viennarna_Tm": list(range(n)),
            "nupack_Tm": list(range(n)),
            "viennarna_hill_coeff": [1.0 + 0.05 * i for i in range(n)],
            "nupack_hill_coeff": [0.9 + 0.04 * i for i in range(n)],
            "viennarna_amplitude": [0.4 + 0.02 * i for i in range(n)],
            "nupack_amplitude": [0.35 + 0.02 * i for i in range(n)],
        }
    )
    high = spearman_consensus(df, require_min_n=True, min_n=SPEARMAN_MIN_N)
    assert high["underpowered"] is True
    assert high["n_complete"] == n
    for pair in high["pairs"].values():
        assert pair["r_s"] is None
        assert pair["underpowered"] is True
    panel = spearman_consensus(df, require_min_n=False)
    assert panel["underpowered"] is False
    assert panel["pairs"]["viennarna_Tm_vs_nupack_Tm"]["r_s"] is not None


def test_visual_checklist_flags() -> None:
    """Assert visual-checklist flags for a steep, switching Vienna row."""
    row = pd.Series(
        {
            "viennarna_hill_coeff": 1.8,
            "viennarna_Tm": 43.5,
            "viennarna_amplitude": 0.55,
            "viennarna_P_open_RBS_37": 0.05,
        }
    )
    flags = visual_checklist_flags(row)
    assert flags["sigmoidal_steepness_snap"]
    assert flags["inflection_tm"]
    assert flags["dynamic_range"]
    assert flags["baseline_repression"]
    assert not flags["sigmoidal_ramp"]


if __name__ == "__main__":
    test_kmer_keys_and_sum()
    test_sd_aug_spacing_known()
    test_sd_aug_missing_sentinel_keeps_row()
    test_p_paired()
    test_posthoc_gates()
    test_spearman_underpowered_high_bin()
    test_visual_checklist_flags()
    print("all noncircular / posthoc unit checks passed")
