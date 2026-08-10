"""Length/GC-match negatives to positives via Z-space global assignment.

Uses cKDTree top-K neighbors + Hungarian assignment (linear_sum_assignment)
in standardized (length, GC) space, with hard |ΔL| / |ΔGC| gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS
from data_engineering.knn_undersample import (
    BALANCED_DIR,
    CDHIT_OUTPUT_DIR,
    METADATA_COLUMNS,
    load_labeled_pool,
    write_dataset,
)
from data_engineering.paths import resolve_path
from thermo_sim.thermo_common import gc_content

RANDOM_STATE = 42
DEFAULT_DELTA_L = 40.0
DEFAULT_DELTA_GC = 0.05
DEFAULT_TOP_K = 50
LARGE_COST = 1e6


def _join_key(row) -> tuple:
    return (str(row["rfamseq_acc"]), int(row["seq_start"]), int(row["seq_end"]))


def _annotate_length_gc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seq_length"] = out["sequence"].str.len().astype(int)
    out["gc_content"] = out["sequence"].map(gc_content).astype(float)
    return out


def _zscore_fit(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-12:
        std = 1.0
    return mean, std


def _transform(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / std


def _truncate_neg_to_length(row: pd.Series, length: int) -> pd.Series:
    """Keep CDS-proximal 3' end of a longer UTR so length matches exactly."""
    out = row.copy()
    seq = str(out["sequence"])
    if len(seq) < length:
        raise ValueError("negative shorter than target length")
    if len(seq) > length:
        out["sequence"] = seq[-length:]
        if "seq_end" in out and "seq_start" in out:
            end = int(out["seq_end"])
            out["seq_start"] = end - length + 1
            out["seq_end"] = end
    out["seq_length"] = length
    out["gc_content"] = float(gc_content(out["sequence"]))
    out["truncated_from_length"] = int(len(seq))
    return out


def match_negatives_cds_truncate(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    delta_gc: float = DEFAULT_DELTA_GC,
    top_k: int = DEFAULT_TOP_K,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Match by truncating longer negatives to exact positive length (CDS-proximal).

    Length bias is removed by construction (ΔL=0). Assignment minimizes |ΔGC|
    among negatives with len >= positive length, with a hard |ΔGC| gate.
    """
    pos = _annotate_length_gc(positives).reset_index(drop=True)
    neg = _annotate_length_gc(negatives).reset_index(drop=True)
    pos_keys = {_join_key(r) for _, r in pos.iterrows()}
    neg = neg.loc[~neg.apply(_join_key, axis=1).isin(pos_keys)].reset_index(drop=True)

    pre = {
        "n_pos": int(len(pos)),
        "n_neg_pool": int(len(neg)),
        "mean_len_pos": float(pos["seq_length"].mean()),
        "mean_len_neg_pool": float(neg["seq_length"].mean()) if len(neg) else None,
        "mean_gc_pos": float(pos["gc_content"].mean()),
        "mean_gc_neg_pool": float(neg["gc_content"].mean()) if len(neg) else None,
        "delta_mu_length_pre": float(pos["seq_length"].mean() - neg["seq_length"].mean())
        if len(neg)
        else None,
        "mode": "cds_truncate",
    }
    if len(neg) == 0:
        return pos.iloc[0:0], neg.iloc[0:0], {
            **pre,
            "n_matched": 0,
            "n_unmatched": int(len(pos)),
            "error": "empty negative pool",
        }

    neg_seqs = neg["sequence"].astype(str).tolist()
    neg_lens = neg["seq_length"].to_numpy(int)
    pos_lens = pos["seq_length"].to_numpy(int)
    pos_gcs = pos["gc_content"].to_numpy(float)

    n_p = len(pos)
    candidate_pairs: list[list[tuple[float, int]]] = [[] for _ in range(n_p)]
    for i in range(n_p):
        lp = int(pos_lens[i])
        eligible = np.where(neg_lens >= lp)[0]
        if len(eligible) == 0:
            continue
        costs = []
        for j in eligible:
            trunc = neg_seqs[j][-lp:]
            g = float(gc_content(trunc))
            d_gc = abs(g - pos_gcs[i])
            if d_gc <= delta_gc:
                costs.append((d_gc, int(j)))
        costs.sort(key=lambda t: t[0])
        for d_gc, j in costs[:top_k]:
            candidate_pairs[i].append((d_gc, j))

    unique_neg = sorted({j for pairs in candidate_pairs for _, j in pairs})
    if not unique_neg:
        return pos.iloc[0:0], neg.iloc[0:0], {
            **pre,
            "n_matched": 0,
            "n_unmatched": int(len(pos)),
            "delta_gc_gate": delta_gc,
            "error": "no eligible truncate candidates within GC gate",
        }
    neg_local = {g: i for i, g in enumerate(unique_neg)}
    cost = np.full((n_p, len(unique_neg)), LARGE_COST, dtype=float)
    for i, pairs in enumerate(candidate_pairs):
        for d_gc, j in pairs:
            cost[i, neg_local[j]] = d_gc

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_pos_rows = []
    matched_neg_rows = []
    assignment_dists = []
    used_neg = set()
    matched_pos_idx = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= LARGE_COST / 2:
            continue
        j = unique_neg[c]
        if j in used_neg:
            continue
        lp = int(pos_lens[r])
        nrow = _truncate_neg_to_length(neg.iloc[j], lp)
        if abs(float(nrow["gc_content"]) - pos_gcs[r]) > delta_gc:
            continue
        matched_pos_rows.append(pos.iloc[r])
        matched_neg_rows.append(nrow)
        assignment_dists.append(float(cost[r, c]))
        used_neg.add(j)
        matched_pos_idx.add(r)

    rng = np.random.default_rng(random_state)
    remaining_pos = [i for i in range(n_p) if i not in matched_pos_idx]
    rng.shuffle(remaining_pos)
    for i in remaining_pos:
        lp = int(pos_lens[i])
        best = None
        for d_gc, j in candidate_pairs[i]:
            if j in used_neg:
                continue
            best = (d_gc, j)
            break
        if best is None:
            for j in np.where(neg_lens >= lp)[0]:
                if int(j) in used_neg:
                    continue
                trunc = neg_seqs[int(j)][-lp:]
                g = float(gc_content(trunc))
                d_gc = abs(g - pos_gcs[i])
                if d_gc <= delta_gc and (best is None or d_gc < best[0]):
                    best = (d_gc, int(j))
            if best is None:
                continue
        d_gc, j = best
        nrow = _truncate_neg_to_length(neg.iloc[j], lp)
        matched_pos_rows.append(pos.iloc[i])
        matched_neg_rows.append(nrow)
        assignment_dists.append(float(d_gc))
        used_neg.add(j)
        matched_pos_idx.add(i)

    matched_pos = pd.DataFrame(matched_pos_rows).reset_index(drop=True)
    matched_neg = pd.DataFrame(matched_neg_rows).reset_index(drop=True)
    unmatched_pos = pos.loc[~pos.index.isin(list(matched_pos_idx))].reset_index(drop=True)
    if len(matched_pos) and len(matched_neg):
        delta_mu_l = float(matched_pos["seq_length"].mean() - matched_neg["seq_length"].mean())
        delta_mu_gc = float(matched_pos["gc_content"].mean() - matched_neg["gc_content"].mean())
        delta_sigma_l = float(matched_pos["seq_length"].std() - matched_neg["seq_length"].std())
    else:
        delta_mu_l = delta_mu_gc = delta_sigma_l = None
    report = {
        **pre,
        "n_matched": int(len(matched_pos)),
        "n_unmatched": int(len(unmatched_pos)),
        "unmatched_fraction": float(len(unmatched_pos) / max(len(pos), 1)),
        "delta_mu_length_post": delta_mu_l,
        "delta_mu_gc_post": delta_mu_gc,
        "delta_sigma_length_post": delta_sigma_l,
        "mean_assignment_z_distance": float(np.mean(assignment_dists)) if assignment_dists else None,
        "delta_l_gate": 0.0,
        "delta_gc_gate": delta_gc,
        "top_k": top_k,
        "long_positive_truncation_risk": bool(
            len(unmatched_pos) > 0 and bool((unmatched_pos["seq_length"] > int(neg_lens.max())).any())
        ),
    }
    if len(unmatched_pos):
        report["unmatched_length_quantiles"] = {
            str(q): float(unmatched_pos["seq_length"].quantile(q)) for q in (0.5, 0.75, 0.9, 1.0)
        }
    return matched_pos, matched_neg, report


def match_negatives_to_positives(
    positives: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    delta_l: float = DEFAULT_DELTA_L,
    delta_gc: float = DEFAULT_DELTA_GC,
    top_k: int = DEFAULT_TOP_K,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (matched_pos, matched_neg, report)."""
    pos = _annotate_length_gc(positives).reset_index(drop=True)
    neg = _annotate_length_gc(negatives).reset_index(drop=True)

    # Exclude negatives that share join keys with positives (should not happen).
    pos_keys = {_join_key(r) for _, r in pos.iterrows()}
    neg = neg.loc[~neg.apply(_join_key, axis=1).isin(pos_keys)].reset_index(drop=True)

    pre = {
        "n_pos": int(len(pos)),
        "n_neg_pool": int(len(neg)),
        "mean_len_pos": float(pos["seq_length"].mean()),
        "mean_len_neg_pool": float(neg["seq_length"].mean()) if len(neg) else None,
        "mean_gc_pos": float(pos["gc_content"].mean()),
        "mean_gc_neg_pool": float(neg["gc_content"].mean()) if len(neg) else None,
        "delta_mu_length_pre": float(pos["seq_length"].mean() - neg["seq_length"].mean())
        if len(neg)
        else None,
    }

    if len(neg) == 0:
        report = {
            **pre,
            "n_matched": 0,
            "n_unmatched": int(len(pos)),
            "long_positive_truncation_risk": True,
            "error": "empty negative pool",
        }
        return pos.iloc[0:0], neg.iloc[0:0], report

    pooled_len = np.concatenate([pos["seq_length"].to_numpy(float), neg["seq_length"].to_numpy(float)])
    pooled_gc = np.concatenate([pos["gc_content"].to_numpy(float), neg["gc_content"].to_numpy(float)])
    len_mean, len_std = _zscore_fit(pooled_len)
    gc_mean, gc_std = _zscore_fit(pooled_gc)

    pos_xy = np.column_stack(
        [
            _transform(pos["seq_length"].to_numpy(float), len_mean, len_std),
            _transform(pos["gc_content"].to_numpy(float), gc_mean, gc_std),
        ]
    )
    neg_xy = np.column_stack(
        [
            _transform(neg["seq_length"].to_numpy(float), len_mean, len_std),
            _transform(neg["gc_content"].to_numpy(float), gc_mean, gc_std),
        ]
    )

    k = min(top_k, len(neg))
    tree = cKDTree(neg_xy)
    dists, idxs = tree.query(pos_xy, k=k)
    if k == 1:
        dists = dists.reshape(-1, 1)
        idxs = idxs.reshape(-1, 1)

    # Sparse bipartite: positives x unique top-K neighbors
    neighbor_sets = [set(int(j) for j in row) for row in idxs]
    unique_neg = sorted({j for s in neighbor_sets for j in s})
    neg_local = {g: i for i, g in enumerate(unique_neg)}
    n_p, n_c = len(pos), len(unique_neg)
    cost = np.full((n_p, n_c), LARGE_COST, dtype=float)
    for i in range(n_p):
        for d, j in zip(dists[i], idxs[i]):
            cost[i, neg_local[int(j)]] = float(d)

    row_ind, col_ind = linear_sum_assignment(cost)

    matched_pos_rows = []
    matched_neg_rows = []
    assignment_dists = []
    used_neg = set()
    matched_pos_idx = set()

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= LARGE_COST / 2:
            continue
        g = unique_neg[c]
        if g in used_neg:
            continue
        prow = pos.iloc[r]
        nrow = neg.iloc[g]
        d_l = abs(float(prow["seq_length"]) - float(nrow["seq_length"]))
        d_gc = abs(float(prow["gc_content"]) - float(nrow["gc_content"]))
        if d_l > delta_l or d_gc > delta_gc:
            continue
        matched_pos_rows.append(prow)
        matched_neg_rows.append(nrow)
        assignment_dists.append(float(cost[r, c]))
        used_neg.add(g)
        matched_pos_idx.add(r)

    # Greedy fallback for unmatched positives (shuffled order)
    rng = np.random.default_rng(random_state)
    remaining_pos = [i for i in range(len(pos)) if i not in matched_pos_idx]
    rng.shuffle(remaining_pos)
    remaining_neg = [j for j in range(len(neg)) if j not in used_neg]
    if remaining_pos and remaining_neg:
        rem_tree = cKDTree(neg_xy[remaining_neg])
        for i in remaining_pos:
            d, loc = rem_tree.query(pos_xy[i], k=min(k, len(remaining_neg)))
            loc = np.atleast_1d(loc)
            d = np.atleast_1d(d)
            for _, loc_j in sorted(zip(d, loc), key=lambda t: t[0]):
                g = remaining_neg[int(loc_j)]
                if g in used_neg:
                    continue
                prow = pos.iloc[i]
                nrow = neg.iloc[g]
                d_l = abs(float(prow["seq_length"]) - float(nrow["seq_length"]))
                d_gc = abs(float(prow["gc_content"]) - float(nrow["gc_content"]))
                if d_l <= delta_l and d_gc <= delta_gc:
                    matched_pos_rows.append(prow)
                    matched_neg_rows.append(nrow)
                    assignment_dists.append(float(np.linalg.norm(pos_xy[i] - neg_xy[g])))
                    used_neg.add(g)
                    matched_pos_idx.add(i)
                    break

    matched_pos = pd.DataFrame(matched_pos_rows).reset_index(drop=True)
    matched_neg = pd.DataFrame(matched_neg_rows).reset_index(drop=True)
    unmatched_pos = pos.loc[~pos.index.isin(matched_pos_idx)].reset_index(drop=True)

    if len(matched_pos) and len(matched_neg):
        delta_mu_l = float(matched_pos["seq_length"].mean() - matched_neg["seq_length"].mean())
        delta_mu_gc = float(matched_pos["gc_content"].mean() - matched_neg["gc_content"].mean())
        delta_sigma_l = float(matched_pos["seq_length"].std() - matched_neg["seq_length"].std())
    else:
        delta_mu_l = delta_mu_gc = delta_sigma_l = None

    unmatched_len = unmatched_pos["seq_length"].to_numpy() if len(unmatched_pos) else np.array([])
    matched_len = matched_pos["seq_length"].to_numpy() if len(matched_pos) else np.array([])
    truncation_risk = False
    if len(unmatched_len) and len(matched_len):
        truncation_risk = bool(np.median(unmatched_len) > np.quantile(matched_len, 0.9))
    elif len(unmatched_len) / max(len(pos), 1) > 0.05:
        truncation_risk = True

    report = {
        **pre,
        "n_matched": int(len(matched_pos)),
        "n_unmatched": int(len(unmatched_pos)),
        "unmatched_fraction": float(len(unmatched_pos) / max(len(pos), 1)),
        "delta_mu_length_post": delta_mu_l,
        "delta_mu_gc_post": delta_mu_gc,
        "delta_sigma_length_post": delta_sigma_l,
        "mean_assignment_z_distance": float(np.mean(assignment_dists)) if assignment_dists else None,
        "delta_l_gate": delta_l,
        "delta_gc_gate": delta_gc,
        "top_k": k,
        "unmatched_length_quantiles": {
            str(q): float(np.quantile(unmatched_len, q))
            for q in (0.5, 0.75, 0.9, 1.0)
        }
        if len(unmatched_len)
        else {},
        "matched_length_quantiles": {
            str(q): float(np.quantile(matched_len, q))
            for q in (0.5, 0.75, 0.9, 1.0)
        }
        if len(matched_len)
        else {},
        "long_positive_truncation_risk": truncation_risk,
    }
    return matched_pos, matched_neg, report


def escalate_long_candidates(
    unmatched_pos: pd.DataFrame,
    negatives: pd.DataFrame,
    *,
    length_floor_quantile: float = 0.5,
) -> pd.DataFrame:
    """Filter negatives into the unmatched positive length band for rematch."""
    if unmatched_pos.empty or negatives.empty:
        return negatives.iloc[0:0]
    floor = float(unmatched_pos["seq_length"].quantile(length_floor_quantile))
    neg = _annotate_length_gc(negatives)
    return neg.loc[neg["seq_length"] >= floor].reset_index(drop=True)


def build_matched_dataset(
    matched_pos: pd.DataFrame,
    matched_neg: pd.DataFrame,
) -> pd.DataFrame:
    pos = matched_pos.copy()
    neg = matched_neg.copy()
    pos["label"] = 1
    neg["label"] = 0
    cols = [c for c in METADATA_COLUMNS if c in pos.columns]
    extra = [c for c in ("sequence", "seq_length", "gc_content") if c in pos.columns]
    keep = list(dict.fromkeys(cols + extra))
    return pd.concat([pos[keep], neg[keep]], ignore_index=True)


def run_length_gc_match(
    *,
    fused_csv: str = "data/processed/fused_features.csv",
    balanced_csv: str = f"{BALANCED_DIR}/balanced_dataset.csv",
    positives_csv: str = f"{CDHIT_OUTPUT_DIR}/positives_deduped.csv",
    positives_fasta: str = f"{CDHIT_OUTPUT_DIR}/positives_deduped.fasta",
    negatives_csv: str = f"{BALANCED_DIR}/enn_cleaned.csv",
    negatives_fasta: str = f"{BALANCED_DIR}/enn_cleaned.fasta",
    fallback_negatives_csv: str = f"{CDHIT_OUTPUT_DIR}/negatives_deduped.csv",
    fallback_negatives_fasta: str = f"{CDHIT_OUTPUT_DIR}/negatives_deduped.fasta",
    output_csv: str = f"{BALANCED_DIR}/length_gc_matched_dataset.csv",
    output_fasta: str = f"{BALANCED_DIR}/length_gc_matched_dataset.fasta",
    report_json: str = f"{BALANCED_DIR}/length_gc_match_report.json",
    delta_l: float = DEFAULT_DELTA_L,
    delta_gc: float = DEFAULT_DELTA_GC,
    top_k: int = DEFAULT_TOP_K,
    random_state: int = RANDOM_STATE,
    require_fused_positives: bool = True,
    cds_truncate: bool = False,
) -> dict:
    pos_pool = load_labeled_pool(positives_csv, positives_fasta, label=1)
    for col in ("seq_start", "seq_end"):
        pos_pool[col] = pos_pool[col].astype(int)

    if require_fused_positives:
        fused = pd.read_csv(resolve_path(fused_csv))
        for col in ("seq_start", "seq_end"):
            fused[col] = fused[col].astype(int)
        fused_pos_keys = fused.loc[fused["label"] == 1, JOIN_COLUMNS]
        positives = pos_pool.merge(fused_pos_keys, on=JOIN_COLUMNS, how="inner")
    else:
        positives = pos_pool

    neg_csv_path = resolve_path(negatives_csv)
    if neg_csv_path.exists() and "sequence" in pd.read_csv(neg_csv_path, nrows=1).columns:
        neg_pool = pd.read_csv(neg_csv_path)
        if "label" not in neg_pool.columns:
            neg_pool["label"] = 0
        for col in ("seq_start", "seq_end"):
            if col in neg_pool.columns:
                neg_pool[col] = neg_pool[col].astype(int)
    else:
        try:
            neg_pool = load_labeled_pool(negatives_csv, negatives_fasta, label=0)
        except Exception:
            neg_pool = load_labeled_pool(fallback_negatives_csv, fallback_negatives_fasta, label=0)
    if "label" in neg_pool.columns:
        neg_pool = neg_pool.loc[neg_pool["label"] == 0].copy()
    for col in ("seq_start", "seq_end"):
        if col in neg_pool.columns:
            neg_pool[col] = neg_pool[col].astype(int)

    if cds_truncate:
        matched_pos, matched_neg, report = match_negatives_cds_truncate(
            positives,
            neg_pool,
            delta_gc=delta_gc,
            top_k=max(top_k, 100),
            random_state=random_state,
        )
    else:
        matched_pos, matched_neg, report = match_negatives_to_positives(
            positives,
            neg_pool,
            delta_l=delta_l,
            delta_gc=delta_gc,
            top_k=top_k,
            random_state=random_state,
        )

    # Escalation: rematch unmatched longs against long-only negative subset
    if (
        not cds_truncate
        and (report.get("long_positive_truncation_risk") or report.get("unmatched_fraction", 0) > 0.05)
    ):
        matched_keys = {_join_key(r) for _, r in matched_pos.iterrows()} if len(matched_pos) else set()
        unmatched = _annotate_length_gc(
            positives.loc[~positives.apply(_join_key, axis=1).isin(matched_keys)].reset_index(drop=True)
        )
        if len(unmatched):
            long_negs = escalate_long_candidates(unmatched, neg_pool)
            # Exclude already-used negatives
            used = {_join_key(r) for _, r in matched_neg.iterrows()} if len(matched_neg) else set()
            long_negs = long_negs.loc[~long_negs.apply(_join_key, axis=1).isin(used)].reset_index(drop=True)
            if len(long_negs):
                esc_pos, esc_neg, esc_report = match_negatives_to_positives(
                    unmatched,
                    long_negs,
                    delta_l=delta_l * 1.5,
                    delta_gc=delta_gc * 1.5,
                    top_k=top_k,
                    random_state=random_state,
                )
                report["escalation"] = esc_report
                if len(esc_pos):
                    matched_pos = pd.concat([matched_pos, esc_pos], ignore_index=True)
                    matched_neg = pd.concat([matched_neg, esc_neg], ignore_index=True)
                    # Refresh headline stats
                    report["n_matched"] = int(len(matched_pos))
                    report["n_unmatched"] = int(len(positives) - len(matched_pos))
                    report["unmatched_fraction"] = float(report["n_unmatched"] / max(len(positives), 1))
                    report["delta_mu_length_post"] = float(
                        matched_pos["seq_length"].mean() - matched_neg["seq_length"].mean()
                    )
                    report["delta_mu_gc_post"] = float(
                        matched_pos["gc_content"].mean() - matched_neg["gc_content"].mean()
                    )
                    unmatched_after = _annotate_length_gc(
                        positives.loc[
                            ~positives.apply(_join_key, axis=1).isin(
                                {_join_key(r) for _, r in matched_pos.iterrows()}
                            )
                        ]
                    )
                    if len(unmatched_after) and len(matched_pos):
                        report["long_positive_truncation_risk"] = bool(
                            unmatched_after["seq_length"].median()
                            > matched_pos["seq_length"].quantile(0.9)
                        )
                        report["unmatched_length_quantiles"] = {
                            str(q): float(unmatched_after["seq_length"].quantile(q))
                            for q in (0.5, 0.75, 0.9, 1.0)
                        }
                    else:
                        report["long_positive_truncation_risk"] = bool(len(unmatched_after) > 0)

    dataset = build_matched_dataset(matched_pos, matched_neg)
    write_dataset(dataset, output_csv, output_fasta)

    # Also write a negatives-only FASTA for thermo re-fold of new keys
    if not require_fused_positives:
        neg_only_csv = resolve_path(f"{BALANCED_DIR}/length_gc_matched_refseq_negatives.csv")
        neg_only_fa = resolve_path(f"{BALANCED_DIR}/length_gc_matched_refseq_negatives.fasta")
    else:
        neg_only_csv = resolve_path(f"{BALANCED_DIR}/length_gc_matched_negatives.csv")
        neg_only_fa = resolve_path(f"{BALANCED_DIR}/length_gc_matched_negatives.fasta")
    write_dataset(matched_neg.assign(label=0), neg_only_csv, neg_only_fa)

    report_path = resolve_path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report["output_csv"] = str(resolve_path(output_csv))
    report["output_fasta"] = str(resolve_path(output_fasta))
    report["negatives_csv"] = str(neg_only_csv)
    report["negatives_fasta"] = str(neg_only_fa)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote matched dataset → {resolve_path(output_csv)}")
    print(f"Wrote report → {report_path}")
    return report


def _build_parser():
    p = argparse.ArgumentParser(description="Length/GC-match negatives to positives.")
    p.add_argument("--delta-l", type=float, default=DEFAULT_DELTA_L)
    p.add_argument("--delta-gc", type=float, default=DEFAULT_DELTA_GC)
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--negatives-csv", default=None)
    p.add_argument("--negatives-fasta", default=None)
    p.add_argument("--output-csv", default=None)
    p.add_argument("--output-fasta", default=None)
    p.add_argument("--report-json", default=None)
    p.add_argument(
        "--all-positives",
        action="store_true",
        help="Use all CD-HIT positives (not only those already in fused_features).",
    )
    p.add_argument(
        "--cds-truncate",
        action="store_true",
        help="Truncate longer negatives to exact positive length (CDS-proximal 3' end).",
    )
    return p


def main():
    args = _build_parser().parse_args()
    kwargs = dict(delta_l=args.delta_l, delta_gc=args.delta_gc, top_k=args.top_k)
    if args.negatives_csv:
        kwargs["negatives_csv"] = args.negatives_csv
    if args.negatives_fasta:
        kwargs["negatives_fasta"] = args.negatives_fasta
    if args.output_csv:
        kwargs["output_csv"] = args.output_csv
    if args.output_fasta:
        kwargs["output_fasta"] = args.output_fasta
    if args.report_json:
        kwargs["report_json"] = args.report_json
    if args.all_positives:
        kwargs["require_fused_positives"] = False
        kwargs.setdefault("output_csv", f"{BALANCED_DIR}/length_gc_matched_refseq_dataset.csv")
        kwargs.setdefault("output_fasta", f"{BALANCED_DIR}/length_gc_matched_refseq_dataset.fasta")
        kwargs.setdefault("report_json", f"{BALANCED_DIR}/length_gc_matched_refseq_report.json")
        kwargs.setdefault("cds_truncate", True)
    if args.cds_truncate:
        kwargs["cds_truncate"] = True
    run_length_gc_match(**kwargs)


if __name__ == "__main__":
    main()
