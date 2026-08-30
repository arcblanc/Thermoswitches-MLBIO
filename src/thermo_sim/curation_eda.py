"""Exploratory visuals and tests for Rfam vs RefSeq curation (Marimo App 1).

Helpers keep plotting and statistics out of reactive cells so Ruff D103 and
type hints stay enforceable under ``uv run ruff check``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from thermo_sim.noncircular_features import (
    DINUC_COLUMNS,
    TRINUC_COLUMNS,
    add_composition_features,
    attach_sequences,
    aug_missing_by_class,
)
from thermo_sim.thermo_classifier import add_intensive_features
from thermo_sim.thermo_common import load_balanced_dataset

CLASS_POS = "thermoswitch (Rfam)"
CLASS_NEG = "RefSeq 5′ UTR"


def resolve_project_root(start: Path) -> Path:
    """Return the repo root that contains ``src/``, walking up from *start*."""
    root = start.resolve()
    if not (root / "src").exists() and (root.parent / "src").exists():
        root = root.parent
    return root


def load_curation_panel(project_root: Path) -> tuple[pd.DataFrame, Path, Path, Path]:
    """Load fused physics, attach sequences, add intensive + k-mer features."""
    fused = project_root / "data" / "processed" / "fused_features_refseq_dynamic.csv"
    dataset_csv = (
        project_root
        / "data"
        / "processed"
        / "balanced"
        / "length_gc_matched_refseq_dataset.csv"
    )
    dataset_fasta = (
        project_root
        / "data"
        / "processed"
        / "balanced"
        / "length_gc_matched_refseq_dataset.fasta"
    )
    frame = pd.read_csv(fused)
    frame = attach_sequences(
        frame, dataset_csv=str(dataset_csv), dataset_fasta=str(dataset_fasta)
    )
    panel = load_balanced_dataset(str(dataset_csv), str(dataset_fasta))
    for col in ("seq_start", "seq_end"):
        frame[col] = frame[col].astype(int)
        panel[col] = panel[col].astype(int)
    tax = panel[
        ["rfamseq_acc", "seq_start", "seq_end", "tax_string", "description", "type"]
    ].drop_duplicates(["rfamseq_acc", "seq_start", "seq_end"])
    frame = frame.merge(tax, on=["rfamseq_acc", "seq_start", "seq_end"], how="left")
    frame = add_intensive_features(frame)
    frame = add_composition_features(frame)
    frame["class"] = frame["label"].map({1: CLASS_POS, 0: CLASS_NEG})
    frame["family_or_control"] = np.where(
        frame["label"] == 1, frame["rfam_id"].astype(str), "RefSeq"
    )
    frame["gc_pct"] = frame["viennarna_gc_content"].astype(float) * 100.0
    # Prefer intensive fractions already built by add_intensive_features.
    if "nupack_max_stem_frac" in frame.columns:
        frame["stem_frac"] = frame["nupack_max_stem_frac"]
    elif "nupack_max_stem_length" in frame.columns and "seq_length" in frame.columns:
        frame["stem_frac"] = frame["nupack_max_stem_length"] / frame["seq_length"].clip(
            lower=1
        )
    if "viennarna_max_loop_frac" in frame.columns:
        frame["loop_frac"] = frame["viennarna_max_loop_frac"]
    elif "viennarna_max_loop_length" in frame.columns and "seq_length" in frame.columns:
        frame["loop_frac"] = frame["viennarna_max_loop_length"] / frame[
            "seq_length"
        ].clip(lower=1)
    return frame, fused, dataset_csv, dataset_fasta


def filter_panel(
    frame: pd.DataFrame,
    *,
    class_filter: str,
    family: str | None,
) -> pd.DataFrame:
    """Apply class and optional Rfam-family filters for reactive views."""
    out = frame
    if class_filter == "Rfam positives":
        out = out.loc[out["label"] == 1]
    elif class_filter == "RefSeq negatives":
        out = out.loc[out["label"] == 0]
    if family and family != "All families" and "rfam_acc" in out.columns:
        out = out.loc[(out["label"] == 0) | (out["rfam_acc"].astype(str) == family)]
    return out.copy()


def kpi_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute balance, missing-AUG, and family-count headline KPIs."""
    n = int(len(frame))
    n_pos = int((frame["label"] == 1).sum())
    n_neg = int((frame["label"] == 0).sum())
    aug = aug_missing_by_class(frame)
    n_fam = int(frame.loc[frame["label"] == 1, "rfam_acc"].nunique())
    missingness = {
        col: float(frame[col].isna().mean())
        for col in (
            "seq_length",
            "viennarna_gc_content",
            "viennarna_MFE_per_nt",
            "viennarna_ensemble_diversity",
            "P_paired_RBS_37",
            "sd_aug_spacing",
        )
        if col in frame.columns
    }
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_families": n_fam,
        "n_missing_aug_pos": int(aug.get("n_missing_aug_pos", 0)),
        "n_missing_aug_neg": int(aug.get("n_missing_aug_neg", 0)),
        "missingness": missingness,
        "balanced": n_pos == n_neg and n_pos > 0,
    }


def two_sample_ks(a: pd.Series, b: pd.Series) -> dict[str, float | None]:
    """Two-sample Kolmogorov–Smirnov test on finite numeric values."""
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 3 or len(y) < 3:
        return {"statistic": None, "pvalue": None, "n_a": len(x), "n_b": len(y)}
    stat, p = stats.ks_2samp(x, y)
    return {
        "statistic": float(stat),
        "pvalue": float(p),
        "n_a": int(len(x)),
        "n_b": int(len(y)),
    }


def wasserstein(a: pd.Series, b: pd.Series) -> float | None:
    """Earth-mover distance between two numeric series (None if underpowered)."""
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) < 3 or len(y) < 3:
        return None
    return float(stats.wasserstein_distance(x, y))


def confounder_audit_table(frame: pd.DataFrame) -> pd.DataFrame:
    """KS + Wasserstein table proving length/%GC matching removed shortcuts."""
    pos = frame.loc[frame["label"] == 1]
    neg = frame.loc[frame["label"] == 0]
    rows = []
    for feature, col in (("Length (nt)", "seq_length"), ("%GC", "gc_pct")):
        ks = two_sample_ks(pos[col], neg[col])
        rows.append(
            {
                "feature": feature,
                "KS_stat": ks["statistic"],
                "KS_p": ks["pvalue"],
                "Wasserstein": wasserstein(pos[col], neg[col]),
                "match_ok_p_gt_0.05": (
                    ks["pvalue"] is not None and float(ks["pvalue"]) > 0.05
                ),
            }
        )
    return pd.DataFrame(rows)


def gc_biological_parity_check(
    frame: pd.DataFrame,
    *,
    d_max: float = 0.06,
    mean_delta_gc_max: float = 1.5,
) -> dict[str, Any]:
    """Judge residual %GC imbalance as biologically minor vs matching failure.

    Compares the two-sample KS statistic $D$ with the mean absolute %GC gap
    $\\overline{\\Delta GC} = |\\mathrm{mean}(\\%GC_{pos}) - \\mathrm{mean}(\\%GC_{neg})|$.
    When $D < d\\_max$ (default 0.06) and $\\overline{\\Delta GC} \\le$
    ``mean_delta_gc_max`` (default 1.5%), the shift is treated as biologically
    minor and still inside the strict 5% Hungarian matching gate — even if KS
    $p$ dips slightly below 0.05 on a large $N$.
    """
    pos = frame.loc[frame["label"] == 1, "gc_pct"]
    neg = frame.loc[frame["label"] == 0, "gc_pct"]
    ks = two_sample_ks(pos, neg)
    mean_pos = float(pd.to_numeric(pos, errors="coerce").mean())
    mean_neg = float(pd.to_numeric(neg, errors="coerce").mean())
    mean_delta = abs(mean_pos - mean_neg)
    d_stat = ks["statistic"]
    d_ok = d_stat is not None and float(d_stat) < d_max
    delta_ok = mean_delta <= mean_delta_gc_max
    passed = bool(d_ok and delta_ok)
    return {
        "KS_D": d_stat,
        "KS_p": ks["pvalue"],
        "mean_gc_pos": mean_pos,
        "mean_gc_neg": mean_neg,
        "mean_abs_delta_gc": mean_delta,
        "d_max": d_max,
        "mean_delta_gc_max": mean_delta_gc_max,
        "d_ok": d_ok,
        "delta_ok": delta_ok,
        "passed": passed,
    }


def cliffs_delta(a: pd.Series, b: pd.Series) -> float | None:
    """Cliff's delta effect size (dominance of *a* over *b*)."""
    x = pd.to_numeric(a, errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(b, errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None
    # Efficient pairwise dominance via sorted search.
    y_sorted = np.sort(y)
    gt = 0
    lt = 0
    for xi in x:
        gt += int(len(y_sorted) - np.searchsorted(y_sorted, xi, side="right"))
        lt += int(np.searchsorted(y_sorted, xi, side="left"))
    n = float(len(x) * len(y))
    return float((gt - lt) / n)


def mannwhitney(a: pd.Series, b: pd.Series) -> dict[str, float | None]:
    """Mann–Whitney U with Cliff's delta for one feature."""
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if len(x) < 3 or len(y) < 3:
        return {
            "U": None,
            "p": None,
            "cliffs_delta": None,
            "n_a": len(x),
            "n_b": len(y),
        }
    u_stat, p_value = stats.mannwhitneyu(x, y, alternative="two-sided")
    return {
        "U": float(u_stat),
        "p": float(p_value),
        "cliffs_delta": cliffs_delta(x, y),
        "n_a": int(len(x)),
        "n_b": int(len(y)),
    }


def biophysics_contrast_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Mann–Whitney / Cliff's δ for ground-state physics features."""
    pos = frame.loc[frame["label"] == 1]
    neg = frame.loc[frame["label"] == 0]
    features = [
        ("MFE/nt (Vienna)", "viennarna_MFE_per_nt"),
        ("MFE/nt (NUPACK)", "nupack_MFE_per_nt"),
        ("Ensemble diversity Q", "viennarna_ensemble_diversity"),
        ("Mean positional entropy S", "viennarna_mean_positional_entropy"),
        ("Max stem fraction", "stem_frac"),
        ("Max loop fraction", "loop_frac"),
    ]
    rows = []
    for name, col in features:
        if col not in frame.columns:
            continue
        mw = mannwhitney(pos[col], neg[col])
        rows.append(
            {
                "feature": name,
                "column": col,
                "MW_U": mw["U"],
                "MW_p": mw["p"],
                "cliffs_delta": mw["cliffs_delta"],
            }
        )
    return pd.DataFrame(rows)


def sd_aug_chi_square(frame: pd.DataFrame) -> dict[str, Any]:
    """χ² independence test for missing-AUG flag vs class label."""
    if "sd_aug_missing" not in frame.columns:
        return {"chi2": None, "p": None, "dof": None, "table": None}
    table = pd.crosstab(frame["label"], frame["sd_aug_missing"])
    if table.shape != (2, 2):
        return {"chi2": None, "p": None, "dof": None, "table": table}
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {
        "chi2": float(chi2),
        "p": float(p_value),
        "dof": int(dof),
        "table": table,
    }


def spacing_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize SD–AUG spacing with the −1 missing-AUG sentinel separated."""
    rows = []
    for label, name in ((1, CLASS_POS), (0, CLASS_NEG)):
        sub = frame.loc[frame["label"] == label]
        spacing = pd.to_numeric(sub["sd_aug_spacing"], errors="coerce")
        present = spacing[spacing >= 0]
        rows.append(
            {
                "class": name,
                "n": int(len(sub)),
                "n_missing_aug": int((spacing < 0).sum()),
                "median_spacing_nt": float(present.median()) if len(present) else None,
                "frac_canonical_5_10": (
                    float(((present >= 5) & (present <= 10)).mean())
                    if len(present)
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def kmer_enrichment_table(
    frame: pd.DataFrame,
    cols: list[str],
    *,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Welch t-tests on k-mer frequencies with Benjamini–Hochberg FDR."""
    pos = frame.loc[frame["label"] == 1]
    neg = frame.loc[frame["label"] == 0]
    rows = []
    for col in cols:
        if col not in frame.columns:
            continue
        a = pd.to_numeric(pos[col], errors="coerce").dropna()
        b = pd.to_numeric(neg[col], errors="coerce").dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
        rows.append(
            {
                "feature": col,
                "mean_pos": float(a.mean()),
                "mean_neg": float(b.mean()),
                "delta_pos_minus_neg": float(a.mean() - b.mean()),
                "t_stat": float(t_stat),
                "p_raw": float(p_value),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values("p_raw")
    m = len(out)
    ranks = np.arange(1, m + 1)
    adj = out["p_raw"].to_numpy() * m / ranks
    # Enforce monotonicity of BH adjusted p-values.
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out["p_fdr"] = np.clip(adj, 0.0, 1.0)
    out["significant"] = out["p_fdr"] <= fdr
    return out.sort_values(["significant", "p_fdr"], ascending=[False, True])


def class_mean_kmers(
    frame: pd.DataFrame, cols: list[str], prefix_strip: str
) -> pd.DataFrame:
    """Average k-mer frequencies per class into a long heatmap table."""
    rows: list[dict[str, object]] = []
    for cls, sub in frame.groupby("class"):
        means = sub[cols].mean()
        for col, val in means.items():
            mer = str(col).replace(prefix_strip, "")
            rows.append({"class": cls, "k-mer": mer, "frequency": float(val)})
    return pd.DataFrame(rows)


def embed_kmer_space(
    frame: pd.DataFrame,
    kmer_cols: list[str],
    *,
    method: str = "UMAP",
) -> tuple[np.ndarray, str]:
    """Embed scaled k-mer frequencies into 2D (UMAP preferred, PCA fallback)."""
    matrix = StandardScaler().fit_transform(frame[kmer_cols].fillna(0.0).to_numpy())
    if method == "UMAP":
        try:
            from umap import UMAP

            coords = UMAP(
                n_components=2, random_state=42, n_neighbors=30, min_dist=0.2
            ).fit_transform(matrix)
            return coords, "UMAP"
        except Exception as exc:  # noqa: BLE001 — UI fallback is intentional
            coords = PCA(n_components=2).fit_transform(matrix)
            return coords, f"PCA (UMAP unavailable: {exc})"
    coords = PCA(n_components=2).fit_transform(matrix)
    return coords, "PCA"


def ecdf_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Build empirical CDF long-form table for Altair step plots."""
    rows = []
    for cls, sub in frame.groupby("class"):
        values = np.sort(
            pd.to_numeric(sub[column], errors="coerce").dropna().to_numpy()
        )
        if len(values) == 0:
            continue
        frac = np.arange(1, len(values) + 1) / len(values)
        for x, y in zip(values, frac, strict=True):
            rows.append({"class": cls, column: float(x), "ecdf": float(y)})
    return pd.DataFrame(rows)


def qq_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Quantile–quantile pairs of positives vs negatives for one feature."""
    pos = np.sort(
        pd.to_numeric(frame.loc[frame["label"] == 1, column], errors="coerce")
        .dropna()
        .to_numpy()
    )
    neg = np.sort(
        pd.to_numeric(frame.loc[frame["label"] == 0, column], errors="coerce")
        .dropna()
        .to_numpy()
    )
    n = min(len(pos), len(neg))
    if n < 3:
        return pd.DataFrame(columns=["q_pos", "q_neg"])
    qs = np.linspace(0.01, 0.99, 99)
    return pd.DataFrame(
        {
            "q_pos": np.quantile(pos, qs),
            "q_neg": np.quantile(neg, qs),
        }
    )


def plot_biophysics_violins(long: pd.DataFrame):
    """Draw split violin+box panels for ground-state physics features.

    Returns a matplotlib Figure. Violins show full distribution shape; inset
    boxes mark median and IQR so outliers do not dominate the eye.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if long.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No biophysics columns available", ha="center")
        ax.axis("off")
        return fig

    features = list(long["feature"].unique())
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]
    palette = {CLASS_POS: "#2a6f97", CLASS_NEG: "#bc4749"}
    for ax, feat in zip(axes, features, strict=True):
        sub = long.loc[long["feature"] == feat]
        sns.violinplot(
            data=sub,
            x="class",
            y="value",
            hue="class",
            palette=palette,
            inner="box",
            cut=0,
            legend=False,
            ax=ax,
        )
        ax.set_title(feat, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("Value" if feat == features[0] else "")
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout()
    return fig


def plot_length_gc_step_densities(frame: pd.DataFrame):
    """Matplotlib paired step histograms for length and %GC (matching audit)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, col, title, xlabel in (
        (axes[0], "seq_length", "Length", "Length (nt)"),
        (axes[1], "gc_pct", "%GC", "%GC"),
    ):
        for label, name, color in (
            (1, CLASS_POS, "#2a6f97"),
            (0, CLASS_NEG, "#bc4749"),
        ):
            vals = (
                pd.to_numeric(frame.loc[frame["label"] == label, col], errors="coerce")
                .dropna()
                .to_numpy()
            )
            ax.hist(
                vals,
                bins=40,
                density=True,
                histtype="step",
                linewidth=1.8,
                label=name,
                color=color,
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


BIOPHYSICS_LONG_COLS = [
    ("viennarna_MFE_per_nt", "MFE/nt (Vienna)"),
    ("nupack_MFE_per_nt", "MFE/nt (NUPACK)"),
    ("viennarna_ensemble_diversity", "Ensemble diversity Q"),
    ("viennarna_mean_positional_entropy", "Positional entropy S"),
    ("stem_frac", "Max stem / L"),
    ("loop_frac", "Max loop / L"),
]


def biophysics_long(frame: pd.DataFrame) -> pd.DataFrame:
    """Melt ground-state physics columns for faceted violin charts."""
    rename = {col: label for col, label in BIOPHYSICS_LONG_COLS if col in frame.columns}
    if not rename:
        return pd.DataFrame(columns=["class", "feature", "value"])
    long = frame[["class", *rename.keys()]].melt(
        id_vars=["class"], var_name="feature", value_name="value"
    )
    long["feature"] = long["feature"].map(rename)
    return long.dropna(subset=["value"])


def family_choices(frame: pd.DataFrame) -> list[str]:
    """Dropdown labels: All families plus each positive Rfam accession."""
    fams = sorted(frame.loc[frame["label"] == 1, "rfam_acc"].astype(str).unique())
    return ["All families", *fams]


__all__ = [
    "BIOPHYSICS_LONG_COLS",
    "CLASS_NEG",
    "CLASS_POS",
    "DINUC_COLUMNS",
    "TRINUC_COLUMNS",
    "biophysics_contrast_table",
    "biophysics_long",
    "class_mean_kmers",
    "confounder_audit_table",
    "ecdf_frame",
    "embed_kmer_space",
    "family_choices",
    "filter_panel",
    "gc_biological_parity_check",
    "kmer_enrichment_table",
    "kpi_summary",
    "load_curation_panel",
    "plot_biophysics_violins",
    "plot_length_gc_step_densities",
    "qq_frame",
    "resolve_project_root",
    "sd_aug_chi_square",
    "spacing_summary",
    "two_sample_ks",
    "wasserstein",
]
