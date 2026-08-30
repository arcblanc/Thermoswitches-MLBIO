"""Thesis Results 4.1–4.3 figure builders (RF collapse, checklist, EVA yield, Hill)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from thermo_sim.rf_posthoc import visual_checklist_flags
from thermo_sim.thermo_common import hill_sigmoid

# Display labels for grouped permutation blocks
PERM_BLOCK_LABELS = {
    "trinucleotides": "Trinucleotides (64)",
    "dinucleotides": "Dinucleotides (16)",
    "static_biophysics": "Static 37 °C biophysics",
    "composition": "Composition (L / %GC / P_pair)",
    "sd_aug": "SD–AUG spacing",
}

# Highlight the dominant crutch (trinucleotide block on both panels)
PERM_DOMINANT = {
    "trinucleotides",
}


class _TmPanelSpec(TypedDict):
    x: str
    y: str
    xlabel: str
    ylabel: str
    title: str
    r_s: float
    legend_loc: str
    rs_xy: tuple[float, float]
    rs_ha: Literal["left", "center", "right"]
    rs_va: Literal["top", "center", "bottom"]


class _GTSaveTable(Protocol):
    def gtsave(
        self,
        filename: str,
        *,
        zoom: float,
        vwidth: int,
        expand: int,
    ) -> None: ...


CHECKLIST_KEYS = (
    ("sigmoidal_steepness_snap", r"Snap $n_H>1.5$"),
    ("inflection_tm", r"$T_m\in[42,45]$"),
    ("dynamic_range", r"$\Delta\theta\geq0.50$"),
    ("baseline_repression", r"Baseline $\leq0.20$"),
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON sidecar."""
    return json.loads(path.read_text())


def stamp_png_dpi(path: Path, dpi: int = 96) -> None:
    """Write a PNG pHYs chunk so Docs/import tools treat pixels as screen DPI."""
    import struct
    import zlib

    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    ppm = int(round(dpi / 0.0254))  # pixels per metre
    phys = struct.pack(">IIB", ppm, ppm, 1)
    chunk_type = b"pHYs"
    chunk = (
        struct.pack(">I", len(phys))
        + chunk_type
        + phys
        + struct.pack(">I", zlib.crc32(chunk_type + phys) & 0xFFFFFFFF)
    )
    out = bytearray(raw[:8])
    pos = 8
    ihdr_end = None
    while pos + 8 <= len(raw):
        length = struct.unpack(">I", raw[pos : pos + 4])[0]
        ctype = raw[pos + 4 : pos + 8]
        chunk_end = pos + 12 + length
        if ctype == b"pHYs":
            pos = chunk_end
            continue
        out.extend(raw[pos:chunk_end])
        if ctype == b"IHDR":
            ihdr_end = len(out)
        pos = chunk_end
    if ihdr_end is None:
        raise ValueError(f"No IHDR in {path}")
    out[ihdr_end:ihdr_end] = chunk
    path.write_bytes(bytes(out))


def fit_docs_png_width(path: Path, *, target_width: int = 4592, dpi: int = 96) -> Path:
    """Scale a Docs table PNG to the shared cohort-map pixel width and re-stamp DPI."""
    from PIL import Image

    path = Path(path)
    image = Image.open(path)
    if image.width == target_width:
        stamp_png_dpi(path, dpi=dpi)
        return path
    new_h = max(1, round(image.height * target_width / image.width))
    image.convert("RGB").resize((target_width, new_h), Image.Resampling.LANCZOS).save(
        path
    )
    stamp_png_dpi(path, dpi=dpi)
    return path


def docs_table_options(px) -> dict[str, object]:
    """Shared great-tables options for Docs-readable thesis table PNGs."""
    return {
        "table_width": "1100px",
        "table_font_size": px(28),
        "table_font_color": "#111111",
        "table_font_weight": "normal",
        "heading_title_font_size": px(40),
        "heading_title_font_weight": "bold",
        "heading_subtitle_font_size": px(26),
        "heading_padding": px(22),
        "column_labels_font_size": px(24),
        "column_labels_font_weight": "bold",
        "column_labels_padding": px(18),
        "column_labels_padding_horizontal": px(20),
        "column_labels_background_color": "#E8EEF2",
        "data_row_padding": px(20),
        "data_row_padding_horizontal": px(20),
        "source_notes_font_size": px(20),
        "table_border_top_width": px(3),
        "table_border_bottom_width": px(3),
        "table_border_left_width": px(2),
        "table_border_right_width": px(2),
        "table_border_top_color": "#222222",
        "table_border_bottom_color": "#222222",
        "table_border_left_color": "#222222",
        "table_border_right_color": "#222222",
        "column_labels_border_bottom_width": px(2),
        "column_labels_border_bottom_color": "#222222",
        "table_body_hlines_style": "solid",
        "table_body_hlines_width": px(1),
        "table_body_hlines_color": "#B0B0B0",
        "row_striping_background_color": "#F7F9FB",
    }


def plot_cv_collapse_roc(curves: dict[str, Any]) -> Figure:
    """Overlay leaky StratifiedKFold vs honest StratifiedGroupKFold ROC curves."""
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    leaky = curves["roc_leaky"]
    honest = curves["roc_honest"]
    ax.plot(
        leaky["fpr"],
        leaky["tpr"],
        color="#C44536",
        lw=2.4,
        label=rf"StratifiedKFold (leaky) AUC = {curves['auc_skf']:.3f}",
    )
    ax.plot(
        honest["fpr"],
        honest["tpr"],
        color="#0D7377",
        lw=2.4,
        label=rf"StratifiedGroupKFold (honest) AUC = {curves['auc_sgkf']:.3f}",
    )
    ax.plot([0, 1], [0, 1], ls="--", color="0.55", lw=1.2, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Supervised baseline — CV vs group holdout", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _perm_frame(diag: dict[str, Any]) -> pd.DataFrame:
    """Build a sorted block / mean / std table from a diagnostics payload."""
    groups = diag["grouped_permutation_importance"]["groups"]
    rows = [
        {
            "key": name,
            "block": PERM_BLOCK_LABELS.get(name, name),
            "mean": float(block["mean_auc_drop"]),
            "std": float(block["std_auc_drop"]),
        }
        for name, block in groups.items()
    ]
    return pd.DataFrame(rows).sort_values("mean", ascending=True)


def plot_grouped_permutation_importance(diag: dict[str, Any]) -> Figure:
    """Single-panel horizontal bar chart (legacy helper)."""
    imp = _perm_frame(diag)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    colors = ["#C44536" if k in PERM_DOMINANT else "#4C72B0" for k in imp["key"]]
    ax.barh(
        imp["block"],
        imp["mean"],
        xerr=imp["std"],
        color=colors,
        ecolor="0.35",
        capsize=3,
        height=0.65,
    )
    ax.set_xlabel("Mean AUC drop (in-sample grouped permutation)")
    ax.set_title("Feature attribution", pad=12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_naive_vs_strict_attribution(
    naive_diag: dict[str, Any],
    strict_diag: dict[str, Any],
) -> Figure:
    """Two-panel thesis figure: naive raw-MFE vs strict MFE/nt attribution.

    Same non-circular feature *blocks* on both sides. The only intentional
    differentiator is extensive MFE (naive) versus intensive MFE/nt (strict).
    """
    # Shared y order (barh: first row = bottom). Trinucleotides on top.
    block_order_bottom_to_top = [
        "sd_aug",
        "composition",
        "static_biophysics",
        "dinucleotides",
        "trinucleotides",
    ]

    def _aligned(diag: dict[str, Any]) -> pd.DataFrame:
        groups = diag["grouped_permutation_importance"]["groups"]
        rows = []
        for key in block_order_bottom_to_top:
            if key not in groups:
                continue
            block = groups[key]
            rows.append(
                {
                    "key": key,
                    "block": PERM_BLOCK_LABELS.get(key, key),
                    "mean": float(block["mean_auc_drop"]),
                    "std": float(block["std_auc_drop"]),
                }
            )
        return pd.DataFrame(rows)

    naive = _aligned(naive_diag)
    strict = _aligned(strict_diag)
    xmax = max(naive["mean"].max(), strict["mean"].max(), 0.05) * 1.18

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    panels = [
        (
            axes[0],
            naive,
            "A  ·  Naive model",
            "Same blocks as strict\nexcept raw MFE (not MFE/nt)",
        ),
        (
            axes[1],
            strict,
            "B  ·  Strict model",
            "Non-circular $X$\nwith intensive MFE/nt",
        ),
    ]
    for ax, imp, title, subtitle in panels:
        colors = ["#C44536" if k in PERM_DOMINANT else "#4C72B0" for k in imp["key"]]
        ax.barh(
            imp["block"],
            imp["mean"],
            xerr=imp["std"],
            color=colors,
            ecolor="0.35",
            capsize=3,
            height=0.62,
        )
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Mean AUC drop (grouped permutation)")
        ax.set_title(title, fontsize=12, pad=10, loc="left")
        ax.text(
            0.98,
            0.02,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color="#555555",
            linespacing=1.35,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.9", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Feature attribution — naive raw MFE vs strict MFE/nt (matched blocks)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_oof_confidence_hist(curves: dict[str, Any]) -> Figure:
    """Histogram of out-of-fold $\\hat{y}$ under StratifiedGroupKFold."""
    hist = curves["yhat_hist"]
    edges = np.asarray(hist["edges"], dtype=float)
    counts = np.asarray(hist["counts"], dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = float(np.diff(edges).mean())

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(centers, counts, width=width * 0.92, color="#4C72B0", edgecolor="white")
    ax.axvline(0.20, color="#C44536", ls="--", lw=1.2, label=r"Low $\leq 0.20$")
    ax.axvline(0.80, color="#0D7377", ls="--", lw=1.2, label=r"High $\geq 0.80$")
    bins = curves["confidence_bins"]
    ax.set_xlabel(r"OOF $\hat{y}$ (StratifiedGroupKFold)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence calibration — out-of-family transfer", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(
        0.98,
        0.72,
        f"Low $n={bins['low']}$\nHigh $n={bins['high']}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75"},
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def _checklist_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """Boolean membership matrix for the four signature gates."""
    flags = [visual_checklist_flags(row) for _, row in frame.iterrows()]
    mat = pd.DataFrame(flags)
    return mat[[k for k, _ in CHECKLIST_KEYS]]


def plot_checklist_upset(frame: pd.DataFrame) -> Figure:
    """Simplified UpSet: top exclusive intersections + all-four column."""
    mat = _checklist_membership(frame)
    keys = list(mat.columns)
    labels = [lab for _, lab in CHECKLIST_KEYS]
    arr = mat.to_numpy().astype(bool)
    n = len(keys)

    specs: list[tuple[str, frozenset[int], np.ndarray]] = []
    # Exclusive singletons
    for i in range(n):
        mask = arr[:, i].copy()
        for j in range(n):
            if j != i:
                mask &= ~arr[:, j]
        specs.append((labels[i], frozenset({i}), mask))
    # Exclusive pairs
    for i in range(n):
        for j in range(i + 1, n):
            mask = arr[:, i] & arr[:, j]
            for t in range(n):
                if t not in (i, j):
                    mask &= ~arr[:, t]
            specs.append((f"{labels[i]} ∩ {labels[j]}", frozenset({i, j}), mask))
    # All four
    specs.append(("All four", frozenset(range(n)), arr.all(axis=1)))

    ranked = sorted(
        [
            (name, members, int(m.sum()))
            for name, members, m in specs
            if name != "All four"
        ],
        key=lambda t: t[2],
        reverse=True,
    )
    show = ranked[:7]
    all_four_n = int(arr.all(axis=1).sum())
    show.append(("All four", frozenset(range(n)), all_four_n))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 5.8),
        gridspec_kw={"height_ratios": [2.2, 1.05], "hspace": 0.1},
        sharex=True,
    )
    ax_bar, ax_mat = axes
    xs = np.arange(len(show))
    counts = [c for _, _, c in show]
    colors = ["#C44536" if name == "All four" else "#4C72B0" for name, _, _ in show]
    ax_bar.bar(xs, counts, color=colors, width=0.72)
    ax_bar.set_ylabel("Intersection size")
    ax_bar.set_title("Visual checklist — gate intersections", pad=12)
    ymax = max(counts) if max(counts) > 0 else 1
    for x, c in zip(xs, counts, strict=True):
        ax_bar.text(x, c + ymax * 0.015, str(c), ha="center", va="bottom", fontsize=8)
    ax_bar.set_ylim(0, ymax * 1.18)
    ax_bar.spines[["top", "right"]].set_visible(False)

    for row in range(n):
        ax_mat.plot([-0.4, len(show) - 0.6], [row, row], color="0.9", lw=5, zorder=1)
        for col, (_, members, _) in enumerate(show):
            on = row in members
            ax_mat.scatter(
                col,
                row,
                s=60,
                c="#222222" if on else "0.85",
                zorder=3,
            )
            if on and len(members) > 1:
                # connect active dots in a column
                pass
    for col, (_, members, _) in enumerate(show):
        if len(members) >= 2:
            ys = sorted(members)
            ax_mat.plot([col, col], [ys[0], ys[-1]], color="#222222", lw=1.5, zorder=2)

    ax_mat.set_yticks(range(n))
    ax_mat.set_yticklabels(labels, fontsize=9)
    ax_mat.set_xticks(xs)
    ax_mat.set_xticklabels(
        [name if name == "All four" else "" for name, _, _ in show],
        fontsize=8,
    )
    ax_mat.set_ylim(-0.7, n - 0.3)
    ax_mat.spines[["top", "right", "bottom"]].set_visible(False)
    ax_mat.tick_params(axis="x", length=0)
    fig.subplots_adjust(hspace=0.12, top=0.90, bottom=0.08)
    return fig


def plot_tm_concordance(
    frame: pd.DataFrame,
    *,
    r_s_mfe: float,
    r_s_tm: float,
) -> Figure:
    """Side-by-side Vienna vs NUPACK: static MFE (A) vs melting $T_m$ (B)."""
    class_style = (
        (1, "Rfam (pos)", "#5E4B8B"),
        (0, "RefSeq (neg)", "#E07A3D"),
    )
    panels: tuple[_TmPanelSpec, _TmPanelSpec] = (
        {
            "x": "viennarna_MFE",
            "y": "nupack_MFE",
            "xlabel": r"Vienna MFE (kcal/mol)",
            "ylabel": r"NUPACK MFE (kcal/mol)",
            "title": r"A · Static architecture (MFE)",
            "r_s": r_s_mfe,
            "legend_loc": "lower right",
            "rs_xy": (0.05, 0.95),
            "rs_ha": "left",
            "rs_va": "top",
        },
        {
            "x": "viennarna_Tm",
            "y": "nupack_Tm",
            "xlabel": r"Vienna $T_m$ (°C)",
            "ylabel": r"NUPACK $T_m$ (°C)",
            "title": r"B · Dynamic phenotype ($T_m$)",
            "r_s": r_s_tm,
            "legend_loc": "upper left",
            "rs_xy": (0.95, 0.05),
            "rs_ha": "right",
            "rs_va": "bottom",
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.8))
    for ax, spec in zip(axes, panels, strict=True):
        cols = [spec["x"], spec["y"], "label"]
        pts = frame[cols].dropna()
        for lab, name, color in class_style:
            sub = pts.loc[pts["label"] == lab]
            ax.scatter(
                sub[spec["x"]],
                sub[spec["y"]],
                s=12,
                alpha=0.35,
                color=color,
                edgecolors="none",
                label=name,
            )
        lo = float(min(pts[spec["x"]].min(), pts[spec["y"]].min()))
        hi = float(max(pts[spec["x"]].max(), pts[spec["y"]].max()))
        ax.plot([lo, hi], [lo, hi], ls="--", color="0.3", lw=1.1, label=r"$y=x$")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel(spec["ylabel"])
        ax.set_title(spec["title"], pad=10, fontsize=11)
        ax.text(
            spec["rs_xy"][0],
            spec["rs_xy"][1],
            rf"$r_s \approx {spec['r_s']:.3f}$",
            transform=ax.transAxes,
            ha=spec["rs_ha"],
            va=spec["rs_va"],
            fontsize=11,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.92,
            },
        )
        ax.legend(frameon=False, fontsize=8, loc=spec["legend_loc"])
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Vienna vs NUPACK — static MFE agrees; melting $T_m$ does not",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    return fig


def plot_eva_attrition_funnel(stages: list[tuple[str, int]]) -> Figure:
    """Professional horizontal attrition chart for EVA yield triage.

    Stage labels sit on the y-axis; bars encode absolute *n*; annotations
    show percent of the soft-drop cohort retained and stepwise loss.
    """
    labels = [s[0] for s in stages]
    values = [int(s[1]) for s in stages]
    n0 = values[0] if values else 1
    fracs = [v / n0 if n0 else 0.0 for v in values]
    drops = [None] + [values[i - 1] - values[i] for i in range(1, len(values))]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    # Higher y at top → reverse index so stage 1 sits first
    y = np.arange(len(stages))[::-1]
    palette = ["#C5DADB", "#7AADB0", "#3D8A8C", "#0D7377"]
    colors = [palette[min(i, len(palette) - 1)] for i in range(len(stages))]

    ax.barh(
        y,
        values,
        height=0.58,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        zorder=2,
    )

    xmax = max(values) if values else 1
    ax.set_xlim(0, xmax * 1.32)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Sequences retained", fontsize=11)
    ax.set_title("EVA generative triage — attrition to yield gate", pad=14, fontsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.9", lw=0.8, zorder=0)
    ax.set_axisbelow(True)

    for yi, n, frac, drop in zip(y, values, fracs, drops, strict=True):
        note = f"{n:,}  ({100 * frac:.1f}%)"
        if drop is not None:
            note = f"{note}   −{drop:,}"
        ax.text(
            n + xmax * 0.018,
            yi,
            note,
            va="center",
            ha="left",
            fontsize=9,
            color="#222222",
        )

    if len(values) >= 2:
        ax.text(
            0.99,
            -0.14,
            f"Overall yield  {values[-1]:,} / {n0:,}  =  {100 * values[-1] / n0:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#333333",
        )

    fig.tight_layout()
    return fig


def plot_hill_melting_overlay(
    rfam_params: pd.DataFrame,
    eva_curves: pd.DataFrame,
    *,
    temps: np.ndarray | None = None,
) -> Figure:
    """Overlay reconstructed Rfam Hill curves vs EVA temperature sweeps.

    Rfam: individual Hill reconstructions for high-cooperativity positives.
    EVA: per-sequence sweeps (thin) + mean ± IQR ribbon.
    """
    if temps is None:
        temps = np.linspace(30.0, 55.0, 251)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.axvspan(37, 42, color="#F5E6C8", zorder=0, label="37–42 °C window")

    rfam_stack = []
    plotted = 0
    for _, row in rfam_params.iterrows():
        bottom = float(row["bottom"])
        top = float(row["top"])
        curve = hill_sigmoid(
            temps,
            bottom,
            top,
            float(row["Tm"]),
            float(row["n_H"]),
        )
        rfam_stack.append(curve)
        if plotted < 50:
            ax.plot(temps, curve, color="#5E4B8B", alpha=0.15, lw=0.9, zorder=2)
            plotted += 1
    if rfam_stack:
        stack = np.vstack(rfam_stack)
        ax.plot(
            temps,
            stack.mean(axis=0),
            color="#5E4B8B",
            lw=2.4,
            label=rf"Rfam selected mean (n={len(rfam_stack)})",
            zorder=4,
        )

    p_col = "p_open" if "p_open" in eva_curves.columns else "p_unpaired"
    if not eva_curves.empty and p_col in eva_curves.columns:
        for _, sub in eva_curves.groupby("record_id"):
            sub = sub.sort_values("temp_C")
            ax.plot(
                sub["temp_C"],
                sub[p_col],
                color="#0D7377",
                alpha=0.07,
                lw=0.75,
                zorder=2,
            )
        g = (
            eva_curves.groupby("temp_C")[p_col]
            .agg(
                mean="mean",
                q25=lambda s: float(s.quantile(0.25)),
                q75=lambda s: float(s.quantile(0.75)),
            )
            .reset_index()
        )
        ax.fill_between(
            g["temp_C"],
            g["q25"],
            g["q75"],
            color="#0D7377",
            alpha=0.18,
            zorder=3,
        )
        ax.plot(
            g["temp_C"],
            g["mean"],
            color="#0D7377",
            lw=2.4,
            label=rf"EVA yield-gated mean (n={eva_curves['record_id'].nunique()})",
            zorder=5,
        )

    ax.set_xlim(30, 55)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"$P_{\mathrm{open}}$ / $\theta(T)$ (SD unpaired)")
    ax.set_title("Dynamic validation — Hill melting overlay", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_thermoswitch_mechanism() -> Figure:
    """Schematic: RNA thermoswitch basal lock (37 °C) vs heat-shock open (42 °C).

    Actors: RNA backbone ribbon, orange RBS block, AUG start codon,
    two-oval ribosome. Left = sequestered RBS / repressed;
    right = melted linear RBS / activated.
    """
    from matplotlib.patches import Arc, Ellipse, FancyArrowPatch, FancyBboxPatch
    import matplotlib.patches as mpatches

    RBS = "#E87722"
    AUG_C = "#6B4C9A"
    RNA = "#2C3E50"
    RIBO_OFF = "#B0B0B0"
    RIBO_ON = "#4A90C8"
    LOCK_BG = "#F8F1EC"
    OPEN_BG = "#EEF6F1"

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    fig.subplots_adjust(wspace=0.12, left=0.03, right=0.97, top=0.80, bottom=0.13)

    # ---- State 0: 37 °C basal lock ----
    ax0 = axes[0]
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.set_aspect("equal")
    ax0.axis("off")
    ax0.add_patch(
        FancyBboxPatch(
            (0.2, 0.3),
            9.6,
            9.1,
            boxstyle="round,pad=0.15,rounding_size=0.35",
            facecolor=LOCK_BG,
            edgecolor="#D4C4B8",
            linewidth=1.2,
            zorder=0,
        )
    )
    ax0.set_title(
        "37 °C  ·  Basal lock", fontsize=16, pad=12, color="#5C4033", fontweight="bold"
    )
    ax0.text(
        5.0,
        9.05,
        "Translation repressed",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#A04020",
    )

    # Hairpin stem (paired zipper)
    stem_x = 3.55
    for dx in (0.0, 1.15):
        ax0.plot(
            [stem_x + dx, stem_x + dx],
            [2.1, 5.7],
            color=RNA,
            lw=3.2,
            solid_capstyle="round",
            zorder=2,
        )
    for y in np.linspace(2.4, 5.4, 7):
        ax0.plot(
            [stem_x, stem_x + 1.15],
            [y, y],
            color=RNA,
            lw=1.1,
            alpha=0.55,
            zorder=2,
        )
    ax0.add_patch(
        Arc(
            (stem_x + 0.575, 5.7),
            1.15,
            1.3,
            theta1=0,
            theta2=180,
            color=RNA,
            lw=3.2,
            zorder=2,
        )
    )
    # 5′ is bottom of left stem; 3′ is bottom of right stem
    ax0.plot(
        [stem_x, 1.6], [2.1, 1.25], color=RNA, lw=3.2, solid_capstyle="round", zorder=2
    )
    ax0.plot(
        [stem_x + 1.15, 6.9],
        [2.1, 1.25],
        color=RNA,
        lw=3.2,
        solid_capstyle="round",
        zorder=2,
    )
    ax0.text(1.35, 1.0, "5′", fontsize=13, color=RNA, ha="center", fontweight="bold")
    ax0.text(7.15, 1.0, "3′", fontsize=13, color=RNA, ha="center", fontweight="bold")

    # RBS (more 5′ on left stem) then AUG (more 3′, toward loop)
    ax0.add_patch(
        FancyBboxPatch(
            (stem_x - 0.28, 2.85),
            0.55,
            1.25,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=RBS,
            edgecolor="#B85A12",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax0.text(
        stem_x - 0.45,
        3.45,
        "RBS",
        fontsize=13,
        fontweight="bold",
        color=RBS,
        ha="right",
        va="center",
    )
    ax0.add_patch(
        FancyBboxPatch(
            (stem_x - 0.28, 4.35),
            0.55,
            0.85,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=AUG_C,
            edgecolor="#4A3570",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax0.text(
        stem_x - 0.45,
        4.75,
        "AUG",
        fontsize=13,
        fontweight="bold",
        color=AUG_C,
        ha="right",
        va="center",
    )
    ax0.text(
        stem_x + 0.575,
        6.75,
        "Stem–loop\n(RBS + AUG sequestered)",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
        color="#222222",
    )

    # Ribosome — dashed / blocked (upper-right clear space)
    cx, cy = 7.85, 6.35
    ax0.add_patch(
        Ellipse(
            (cx - 0.35, cy),
            1.55,
            1.15,
            facecolor="#E8E8E8",
            edgecolor=RIBO_OFF,
            linewidth=1.6,
            linestyle="--",
            zorder=3,
        )
    )
    ax0.add_patch(
        Ellipse(
            (cx + 0.45, cy - 0.05),
            1.15,
            0.9,
            facecolor="#F2F2F2",
            edgecolor=RIBO_OFF,
            linewidth=1.6,
            linestyle="--",
            zorder=3,
        )
    )
    ax0.text(
        cx,
        cy + 1.05,
        "Ribosome",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#444444",
    )
    ax0.annotate(
        "",
        xy=(stem_x + 1.4, 3.5),
        xytext=(cx - 1.0, cy - 0.35),
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#999999",
            "lw": 1.4,
            "ls": "--",
            "connectionstyle": "arc3,rad=0.2",
        },
    )
    # Label in clear space under the ribosome (not on the stem)
    ax0.text(
        7.95,
        3.85,
        "Docking blocked",
        fontsize=12,
        color="#222222",
        fontweight="bold",
        ha="center",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#888888",
            "linewidth": 1.1,
        },
        zorder=6,
    )

    # ---- State 1: 42 °C heat shock ----
    ax1 = axes[1]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.add_patch(
        FancyBboxPatch(
            (0.2, 0.3),
            9.6,
            9.1,
            boxstyle="round,pad=0.15,rounding_size=0.35",
            facecolor=OPEN_BG,
            edgecolor="#B7CDBF",
            linewidth=1.2,
            zorder=0,
        )
    )
    ax1.set_title(
        "42 °C  ·  Heat shock", fontsize=16, pad=12, color="#1B5E4A", fontweight="bold"
    )
    ax1.text(
        5.0,
        9.05,
        "Translation activated",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#1B7A4A",
    )

    # Linear melted backbone — lower so labels sit above cleanly
    y_line = 3.55
    ax1.plot(
        [1.2, 8.8],
        [y_line, y_line],
        color=RNA,
        lw=3.4,
        solid_capstyle="round",
        zorder=2,
    )
    ax1.text(
        1.05, y_line - 0.5, "5′", fontsize=13, color=RNA, ha="center", fontweight="bold"
    )
    ax1.text(
        8.95, y_line - 0.5, "3′", fontsize=13, color=RNA, ha="center", fontweight="bold"
    )
    ax1.text(
        5.0,
        y_line - 1.0,
        "Linear RNA (melted stem)",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="#222222",
    )

    # 5′ → RBS → AUG → 3′
    rbs_x0, rbs_w = 3.35, 1.7
    aug_x0, aug_w = 5.25, 1.15
    ax1.add_patch(
        FancyBboxPatch(
            (rbs_x0, y_line - 0.38),
            rbs_w,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=RBS,
            edgecolor="#B85A12",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax1.text(
        rbs_x0 + rbs_w / 2,
        y_line,
        "RBS",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        zorder=5,
    )
    ax1.add_patch(
        FancyBboxPatch(
            (aug_x0, y_line - 0.38),
            aug_w,
            0.76,
            boxstyle="round,pad=0.02,rounding_size=0.1",
            facecolor=AUG_C,
            edgecolor="#4A3570",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax1.text(
        aug_x0 + aug_w / 2,
        y_line,
        "AUG",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
        zorder=5,
    )

    # Ribosome docked on RBS (not covering labels)
    rcx, rcy = rbs_x0 + rbs_w / 2, y_line + 1.85
    ax1.add_patch(
        Ellipse(
            (rcx - 0.4, rcy),
            1.7,
            1.25,
            facecolor="#D6EAF8",
            edgecolor=RIBO_ON,
            linewidth=2.0,
            zorder=3,
        )
    )
    ax1.add_patch(
        Ellipse(
            (rcx + 0.5, rcy - 0.05),
            1.25,
            1.0,
            facecolor="#EAF4FB",
            edgecolor=RIBO_ON,
            linewidth=2.0,
            zorder=3,
        )
    )
    ax1.text(
        rcx,
        rcy + 1.05,
        "Ribosome",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=RIBO_ON,
    )
    ax1.annotate(
        "",
        xy=(rcx, y_line + 0.42),
        xytext=(rcx, rcy - 0.7),
        arrowprops={"arrowstyle": "-|>", "color": RIBO_ON, "lw": 1.8},
    )
    # Callout clear of the ribosome (upper-right of panel)
    ax1.annotate(
        "Successfully docks",
        xy=(rcx + 0.95, rcy + 0.15),
        xytext=(8.15, 7.15),
        fontsize=12,
        color="#1A5F8F",
        fontweight="bold",
        ha="center",
        va="center",
        arrowprops={
            "arrowstyle": "-",
            "color": RIBO_ON,
            "lw": 1.2,
            "connectionstyle": "arc3,rad=-0.15",
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#4A90C8",
            "linewidth": 1.1,
        },
        zorder=6,
    )

    # Centre heat-shock arrow
    fig.patches.append(
        FancyArrowPatch(
            (0.475, 0.46),
            (0.525, 0.46),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.4,
            color="#333333",
            zorder=10,
        )
    )
    fig.text(
        0.50,
        0.545,
        r"Heat shock  $\Delta T$",
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="#222222",
    )
    fig.text(
        0.50,
        0.50,
        "37 °C  →  42 °C",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#333333",
    )

    fig.suptitle(
        "RNA thermoswitch mechanism — RBS sequestration vs thermal release",
        fontsize=17,
        fontweight="bold",
        y=0.97,
    )

    legend_elems = [
        mpatches.Patch(facecolor=RBS, edgecolor="#B85A12", label="RBS (docking zone)"),
        mpatches.Patch(facecolor=AUG_C, edgecolor="#4A3570", label="AUG (start codon)"),
        mpatches.Patch(
            facecolor="none", edgecolor=RNA, linewidth=2.5, label="RNA backbone"
        ),
        mpatches.Patch(
            facecolor="#D6EAF8", edgecolor=RIBO_ON, linewidth=1.5, label="Ribosome"
        ),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.005),
    )
    return fig


def rfam_hill_param_table(
    frame: pd.DataFrame, *, min_nh: float = 3.0, min_amp: float = 0.35, max_n: int = 80
) -> pd.DataFrame:
    """Build Hill reconstruction params for selected Rfam positives."""
    need = [
        "label",
        "viennarna_hill_coeff",
        "viennarna_Tm",
        "viennarna_amplitude",
        "viennarna_P_open_RBS_37",
    ]
    sub = frame.loc[frame["label"] == 1, need].dropna().copy()
    sub = sub[
        (sub["viennarna_hill_coeff"] >= min_nh)
        & (sub["viennarna_amplitude"] >= min_amp)
    ]
    sub = sub.sort_values("viennarna_amplitude", ascending=False).head(max_n)
    bottom = sub["viennarna_P_open_RBS_37"].astype(float)
    amp = sub["viennarna_amplitude"].astype(float)
    return pd.DataFrame(
        {
            "bottom": bottom,
            "top": (bottom + amp).clip(upper=1.0),
            "Tm": sub["viennarna_Tm"].astype(float),
            "n_H": sub["viennarna_hill_coeff"].astype(float),
        }
    )


RF_ENSEMBLE_CV_MERMAID = r"""
graph TD
    Dataset["Total Matched Dataset: N=2396"]

    Dataset -->|"5-Fold Grouped CV"| Train["Training Cohort: 4 folds"]
    Dataset -->|"5-Fold Grouped CV"| Test["Held-out Cohort: 1 fold"]

    Group["Grouped by RNA Covariance Family rfam_acc
and RefSeq Assembly
No homology overlap between Train and Validation Folds"]

    Dataset -.-> Group

    Train --> A["Input Vector x: k-mers and Thermodynamics"]

    A --> B("Decision Tree 1")
    A --> C("Decision Tree 2")
    A --> D("Decision Tree ... T=200")

    B -->|Gini Split| E["Prediction: 1"]
    C -->|Gini Split| F["Prediction: 0"]
    D -->|Gini Split| G["Prediction: 1"]

    E --> H{"Majority Vote / Aggregation"}
    F --> H
    G --> H

    H --> I["Final Ensemble Prediction
OOF y-hat · k=1…5"]
    Test -.->|OOD Validation| I

    style Dataset fill:#f3e5f5,stroke:#4a148c
    style Train fill:#e8f5e9,stroke:#1b5e20
    style Test fill:#ffebee,stroke:#b71c1c
    style Group fill:#fafafa,stroke:#616161,stroke-dasharray: 5 5
    style A fill:#e1f5fe,stroke:#01579b
    style H fill:#fff3e0,stroke:#e65100
"""

RF_ENSEMBLE_CV_MERMAID_V2 = r"""
flowchart TD
    Dataset["Total Matched Dataset (N = 2,396)"]

    SplitNote["Grouped by RNA Family (rfam_acc) & RefSeq Assembly<br><i>(Zero homology overlap across folds)</i>"]
    Dataset -.-> SplitNote

    TrainCohort["Training Cohort: 4 Folds"]
    TestCohort["Held-out Cohort: 1 Fold"]

    Dataset -->|"5-Fold Grouped CV"| TrainCohort
    Dataset -->|"5-Fold Grouped CV"| TestCohort

    FeatureVec["Non-Circular Feature Vector x (p = 92)<br>• 80 k-mers (16 di + 64 tri)<br>• 7 Static 37°C Physics<br>• 3 Compositional | 2 Spacing<br><i>[All dynamic Tm, Hill, Z excluded]</i>"]

    TrainCohort -->|"Fit Trees (T=200)"| Trees
    TestCohort --> FeatureVec
    FeatureVec --> Trees

    subgraph Trees ["Random Forest Ensemble (T = 200 Trees)"]
        direction LR
        T1["Tree 1<br>(Gini Split)"]
        T2["Tree 2<br>(Gini Split)"]
        Tdots["..."]
        T200["Tree 200<br>(Gini Split)"]
    end

    Trees -->|"h_t(x) ∈ [0, 1]"| Aggregation{"Mean Probability Aggregation<br>ŷ(x) = (1/T) ∑ h_t(x)"}

    Aggregation --> OOF["Final Out-of-Fold Predictions (ŷ)<br>• StratifiedGroupKFold ROC-AUC = 0.277<br>• Out-of-Distribution Generalisation Baseline"]
"""


def _export_mermaid_png(
    mermaid_src: str,
    path: Path,
    *,
    scale: int = 3,
    background: str = "white",
) -> Path:
    """Render a mermaid source string to PNG via mermaid-cli."""
    import shutil
    import subprocess
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    npx = shutil.which("npx")
    if npx is None:
        raise RuntimeError("npx not found — install Node.js for mermaid-cli PNG export")

    with tempfile.TemporaryDirectory() as tmp:
        mmd = Path(tmp) / "diagram.mmd"
        mmd.write_text(mermaid_src.strip() + "\n", encoding="utf-8")
        subprocess.run(
            [
                npx,
                "--yes",
                "@mermaid-js/mermaid-cli@11",
                "-i",
                str(mmd),
                "-o",
                str(path),
                "-b",
                background,
                "-s",
                str(scale),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    return path


def export_rf_grouped_cv_png(path: Path) -> Path:
    """Export 5-fold grouped-CV RF schematic (original tree-fan mermaid layout)."""
    return _export_mermaid_png(RF_ENSEMBLE_CV_MERMAID, path)


def export_rf_grouped_cv_v2_png(path: Path) -> Path:
    """Export version-2 potential RF schematic (p=92 feature vector + mean ŷ aggregation)."""
    return _export_mermaid_png(RF_ENSEMBLE_CV_MERMAID_V2, path)


def export_workflow_architecture_png(path: Path) -> Path:
    """Export the square EVA / RF / biophysics workflow diagram for thesis figures.

    Uses HTML layout (not mermaid-cli) so the PNG matches the Marimo mermaid view:
    EVA bar on top; Supervised RF bottom-left; in silico biophysics bottom-right.
    """
    from nokap import close, from_html

    html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px; background: #fff; font-family: "Helvetica Neue", Arial, sans-serif; color: #222; }
  #workflow-root { width: 980px; margin: 0 auto; position: relative; }
  .panel-title { font-size: 15px; font-weight: 600; margin: 0 0 12px 2px; }
  .panel { border: 2px solid; border-radius: 2px; padding: 16px 18px 18px; position: relative; }
  .panel.eva { border-color: #388e3c; background: #e8f5e9; margin-bottom: 18px; }
  .panel.eva .panel-title { color: #1b5e20; }
  .panel.rf { border-color: #616161; background: #fafafa; flex: 1 1 0; min-width: 0; }
  .panel.rf .panel-title { color: #424242; }
  .panel.bio { border-color: #f57c00; background: #fff8e1; flex: 1 1 0; min-width: 0; }
  .panel.bio .panel-title { color: #e65100; }
  .bottom-row { display: flex; gap: 18px; align-items: stretch; }
  .row { display: flex; gap: 14px; align-items: center; justify-content: center; flex-wrap: wrap; }
  .row + .row { margin-top: 14px; }
  .node {
    border: 2px solid; border-radius: 2px; padding: 10px 12px; text-align: center;
    font-size: 13px; line-height: 1.35; background: #fff; min-width: 118px;
  }
  .node.model { border-color: #388e3c; background: #e8f5e9; }
  .node.process { border-color: #0288d1; background: #e1f5fe; }
  .node.data { border-color: #333; background: #f9f9f9; }
  .node.denovo, .node.output { border-color: #f57c00; background: #fff3e0; }
  .node.wide { min-width: 250px; max-width: 360px; }
  .arrow { color: #333; font-size: 18px; line-height: 1; user-select: none; }
  .stack { display: flex; flex-direction: column; gap: 14px; align-items: center; }
  svg.cross { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; overflow: visible; }
  svg.cross path, svg.cross line { stroke: #333; stroke-width: 1.6; fill: none; marker-end: url(#arrowhead); }
</style></head><body>
<div id="workflow-root">
  <svg class="cross" viewBox="0 0 980 430" preserveAspectRatio="none">
    <defs>
      <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
        <path d="M0,0 L8,4 L0,8 z" fill="#333"/>
      </marker>
    </defs>
    <!-- De Novo → Thermal Sweep -->
    <path d="M 760 118 C 760 170, 700 205, 640 248"/>
    <!-- Rfam Positive → Thermal Sweep -->
    <path d="M 470 248 C 520 248, 580 252, 640 268"/>
  </svg>

  <section class="panel eva">
    <div class="panel-title">EVA generation</div>
    <div class="row">
      <div class="node model" id="eva-model">EVA Foundation<br>Model</div>
      <span class="arrow">→</span>
      <div class="node process">Latent Space<br>Sampling</div>
      <span class="arrow">→</span>
      <div class="node process">Yield-Gated<br>Filtering</div>
      <span class="arrow">→</span>
      <div class="node denovo" id="de-novo">De Novo Candidates<br>N=105</div>
    </div>
  </section>

  <div class="bottom-row">
    <section class="panel rf">
      <div class="panel-title">Supervised RF</div>
      <div class="row">
        <div class="node data">RefSeq Negative<br>N=1198</div>
        <div class="node data" id="rfam-pos">Rfam Positive<br>N=1198</div>
      </div>
      <div class="row"><span class="arrow">↓</span><span class="arrow">↓</span></div>
      <div class="row">
        <div class="node process wide">Feature Extraction<br>K-mers + Static Thermo Features</div>
      </div>
      <div class="row"><span class="arrow">↓</span></div>
      <div class="row">
        <div class="node model">Random Forest<br>Classifier · 5-fold grouped CV</div>
      </div>
    </section>

    <section class="panel bio">
      <div class="panel-title">In silico biophysics</div>
      <div class="stack">
        <div class="node process">Biophysical<br>Pipeline</div>
        <span class="arrow">↓</span>
        <div class="node process" id="thermal-sweep">Thermal Sweep<br>37C to 42C</div>
        <span class="arrow">↓</span>
        <div class="node output">Hill Metric<br>Extraction</div>
      </div>
    </section>
  </div>
</div>
</body></html>"""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from_html(
            html,
            path,
            selector="#workflow-root",
            expand=24,
            zoom=3,
            vwidth=1100,
            vheight=760,
            delay=0.4,
        )
    finally:
        close()
    return path


def _fmt_num(value: object, digits: int = 3) -> str:
    """Format a finite number for Docs tables; blank if missing."""
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "—"
        if not np.isfinite(number):
            return "—"
        return f"{number:.{digits}f}"
    return "—"


def _p_open_at_temp(row: pd.Series, temp_c: float) -> float:
    """Reconstruct RBS P_open at one temperature from Vienna Hill parameters."""
    from thermo_sim.eva_denovo_characterization import reconstruct_theta

    temps = np.asarray([float(temp_c)], dtype=float)
    theta = reconstruct_theta(row, temps, prefix="viennarna")
    return float(theta[0])


def _eva_mfe_per_nt(sequences: pd.Series) -> np.ndarray:
    """Vienna MFE density (kcal/mol/nt) for each sequence."""
    import RNA

    out = np.empty(len(sequences), dtype=float)
    for i, seq in enumerate(sequences.astype(str)):
        _struct, mfe = RNA.fold(seq)
        out[i] = float(mfe) / max(len(seq), 1)
    return out


def build_thermo_benchmark_tables(project_root: Path) -> pd.DataFrame:
    """Table 3: Rfam positives, EVA yield-gated (all), Tier-1 candidates A/B."""
    from thermo_sim.eva_denovo_characterization import (
        characterization_paths,
        load_control_panel,
        load_eva_passers,
        merge_eva_fits,
        tier_gate_flags,
    )

    root = Path(project_root)
    paths = characterization_paths(root)
    controls = load_control_panel(paths["fused"]).copy()
    rfam = controls.loc[controls["label"] == 1].copy()
    eva = load_eva_passers(paths).copy()
    hill = (
        pd.read_csv(paths["hill_cache"])
        if paths["hill_cache"].exists()
        else pd.DataFrame()
    )
    eva = merge_eva_fits(eva, hill)

    sweeps = (
        pd.read_csv(paths["sweep_cache"])
        if paths["sweep_cache"].exists()
        else pd.DataFrame()
    )
    if not sweeps.empty and {"record_id", "temp_C", "p_open"} <= set(sweeps.columns):
        p42 = (
            sweeps.loc[
                np.isclose(sweeps["temp_C"].astype(float), 42.0),
                ["record_id", "p_open"],
            ]
            .drop_duplicates("record_id")
            .rename(columns={"p_open": "P_open_42"})
        )
        eva = eva.merge(p42, on="record_id", how="left")
    else:
        eva["P_open_42"] = np.nan

    rfam["mfe_per_nt"] = rfam["viennarna_MFE"].astype(float) / rfam[
        "seq_length"
    ].astype(float)
    rfam["P_open_42"] = [_p_open_at_temp(row, 42.0) for _, row in rfam.iterrows()]

    cache_path = (
        root / "data" / "processed" / "eva_characterization" / "eva_mfe_per_nt.csv"
    )
    if cache_path.exists():
        mfe_cache = pd.read_csv(cache_path)
        eva = eva.drop(columns=["mfe_per_nt"], errors="ignore").merge(
            mfe_cache, on="record_id", how="left"
        )
    if "mfe_per_nt" not in eva.columns:
        eva["mfe_per_nt"] = np.nan
    need = eva["mfe_per_nt"].isna()
    if bool(need.any()):
        eva.loc[need, "mfe_per_nt"] = _eva_mfe_per_nt(eva.loc[need, "sequence"])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        eva[["record_id", "mfe_per_nt"]].to_csv(cache_path, index=False)

    miss42 = eva["P_open_42"].isna()
    if bool(miss42.any()):
        eva.loc[miss42, "P_open_42"] = [
            _p_open_at_temp(row, 42.0) for _, row in eva.loc[miss42].iterrows()
        ]

    def _cohort_row(name: str, frame: pd.DataFrame) -> dict[str, object]:
        return {
            "Cohort / Candidate": name,
            "N": int(len(frame)),
            "ΔGdensity": _fmt_num(frame["mfe_per_nt"].mean(), 3),
            "Popen(37°C)": _fmt_num(frame["viennarna_P_open_RBS_37"].mean(), 3),
            "Popen(42°C)": _fmt_num(frame["P_open_42"].mean(), 3),
            "ΔPRBS": _fmt_num(frame["viennarna_delta_P_RBS"].mean(), 3),
            "Tm (°C)": _fmt_num(frame["viennarna_Tm"].mean(), 1),
            "Hill h": _fmt_num(frame["viennarna_hill_coeff"].mean(), 2),
        }

    def _candidate_row(label: str, row: pd.Series) -> dict[str, object]:
        return {
            "Cohort / Candidate": f"{label} ({row['record_id']})",
            "N": 1,
            "ΔGdensity": _fmt_num(row.get("mfe_per_nt"), 3),
            "Popen(37°C)": _fmt_num(row.get("viennarna_P_open_RBS_37"), 3),
            "Popen(42°C)": _fmt_num(row.get("P_open_42"), 3),
            "ΔPRBS": _fmt_num(row.get("viennarna_delta_P_RBS"), 3),
            "Tm (°C)": _fmt_num(row.get("viennarna_Tm"), 1),
            "Hill h": _fmt_num(row.get("viennarna_hill_coeff"), 2),
        }

    hits = [
        row for _, row in eva.iterrows() if all(tier_gate_flags(row, tier=1).values())
    ]
    by_id = {str(row["record_id"]): row for row in hits}
    cand_a = by_id.get("eva_sample_315", hits[0] if hits else None)
    cand_b = by_id.get(
        "eva_sample_1858",
        hits[1] if len(hits) > 1 else (hits[0] if hits else None),
    )

    rows = [
        _cohort_row("Rfam Positive Controls", rfam),
        _cohort_row("EVA Yield-Gated (All)", eva),
    ]
    if cand_a is not None:
        rows.append(_candidate_row("Tier-1 A", cand_a))
    if cand_b is not None and (
        cand_a is None or str(cand_b["record_id"]) != str(cand_a["record_id"])
    ):
        rows.append(_candidate_row("Tier-1 B", cand_b))
    return pd.DataFrame(rows)


def _probe_pkg_version(import_name: str, dist_name: str | None = None) -> str:
    """Return installed package version or a fallback label."""
    import importlib
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    dist = dist_name or import_name
    try:
        return pkg_version(dist)
    except PackageNotFoundError:
        pass
    try:
        mod = importlib.import_module(import_name)
    except Exception:
        return "cluster / conda CLI"
    ver = getattr(mod, "__version__", None)
    return str(ver) if ver else "imported"


def _probe_cli_version(exe: str, *, flags: tuple[str, ...] | None = None) -> str:
    """Return a CLI version string, trying common flags and rejecting usage errors."""
    import re
    import shutil
    import subprocess

    if shutil.which(exe) is None:
        return "bioconda (thesis pipeline)"

    default_flags = ("-version", "--version", "-V", "-h", "-help", "--help")
    for flag in flags or default_flags:
        try:
            out = subprocess.run(
                [exe, flag],
                capture_output=True,
                text=True,
                check=False,
                timeout=8,
            )
        except Exception:
            continue
        text = "\n".join(part for part in (out.stdout, out.stderr) if part).strip()
        if not text:
            continue
        first = text.splitlines()[0].strip()
        lower = first.lower()
        if any(
            token in lower
            for token in (
                "failed to parse",
                "no such option",
                "usage:",
                "usage ",
                "unrecognized",
                "unknown option",
                "invalid option",
                "error:",
            )
        ):
            continue
        if lower in {"usage", "help"}:
            continue

        # HMMER / Infernal banner: "# HMMER 3.4 (Aug 2023); ..."
        banner = re.search(
            r"#\s*(?:INFERNAL|HMMER|NCBI BLAST)\s+([^\n;]+)",
            text,
            flags=re.IGNORECASE,
        )
        if banner:
            return banner.group(1).strip()[:80]

        # BLAST-style: "blastn: 2.17.0+"
        blast = re.search(rf"{re.escape(exe)}:\s*(\S+)", first, flags=re.IGNORECASE)
        if blast:
            return blast.group(1)[:80]

        # Generic version token on the first line.
        version = re.search(
            r"(?:version|release)\s*[:v]?\s*([0-9][0-9A-Za-z._+-]*)",
            first,
            flags=re.IGNORECASE,
        )
        if version:
            return version.group(1)[:80]

        if re.search(r"\b[0-9]+\.[0-9]+", first):
            return first[:80]

    return "bioconda (thesis pipeline)"


def build_thesis_software_table() -> pd.DataFrame:
    """Methods Table 2 — curated thesis software stack (Methods prose only)."""
    import sys

    py = sys.version.split()[0]
    pandas_v = _probe_pkg_version("pandas")
    sklearn_v = _probe_pkg_version("sklearn", "scikit-learn")
    scipy_v = _probe_pkg_version("scipy")
    rows: list[dict[str, str]] = [
        {
            "Category": "Runtime & Data",
            "Package / Tool": "Python",
            "Version / Source": py,
            "Primary Function in Pipeline": "Base runtime environment and script execution",
        },
        {
            "Category": "Runtime & Data",
            "Package / Tool": "Pandas / NumPy",
            "Version / Source": f"{pandas_v} / Standard",
            "Primary Function in Pipeline": "Tabular feature matrix construction and array operations",
        },
        {
            "Category": "Runtime & Data",
            "Package / Tool": "BioPython",
            "Version / Source": "Standard",
            "Primary Function in Pipeline": "FASTA sequence parsing and sequence I/O management",
        },
        {
            "Category": "Machine Learning",
            "Package / Tool": "Scikit-Learn",
            "Version / Source": f"{sklearn_v} (Pedregosa et al., 2011)",
            "Primary Function in Pipeline": "Random Forest training, bagging, and cross-validation",
        },
        {
            "Category": "Machine Learning",
            "Package / Tool": "SciPy",
            "Version / Source": f"{scipy_v} (Virtanen et al., 2020)",
            "Primary Function in Pipeline": "Non-linear least-squares Hill curve fitting (curve_fit)",
        },
        {
            "Category": "Bioinformatics",
            "Package / Tool": "CD-HIT",
            "Version / Source": "Fu et al. (2012)",
            "Primary Function in Pipeline": "Sequence redundancy reduction at 80% identity threshold",
        },
        {
            "Category": "Bioinformatics",
            "Package / Tool": "Infernal (cmscan)",
            "Version / Source": "Nawrocki & Eddy (2013)",
            "Primary Function in Pipeline": "Rfam covariance model search and negative decontamination",
        },
        {
            "Category": "Bioinformatics",
            "Package / Tool": "BLAST+ (blastn)",
            "Version / Source": "Camacho et al. (2009)",
            "Primary Function in Pipeline": "Nucleotide local alignment for EVA sequence novelty filter",
        },
        {
            "Category": "Bioinformatics",
            "Package / Tool": "HMMER (nhmmer)",
            "Version / Source": "Wheeler & Eddy (2013)",
            "Primary Function in Pipeline": "Profile HMM sequence alignment against Rfam database",
        },
        {
            "Category": "Biophysics",
            "Package / Tool": "ViennaRNA",
            "Version / Source": "2.0 (Lorenz et al., 2011)",
            "Primary Function in Pipeline": "Static MFE, Z-score, and 30–60°C partition function sweeps",
        },
        {
            "Category": "Biophysics",
            "Package / Tool": "NUPACK",
            "Version / Source": "Fornace et al. (2025)",
            "Primary Function in Pipeline": "Static 37°C ensemble free energy and pairing matrices",
        },
        {
            "Category": "Generative Model",
            "Package / Tool": "EVA (GENTEL-Lab)",
            "Version / Source": "1.4B CLM (Huang et al., 2026)",
            "Primary Function in Pipeline": "Frozen autoregressive de novo 5′ UTR generation",
        },
        {
            "Category": "Generative Model",
            "Package / Tool": "PyTorch / CUDA",
            "Version / Source": "Official framework",
            "Primary Function in Pipeline": "GPU tensor acceleration for EVA inference",
        },
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "Category",
            "Package / Tool",
            "Version / Source",
            "Primary Function in Pipeline",
        ],
    )


SOFTWARE_TABLE_PART_A: tuple[str, ...] = (
    "Runtime & Data",
    "Machine Learning",
    "Bioinformatics",
)
SOFTWARE_TABLE_PART_B: tuple[str, ...] = (
    "Biophysics",
    "Generative Model",
)
DOCS_TABLE_VWIDTH = 1200
DOCS_TABLE_ZOOM = 4.0
DOCS_TABLE_TARGET_WIDTH = 4592
DOCS_TABLE_DPI = 96
# Tall tables: lower zoom keeps Docs imports under pixel / size limits.
DOCS_TABLE_TALL_ZOOM = 3.0
DOCS_TABLE_TALL_TARGET_WIDTH = 3444


def build_thesis_software_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split thesis software inventory into two Docs-sized tables."""
    full = build_thesis_software_table()
    part_a = full[full["Category"].isin(SOFTWARE_TABLE_PART_A)].reset_index(drop=True)
    part_b = full[full["Category"].isin(SOFTWARE_TABLE_PART_B)].reset_index(drop=True)
    return part_a, part_b


def optimize_docs_png(path: Path, *, dpi: int = DOCS_TABLE_DPI) -> Path:
    """Re-save PNG with compression to stay under Docs import limits."""
    from PIL import Image

    path = Path(path)
    image = Image.open(path).convert("RGB")
    image.save(path, format="PNG", optimize=True, compress_level=9)
    stamp_png_dpi(path, dpi=dpi)
    return path


def export_docs_gt_png(
    table: _GTSaveTable,
    path: Path,
    *,
    zoom: float = DOCS_TABLE_ZOOM,
    target_width: int = DOCS_TABLE_TARGET_WIDTH,
) -> Path:
    """Save a great-tables GT object as a Docs-ready PNG (96 DPI, compressed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.gtsave(
        str(path),
        zoom=zoom,
        vwidth=DOCS_TABLE_VWIDTH,
        expand=24,
    )
    fit_docs_png_width(path, target_width=target_width, dpi=DOCS_TABLE_DPI)
    optimize_docs_png(path, dpi=DOCS_TABLE_DPI)
    return path


def save_fig(fig: Figure, path: Path) -> Path:
    """Save a figure at publication DPI."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    return path


__all__ = [
    "RF_ENSEMBLE_CV_MERMAID",
    "RF_ENSEMBLE_CV_MERMAID_V2",
    "build_thesis_software_table",
    "build_thesis_software_tables",
    "build_thermo_benchmark_tables",
    "docs_table_options",
    "export_docs_gt_png",
    "export_rf_grouped_cv_png",
    "export_rf_grouped_cv_v2_png",
    "export_workflow_architecture_png",
    "fit_docs_png_width",
    "load_json",
    "optimize_docs_png",
    "stamp_png_dpi",
    "plot_checklist_upset",
    "plot_cv_collapse_roc",
    "plot_eva_attrition_funnel",
    "plot_grouped_permutation_importance",
    "plot_hill_melting_overlay",
    "plot_naive_vs_strict_attribution",
    "plot_oof_confidence_hist",
    "plot_thermoswitch_mechanism",
    "plot_tm_concordance",
    "rfam_hill_param_table",
    "save_fig",
]
