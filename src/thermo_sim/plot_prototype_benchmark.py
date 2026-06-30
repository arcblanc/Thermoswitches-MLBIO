import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import PROJECT_ROOT, resolve_path
from thermo_sim.thermo_common import fit_hill_curve, hill_sigmoid

DEFAULT_REPORT = "data/processed/prototype/benchmark_report.json"
DEFAULT_FUSED = "data/processed/prototype/fused_features.csv"
OUTPUT_DIR = "reports/figures/prototype"

ROLE_LABELS = {
    "canonical_positive": "FourU",
    "anomaly_cspA": "cspA-425",
    "short_negative": "Guanidine-II",
    "stress_512nt": "cspA-512",
}

FEATURE_PAIRS = [
    ("Tm", "viennarna_Tm", "nupack_Tm"),
    ("Hill coefficient", "viennarna_hill_coeff", "nupack_hill_coeff"),
    ("Amplitude", "viennarna_amplitude", "nupack_amplitude"),
    ("Exposure / unpaired", "viennarna_mean_unpaired_prob", "nupack_mean_exposure"),
    ("MFE", None, "nupack_MFE"),
    ("Max stem length", None, "nupack_max_stem_length"),
]


def load_benchmark_report(path=DEFAULT_REPORT):
    with resolve_path(path).open() as handle:
        return json.load(handle)


def load_fused_features(path=DEFAULT_FUSED):
    return pd.read_csv(resolve_path(path))


def ensure_output_dir(output_dir=OUTPUT_DIR):
    output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_figure(fig, output_dir, name):
    path = output_dir / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _sequence_rows(report):
    rows = []
    runtime = report.get("runtime_sec")
    n_seq = len(report.get("sequences", [])) or 1
    for seq in report.get("sequences", []):
        vienna_mem = seq["vienna"]["memory_mb"]["rss_after_mb"]
        nupack_mem = seq.get("nupack", {}).get("memory_mb", {}).get("rss_after_mb", vienna_mem)
        elapsed = seq.get("elapsed_sec") or {}
        total_time = elapsed.get("total")
        if total_time is None and runtime is not None:
            total_time = runtime / n_seq
        rows.append(
            {
                "panel_role": seq["panel_role"],
                "rfam_id": seq["rfam_id"],
                "label": ROLE_LABELS.get(seq["panel_role"], seq["rfam_id"]),
                "seq_length": seq["seq_length"],
                "peak_rss_mb": max(vienna_mem, nupack_mem),
                "elapsed_sec": total_time,
            }
        )
    return pd.DataFrame(rows)


def plot_length_vs_cost(report, output_dir):
    df = _sequence_rows(report)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_col, ylabel in [
        (axes[0], "peak_rss_mb", "Peak RSS (MB)"),
        (axes[1], "elapsed_sec", "Wall time (s)"),
    ]:
        for role, group in df.groupby("panel_role"):
            ax.scatter(
                group["seq_length"],
                group[y_col],
                label=ROLE_LABELS.get(role, role),
                s=80,
            )
        lengths = np.array(sorted(df["seq_length"].unique()), dtype=float)
        if len(lengths) >= 2:
            anchor = df.loc[df["seq_length"].idxmin()]
            ref_y = anchor[y_col]
            ref_x = anchor["seq_length"]
            guide = ref_y * (lengths / ref_x) ** 3
            ax.plot(lengths, guide, "--", color="gray", alpha=0.6, label="O(N³) guide")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sequence length (nt)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        for _, row in df.iterrows():
            ax.annotate(row["label"], (row["seq_length"], row[y_col]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    memory = report.get("memory", {})
    estimate = memory.get("estimated_ram_gb_full_run")
    if estimate is not None:
        fig.text(
            0.5,
            0.01,
            f"GCE RAM estimate (naive): {estimate:.1f} GB for full 2,396-sequence run",
            ha="center",
            fontsize=9,
        )

    fig.suptitle("Sequence length vs computational cost", fontsize=13)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return save_figure(fig, output_dir, "length_vs_cost_scatter.png")


def plot_categorical_bars(report, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cpu = report.get("cpu", {})

    cpu_labels = ["Mean CPU %", "Peak CPU %"]
    cpu_values = [cpu.get("mean_cpu_pct"), cpu.get("peak_cpu_pct")]
    axes[0].barh(cpu_labels, cpu_values, color=["#4C72B0", "#DD8452"])
    axes[0].set_xlabel("CPU utilization (%)")
    axes[0].set_title("Thread benchmark")

    dangle_cmp = report.get("biophysics", {}).get("dangle_comparison", {})
    chosen = report.get("biophysics", {}).get("chosen_vienna_dangles")
    if dangle_cmp:
        labels = []
        values = []
        for key in sorted(dangle_cmp, key=lambda x: int(x)):
            label = f"ViennaRNA -d{key}"
            if str(chosen) == str(key):
                label += " *"
            labels.append(label)
            values.append(dangle_cmp[key].get("rmse"))
        axes[1].barh(labels, values, color=["#55A868", "#C44E52"])
        axes[1].set_xlabel("Hill-fit RMSE (FourU SD-window curve)")
        axes[1].set_title("Dangling-end model comparison")
    else:
        axes[1].text(0.5, 0.5, "dangle_comparison not in report", ha="center", va="center")
        axes[1].set_axis_off()

    fig.suptitle("Categorical performance metrics", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, output_dir, "categorical_bars.png")


def plot_feature_violins(fused_df, output_dir):
    records = []
    for feature_name, vienna_col, nupack_col in FEATURE_PAIRS:
        for _, row in fused_df.iterrows():
            if vienna_col and vienna_col in fused_df.columns:
                records.append(
                    {
                        "feature": feature_name,
                        "engine": "ViennaRNA",
                        "panel_role": row["panel_role"],
                        "value": row[vienna_col],
                    }
                )
            if nupack_col and nupack_col in fused_df.columns:
                records.append(
                    {
                        "feature": feature_name,
                        "engine": "NUPACK",
                        "panel_role": row["panel_role"],
                        "value": row[nupack_col],
                    }
                )

    long_df = pd.DataFrame(records)
    long_df["label"] = long_df["panel_role"].map(ROLE_LABELS)

    n_features = len(FEATURE_PAIRS)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax, feature_name in zip(axes, [f[0] for f in FEATURE_PAIRS]):
        subset = long_df[long_df["feature"] == feature_name].copy()
        if subset.empty:
            ax.set_axis_off()
            continue
        sns.violinplot(data=subset, x="engine", y="value", hue="label", ax=ax, inner="box", cut=0)
        ax.set_title(feature_name)
        ax.set_xlabel("")
        ax.set_ylabel("")

    fig.suptitle("Feature distributions across prototype sequences", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_figure(fig, output_dir, "feature_distribution_violin.png")


def _hill_overlay(temps, values):
    fit = fit_hill_curve(temps, values)
    dense_t = np.linspace(min(temps), max(temps), 200)
    if fit["fit_status"] == "ok" and fit["bottom"] is not None:
        curve = hill_sigmoid(dense_t, fit["bottom"], fit["top"], fit["Tm"], fit["hill_coeff"])
    else:
        curve = np.full_like(dense_t, np.nan)
    return fit, dense_t, curve


def plot_melting_profiles(report, output_dir):
    sequences = report.get("sequences", [])
    n_rows = len(sequences)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows), squeeze=False)

    for row_idx, seq in enumerate(sequences):
        for col_idx, engine in enumerate(("vienna", "nupack")):
            ax = axes[row_idx][col_idx]
            block = seq.get(engine)
            if not block:
                ax.set_axis_off()
                continue
            curves = block.get("curves") or {}
            temps = curves.get("temps") or report.get("temp_range", [])
            values = curves.get("hill_curve") or curves.get("exposure_curve") or []
            features = block.get("features", {})
            if engine == "vienna":
                fit_status = features.get("viennarna_fit_status")
                tm = features.get("viennarna_Tm")
            else:
                fit_status = features.get("nupack_fit_status")
                tm = features.get("nupack_Tm")

            ax.scatter(temps, values, s=20, alpha=0.8, label="SD-window / exposure")
            if values:
                fit, dense_t, curve = _hill_overlay(temps, values)
                if np.all(np.isfinite(curve)):
                    ax.plot(dense_t, curve, color="crimson", linewidth=2, label="Hill fit")
                tm = fit.get("Tm", tm)
                fit_status = fit.get("fit_status", fit_status)

            engine_name = "ViennaRNA" if engine == "vienna" else "NUPACK"
            title = f"{seq['rfam_id']} ({ROLE_LABELS.get(seq['panel_role'], seq['panel_role'])}) — {engine_name}\n"
            if tm is not None:
                title += f"fit: {fit_status}, Tm={tm:.1f}"
            else:
                title += f"fit: {fit_status}"
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Temperature (°C)")
            ax.set_ylabel("Unpaired / exposure probability")
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    fig.suptitle("Melting profiles: scatter + Generalised Hill overlay", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_figure(fig, output_dir, "melting_profiles_hill_overlay.png")


def generate_all_figures(report_path=DEFAULT_REPORT, fused_path=DEFAULT_FUSED, output_dir=OUTPUT_DIR):
    sns.set_theme(style="whitegrid")
    output_dir = ensure_output_dir(output_dir)
    report = load_benchmark_report(report_path)
    fused_df = load_fused_features(fused_path)

    paths = [
        plot_length_vs_cost(report, output_dir),
        plot_categorical_bars(report, output_dir),
        plot_feature_violins(fused_df, output_dir),
        plot_melting_profiles(report, output_dir),
    ]
    return paths


def _build_parser():
    parser = argparse.ArgumentParser(description="Generate prototype benchmark visualization figures.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--fused", default=DEFAULT_FUSED)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    return parser


def main():
    args = _build_parser().parse_args()
    paths = generate_all_figures(
        report_path=args.report,
        fused_path=args.fused,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
