"""EVA de novo biophysical characterization against Rfam / RefSeq controls.

Compares the 105 yield-gated EVA passers (31 pilot + 74 stream) to the labelled
panel using Hill melting parameters, four visual-checklist gates, and
Vienna–NUPACK concordance. Full 1 °C temperature sweeps are cached; notebooks
reconstruct control ribbons from fitted Hill parameters so the 2,396-row panel
is not re-folded interactively.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence, cast
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import ks_2samp, mannwhitneyu, spearmanr

from data_engineering.paths import PROJECT_ROOT, resolve_path
from thermo_sim.rf_posthoc import (
    HILL_SNAP,
    TM_MAX,
    TM_MIN,
    gate_amplitude,
    gate_baseline_repression,
    gate_hill,
    gate_tm,
)
from thermo_sim.thermo_common import (
    build_temp_range,
    fit_hill_curve,
    hill_sigmoid,
    sd_window_indices,
)

COHORT_EVA_PILOT = "EVA Pilot Top Passers"
COHORT_EVA_STREAM = "EVA Stream Top Passers"
COHORT_RFAM = "Natural Rfam Positives"
COHORT_REFSEQ = "RefSeq Negative Controls"
COHORT_PROTOTYPE = "Gold-Standard Prototypes"

SWEEP_TEMP_MIN = 30
SWEEP_TEMP_MAX = 60
SWEEP_TEMP_STEP = 1
IDEAL_TM_C = 43.5
LEAKY_POPEN_MAX = 0.15
IDEAL_DELTA_P = 0.60
FRAGILE_DELTA_TM = 5.0
CONCORDANT_DELTA_TM = 3.0
HILL_R2_MIN = 0.95
OPEN_COIL_HILL_MAX = 1.0

# Tier 1 — biological plausibility (wide heat-shock window).
TIER1_HILL = 1.2
TIER1_TM_LO = 35.0
TIER1_TM_HI = 55.0
TIER1_AMP_MIN = 0.20
TIER1_POPEN_MAX = 0.35

# Tier 2 — Neupert 2008 / synbio heat-inducible RBS-unmasking spec.
TIER2_HILL = 1.5
TIER2_TM_LO = 42.0
TIER2_TM_HI = 45.0
TIER2_AMP_MIN = 0.40  # gate floor; ideal stroke approaches 0.50
TIER2_AMP_IDEAL = 0.50
TIER2_POPEN_MAX = 0.20

SCORE_WEIGHTS: dict[str, float] = {
    "w_hill": 1.0,
    "w_amp": 2.0,  # reward dynamic stroke |Δθ|
    "w_tm": 0.4,
    "w_leak": 6.0,  # prioritize basal lock (Tier-2 P_open ≤ 0.20)
    "w_engine": 0.25,
    "w_stroke_penalty": 12.0,  # severe drop for near-zero amplitude
    "n_h_cap": 3.0,
    "amp_floor": 0.25,
}

RESTRICTION_SITES: dict[str, str] = {
    "EcoRI": "GAATTC",
    "BamHI": "GGATCC",
    "XhoI": "CTCGAG",
}

COHORT_COLORS: dict[str, str] = {
    COHORT_EVA_STREAM: "#1B9E77",
    COHORT_EVA_PILOT: "#66A61E",
    COHORT_RFAM: "#7570B3",
    COHORT_REFSEQ: "#D95F02",
    COHORT_PROTOTYPE: "#E7298A",
}

DEFAULT_FUSED = "data/processed/fused_features_refseq_dynamic.csv"
DEFAULT_PILOT_FASTA = "data/processed/eva_pilot/top_candidates.fasta"
DEFAULT_STREAM_FASTA = "data/processed/eva_stream/top_candidates.fasta"
DEFAULT_PILOT_DYNAMIC = "data/processed/eva_pilot/dynamic_features.csv"
DEFAULT_PILOT_YIELD = "data/processed/eva_pilot/yield_ratio_sequences.csv"
DEFAULT_PILOT_NOVELTY = "data/processed/novelty/eva_pilot_novelty_report.csv"
DEFAULT_STREAM_MASTER = "data/processed/eva_stream/dynamic_features_master.csv"
DEFAULT_CACHE_DIR = "data/processed/eva_characterization"
DEFAULT_SWEEP_CACHE = f"{DEFAULT_CACHE_DIR}/eva_temp_sweeps.csv"
DEFAULT_HILL_CACHE = f"{DEFAULT_CACHE_DIR}/eva_hill_fits.csv"
DEFAULT_CHECKLIST_JSON = "data/processed/eva_denovo_checklist.json"
DEFAULT_LEADS_FASTA = "data/processed/leads/eva_top10_experimental_leads.fasta"
DEFAULT_FIG_DIR = "notebooks/figures/08_eva_denovo"


def resolve_project_root(start: Path | None = None) -> Path:
    """Return the repo root containing ``src/``, walking up from *start*."""
    root = (start or Path.cwd()).resolve()
    if not (root / "src").exists() and (root.parent / "src").exists():
        root = root.parent
    return root


def characterization_paths(root: Path | None = None) -> dict[str, Path]:
    """Return resolved default artifact paths for the characterization suite."""
    base = root or PROJECT_ROOT
    keys = {
        "fused": DEFAULT_FUSED,
        "pilot_fasta": DEFAULT_PILOT_FASTA,
        "stream_fasta": DEFAULT_STREAM_FASTA,
        "pilot_dynamic": DEFAULT_PILOT_DYNAMIC,
        "pilot_yield": DEFAULT_PILOT_YIELD,
        "pilot_novelty": DEFAULT_PILOT_NOVELTY,
        "stream_master": DEFAULT_STREAM_MASTER,
        "cache_dir": DEFAULT_CACHE_DIR,
        "sweep_cache": DEFAULT_SWEEP_CACHE,
        "hill_cache": DEFAULT_HILL_CACHE,
        "checklist_json": DEFAULT_CHECKLIST_JSON,
        "leads_fasta": DEFAULT_LEADS_FASTA,
        "fig_dir": DEFAULT_FIG_DIR,
        "prototype_csv": "data/processed/prototype/prototype_panel.csv",
        "prototype_fasta": "data/processed/prototype/prototype_panel.fasta",
        "prototype_curves": "data/processed/prototype/curves",
        "posthoc_json": "data/processed/rf_posthoc_report.json",
    }
    return {name: base / rel for name, rel in keys.items()}


def iter_fasta_headers(fasta_path: Path) -> list[tuple[str, str]]:
    """Return (full header, sequence) pairs without truncating annotation fields."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    with fasta_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header = line[1:]
                parts = []
            else:
                parts.append(line)
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def load_fasta_table(fasta_path: Path, *, cohort: str) -> pd.DataFrame:
    """Load FASTA records into a table with optional header key=value fields."""
    rows: list[dict[str, object]] = []
    if not fasta_path.exists():
        return pd.DataFrame()
    for header, sequence in iter_fasta_headers(fasta_path):
        record_id = header.split()[0]
        fields = _parse_header_fields(header)
        seq = sequence.replace("T", "U").upper()
        rows.append(
            {
                "record_id": record_id,
                "sequence": seq,
                "seq_length": len(seq),
                "cohort": cohort,
                **fields,
            }
        )
    return pd.DataFrame(rows)


def _parse_header_fields(header: str) -> dict[str, float]:
    """Parse numeric ``key=value`` annotations from a FASTA header line."""
    out: dict[str, float] = {}
    for match in re.finditer(r"([A-Za-z_]+)=([-+0-9.eE]+)", header):
        key = match.group(1)
        try:
            value = float(match.group(2))
        except ValueError:
            continue
        if key.upper() in {"Z"}:
            out["viennarna_mfe_zscore"] = value
        elif key in {"dP_RBS", "delta_P_RBS", "dP"}:
            out["viennarna_delta_P_RBS"] = value
    return out


def load_eva_passers(paths: dict[str, Path] | None = None) -> pd.DataFrame:
    """Load the 31 pilot + 74 stream yield-gated FASTA passers."""
    paths = paths or characterization_paths()
    frames = [
        load_fasta_table(paths["pilot_fasta"], cohort=COHORT_EVA_PILOT),
        load_fasta_table(paths["stream_fasta"], cohort=COHORT_EVA_STREAM),
    ]
    eva = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if eva.empty:
        return eva
    physics = _load_eva_physics_tables(paths)
    if not physics.empty:
        eva = eva.merge(physics, on="record_id", how="left", suffixes=("", "_dyn"))
        for col in (
            "viennarna_mfe_zscore",
            "viennarna_delta_P_RBS",
            "viennarna_P_open_RBS_37",
            "viennarna_P_open_RBS_55",
            "E_Rfam",
        ):
            dyn = f"{col}_dyn"
            if dyn in eva.columns:
                eva[col] = (
                    eva[col].combine_first(eva[dyn]) if col in eva.columns else eva[dyn]
                )
                eva = eva.drop(columns=[dyn])
    novelty = _load_novelty_tables(paths)
    if not novelty.empty:
        eva = eva.merge(novelty, on="record_id", how="left")
    eva["E_Rfam"] = _coalesce_erfam(eva)
    drop_e = [c for c in eva.columns if c.startswith("E_Rfam_") or c == "best_evalue"]
    eva = eva.drop(columns=[c for c in drop_e if c in eva.columns])
    eva["label"] = 1
    return eva


def _coalesce_erfam(frame: pd.DataFrame) -> pd.Series:
    """Return the tightest finite Rfam E-value across duplicate merge columns."""
    cols = [
        c
        for c in frame.columns
        if c == "E_Rfam" or c.startswith("E_Rfam_") or c == "best_evalue"
    ]
    values = np.full(len(frame), np.inf)
    for col in cols:
        values = np.minimum(values, frame[col].map(_as_erfam).to_numpy())
    return pd.Series(values, index=frame.index, dtype=float)


def _load_eva_physics_tables(paths: dict[str, Path]) -> pd.DataFrame:
    """Concatenate pilot/stream dynamic and yield tables on record_id."""
    chunks: list[pd.DataFrame] = []
    for key in ("pilot_dynamic", "pilot_yield", "stream_master"):
        path = paths[key]
        if path.exists():
            chunks.append(pd.read_csv(path))
    stream_root = paths["stream_master"].parent
    for csv_path in sorted(stream_root.glob("slice_*_yield_sequences.csv")):
        chunks.append(pd.read_csv(csv_path))
    if not chunks:
        return pd.DataFrame()
    keep_cols = [
        "record_id",
        "viennarna_mfe_zscore",
        "viennarna_delta_P_RBS",
        "viennarna_P_open_RBS_37",
        "viennarna_P_open_RBS_55",
        "E_Rfam",
        "yield_pass",
    ]
    cleaned: list[pd.DataFrame] = []
    for frame in chunks:
        if "record_id" not in frame.columns:
            continue
        present = [c for c in keep_cols if c in frame.columns]
        cleaned.append(frame[present].drop_duplicates("record_id"))
    if not cleaned:
        return pd.DataFrame()
    merged = cleaned[0]
    for extra in cleaned[1:]:
        merged = merged.merge(extra, on="record_id", how="outer", suffixes=("", "_dup"))
        for col in list(merged.columns):
            if col.endswith("_dup"):
                base = col[: -len("_dup")]
                if base in merged.columns:
                    if base == "E_Rfam":
                        merged[base] = [
                            min(_as_erfam(a), _as_erfam(b))
                            for a, b in zip(merged[base], merged[col], strict=True)
                        ]
                    else:
                        merged[base] = merged[base].combine_first(merged[col])
                merged = merged.drop(columns=[col])
    return merged.drop_duplicates("record_id")


def _load_novelty_tables(paths: dict[str, Path]) -> pd.DataFrame:
    """Load Rfam search E-values / categories for EVA record IDs."""
    frames: list[pd.DataFrame] = []
    if paths["pilot_novelty"].exists():
        frames.append(pd.read_csv(paths["pilot_novelty"]))
    stream_nov = paths["stream_master"].parent / "novelty"
    if stream_nov.is_dir():
        for csv_path in sorted(stream_nov.glob("slice_*_novelty_report.csv")):
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame()
    nov = pd.concat(frames, ignore_index=True)
    if "record_id" not in nov.columns:
        return pd.DataFrame()
    keep = ["record_id"]
    for col in ("best_evalue", "novelty_category", "has_hit"):
        if col in nov.columns:
            keep.append(col)
    nov = nov[keep].drop_duplicates("record_id")
    if "best_evalue" in nov.columns:
        nov["E_Rfam"] = nov["best_evalue"].apply(_as_erfam)
    return nov


def _as_erfam(value: object) -> float:
    """Coerce a search E-value to float, using +inf for missing / no-hit."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("inf")
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "inf", "none"}:
        return float("inf")
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("inf")
    else:
        return float("inf")
    if not math.isfinite(number):
        return float("inf")
    return number


def load_control_panel(fused_csv: Path | None = None) -> pd.DataFrame:
    """Load Rfam positives and RefSeq negatives from the fused physics table."""
    path = fused_csv or resolve_path(DEFAULT_FUSED)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["cohort"] = np.where(
        frame["label"].astype(int) == 1, COHORT_RFAM, COHORT_REFSEQ
    )
    frame["record_id"] = (
        frame["rfamseq_acc"].astype(str)
        + ":"
        + frame["seq_start"].astype(str)
        + "-"
        + frame["seq_end"].astype(str)
    )
    if "E_Rfam" not in frame.columns:
        frame["E_Rfam"] = np.where(frame["label"].astype(int) == 1, 0.0, float("inf"))
    return frame


def load_prototype_overlay(paths: dict[str, Path] | None = None) -> pd.DataFrame:
    """Load FourU (and other prototype) Vienna Hill curves as gold-standard overlays."""
    paths = paths or characterization_paths()
    panel = (
        pd.read_csv(paths["prototype_csv"])
        if paths["prototype_csv"].exists()
        else pd.DataFrame()
    )
    fasta = load_fasta_table(paths["prototype_fasta"], cohort=COHORT_PROTOTYPE)
    if not fasta.empty and "rfam_id" in panel.columns:
        fasta = fasta.copy()
        fasta["rfam_id"] = panel["rfam_id"].to_numpy()[: len(fasta)]
        fasta["panel_role"] = panel.get("panel_role", pd.Series(dtype=str)).to_numpy()[
            : len(fasta)
        ]
    rows: list[dict[str, object]] = []
    curves_dir = paths["prototype_curves"]
    if not curves_dir.is_dir():
        fasta["cohort"] = COHORT_PROTOTYPE
        return fasta
    for json_path in sorted(curves_dir.glob("*_vienna.json")):
        if json_path.name.startswith("vienna_fourU"):
            continue
        payload = json.loads(json_path.read_text())
        stem = json_path.stem.replace("_vienna", "")
        rfam_id, _, role = stem.partition("_")
        temps = [float(t) for t in payload.get("temps", [])]
        values = payload.get("hill_curve") or payload.get("melting_curve") or []
        fit = fit_curve_with_r2(
            temps, [float(v) if v is not None else math.nan for v in values]
        )
        rows.append(
            {
                "record_id": stem,
                "rfam_id": rfam_id,
                "panel_role": role,
                "cohort": COHORT_PROTOTYPE,
                "temps": temps,
                "p_open_curve": [float(v) for v in values],
                "viennarna_Tm": fit.get("Tm"),
                "viennarna_hill_coeff": fit.get("hill_coeff"),
                "viennarna_amplitude": fit.get("amplitude"),
                "viennarna_hill_bottom": fit.get("bottom"),
                "viennarna_hill_top": fit.get("top"),
                "hill_r2": fit.get("r2"),
                "viennarna_fit_status": fit.get("fit_status"),
            }
        )
    overlay = pd.DataFrame(rows)
    if overlay.empty:
        return fasta
    if not fasta.empty:
        overlay = overlay.merge(
            fasta[["record_id", "sequence", "seq_length"]],
            on="record_id",
            how="left",
        )
    return overlay


def fit_curve_with_r2(
    temps: list[float] | np.ndarray,
    values: list[float] | np.ndarray,
) -> dict[str, float | str | None]:
    """Fit a Hill sigmoid and attach R² plus a convergence flag."""
    fit = dict(fit_hill_curve(list(temps), list(values)))
    temps_arr = np.asarray(temps, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    tm = fit.get("Tm")
    n_h = fit.get("hill_coeff")
    bottom = fit.get("bottom")
    top = fit.get("top")
    if (
        tm is None
        or n_h is None
        or bottom is None
        or top is None
        or not np.all(np.isfinite(values_arr))
    ):
        fit["r2"] = None
        fit["fit_converged"] = False
        return fit
    fitted = hill_sigmoid(temps_arr, float(bottom), float(top), float(tm), float(n_h))
    ss_res = float(np.sum((values_arr - fitted) ** 2))
    ss_tot = float(np.sum((values_arr - np.mean(values_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    fit["r2"] = r2
    fit["fit_converged"] = bool(fit.get("fit_status") == "ok" and r2 >= HILL_R2_MIN)
    return fit


def reconstruct_theta(
    row: pd.Series,
    temps: np.ndarray,
    *,
    prefix: str = "viennarna",
) -> np.ndarray:
    """Reconstruct θ(T) from fitted Hill parameters, anchored at P_open(37) when present."""
    tm = row.get(f"{prefix}_Tm")
    n_h = row.get(f"{prefix}_hill_coeff")
    amp = row.get(f"{prefix}_amplitude")
    if pd.isna(tm) or pd.isna(n_h) or pd.isna(amp):
        return np.full(temps.shape, np.nan)
    frac = hill_sigmoid(temps, 0.0, 1.0, float(tm), float(n_h))
    p37 = row.get("viennarna_P_open_RBS_37") if prefix == "viennarna" else None
    bottom = row.get(f"{prefix}_hill_bottom")
    if pd.notna(bottom):
        base = float(bottom)
    elif pd.notna(p37) and isinstance(p37, (int, float)):
        p37_val = float(p37)
        f37 = float(hill_sigmoid(np.array([37.0]), 0.0, 1.0, float(tm), float(n_h))[0])
        base = p37_val - float(amp) * f37
    else:
        base = 0.0
    theta = base + float(amp) * frac
    return np.clip(theta, 0.0, 1.0)


def ribbon_from_hill_params(
    frame: pd.DataFrame,
    temps: np.ndarray,
    *,
    prefix: str = "viennarna",
) -> pd.DataFrame:
    """Grand-mean θ(T) with IQR and 95% mean CI from reconstructed Hill curves."""
    rows: list[dict[str, object]] = []
    for cohort, sub in frame.groupby("cohort", sort=False):
        matrix = np.vstack(
            [reconstruct_theta(row, temps, prefix=prefix) for _, row in sub.iterrows()]
        )
        finite = np.isfinite(matrix).all(axis=1)
        matrix = matrix[finite]
        if matrix.size == 0:
            continue
        mean = matrix.mean(axis=0)
        lo, hi = np.percentile(matrix, [25, 75], axis=0)
        n = matrix.shape[0]
        se = matrix.std(axis=0, ddof=1) / math.sqrt(n) if n > 1 else np.zeros_like(mean)
        rows.extend(
            {
                "cohort": cohort,
                "temp_C": float(temp),
                "mean": float(mu),
                "iqr_lo": float(a),
                "iqr_hi": float(b),
                "ci95_lo": float(mu - 1.96 * s),
                "ci95_hi": float(mu + 1.96 * s),
                "n": int(n),
            }
            for temp, mu, a, b, s in zip(temps, mean, lo, hi, se, strict=True)
        )
    return pd.DataFrame(rows)


def ribbon_from_sweep_table(
    sweeps: pd.DataFrame, *, value_col: str = "p_open"
) -> pd.DataFrame:
    """Grand-mean ribbon statistics from long-form temperature-sweep rows."""
    rows: list[dict[str, object]] = []
    for (cohort, temp), sub in cast(
        Any, sweeps.groupby(["cohort", "temp_C"], sort=True)
    ):
        values = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy()
        if values.size == 0:
            continue
        n = int(values.size)
        mean = float(values.mean())
        se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        lo, hi = np.percentile(values, [25, 75])
        rows.append(
            {
                "cohort": cohort,
                "temp_C": float(temp),
                "mean": mean,
                "iqr_lo": float(lo),
                "iqr_hi": float(hi),
                "ci95_lo": mean - 1.96 * se,
                "ci95_hi": mean + 1.96 * se,
                "n": n,
            }
        )
    return pd.DataFrame(rows)


def sweep_temps(
    temp_min: int = SWEEP_TEMP_MIN,
    temp_max: int = SWEEP_TEMP_MAX,
    temp_step: int = SWEEP_TEMP_STEP,
) -> list[int]:
    """Return the inclusive Celsius grid used for EVA temperature sweeps."""
    return build_temp_range(temp_min, temp_max, temp_step)


def run_or_load_eva_sweeps(
    eva: pd.DataFrame,
    *,
    cache_path: Path | None = None,
    run: bool = False,
    engine: str = "vienna",
    temp_min: int = SWEEP_TEMP_MIN,
    temp_max: int = SWEEP_TEMP_MAX,
    temp_step: int = SWEEP_TEMP_STEP,
) -> pd.DataFrame:
    """Load cached EVA P_open(T) sweeps, or compute them when *run* is True."""
    cache = cache_path or resolve_path(DEFAULT_SWEEP_CACHE)
    if cache.exists() and not run:
        cached = pd.read_csv(cache)
        wanted = set(eva["record_id"].astype(str))
        if wanted.issubset(set(cached["record_id"].astype(str))):
            return cached[cached["record_id"].astype(str).isin(wanted)].copy()
    if not run:
        return pd.DataFrame()
    temps = sweep_temps(temp_min, temp_max, temp_step)
    records: list[dict[str, object]] = []
    for _, row in eva.iterrows():
        sequence = str(row.get("sequence") or "")
        if not sequence:
            continue
        vienna_curve = (
            _vienna_hill_curve(sequence, temps)
            if engine in {"vienna", "both"}
            else None
        )
        nupack_curve = (
            _nupack_exposure_curve(sequence, temps)
            if engine in {"nupack", "both"}
            else None
        )
        for i, temp in enumerate(temps):
            records.append(
                {
                    "record_id": row["record_id"],
                    "cohort": row["cohort"],
                    "temp_C": int(temp),
                    "p_open": None if vienna_curve is None else vienna_curve[i],
                    "nupack_exposure": None
                    if nupack_curve is None
                    else nupack_curve[i],
                }
            )
    sweeps = pd.DataFrame(records)
    cache.parent.mkdir(parents=True, exist_ok=True)
    sweeps.to_csv(cache, index=False)
    return sweeps


def _vienna_hill_curve(sequence: str, temps: list[int]) -> list[float | None]:
    """Return the SD-window unpaired series used by ``extract_vienna_features``."""
    from thermo_sim.vienna_rna import run_vienna_worker

    result = run_vienna_worker(
        {
            "sequence": sequence,
            "rfamseq_acc": "eva",
            "seq_start": 0,
            "seq_end": len(sequence),
        },
        temps,
        2,
    )
    curve = (result.get("_curves") or {}).get("hill_curve") or []
    return [None if v is None else float(v) for v in curve]


def _nupack_exposure_curve(sequence: str, temps: list[int]) -> list[float | None]:
    """Return the NUPACK exposure series, or empty if NUPACK is not installed."""
    try:
        from thermo_sim.nupack_engine import run_nupack_worker
    except (ImportError, OSError):
        return [None] * len(temps)
    try:
        result = run_nupack_worker(
            {"sequence": sequence},
            temps,
            0.05,
            0.0,
            1,
            1e-8,
        )
    except (ImportError, OSError, RuntimeError):
        return [None] * len(temps)
    curve = (result.get("_curves") or {}).get("exposure_curve") or []
    return [None if v is None else float(v) for v in curve]


def hill_fits_from_sweeps(sweeps: pd.DataFrame) -> pd.DataFrame:
    """Fit Hill parameters (Vienna and NUPACK) for each EVA record in *sweeps*."""
    rows: list[dict[str, object]] = []
    if sweeps.empty:
        return pd.DataFrame()
    for record_id, sub in sweeps.groupby("record_id", sort=False):
        ordered = sub.sort_values("temp_C")
        temps = ordered["temp_C"].astype(float).tolist()
        vienna = fit_curve_with_r2(temps, ordered["p_open"].tolist())
        nupack_vals = (
            ordered["nupack_exposure"].tolist() if "nupack_exposure" in ordered else []
        )
        nupack = (
            fit_curve_with_r2(temps, nupack_vals)
            if nupack_vals and any(v is not None and pd.notna(v) for v in nupack_vals)
            else {}
        )
        p37 = _interp_at(ordered, "p_open", 37.0)
        p55 = _interp_at(ordered, "p_open", 55.0)
        rows.append(
            {
                "record_id": record_id,
                "cohort": sub["cohort"].iloc[0],
                "viennarna_Tm": vienna.get("Tm"),
                "viennarna_hill_coeff": vienna.get("hill_coeff"),
                "viennarna_amplitude": vienna.get("amplitude"),
                "viennarna_hill_bottom": vienna.get("bottom"),
                "viennarna_hill_top": vienna.get("top"),
                "viennarna_fit_status": vienna.get("fit_status"),
                "hill_r2": vienna.get("r2"),
                "fit_converged": vienna.get("fit_converged"),
                "nupack_Tm": nupack.get("Tm"),
                "nupack_hill_coeff": nupack.get("hill_coeff"),
                "nupack_amplitude": nupack.get("amplitude"),
                "nupack_fit_status": nupack.get("fit_status"),
                "viennarna_P_open_RBS_37": p37,
                "viennarna_P_open_RBS_55": p55,
            }
        )
    return pd.DataFrame(rows)


def _interp_at(frame: pd.DataFrame, column: str, temp_c: float) -> float | None:
    """Linearly interpolate a sweep column at *temp_c*, or None if missing."""
    if column not in frame.columns:
        return None
    ordered = frame.sort_values("temp_C")
    temps = ordered["temp_C"].astype(float).to_numpy()
    values = pd.to_numeric(ordered[column], errors="coerce").to_numpy()
    if not np.any(np.isfinite(values)):
        return None
    mask = np.isfinite(values)
    return float(np.interp(temp_c, temps[mask], values[mask]))


def merge_eva_fits(eva: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Left-join Hill-fit columns onto the EVA passer table."""
    if fits.empty:
        return eva
    overlap = [c for c in fits.columns if c not in {"record_id", "cohort"}]
    return eva.merge(
        fits[["record_id", *overlap]], on="record_id", how="left", suffixes=("", "_fit")
    )


def backfill_eva_p_open(eva: pd.DataFrame, *, run: bool = True) -> pd.DataFrame:
    """Fill missing 37/55 °C RBS P_open via Vienna (no dinucleotide shuffles)."""
    if eva.empty or not run:
        return eva
    missing = (
        eva["viennarna_P_open_RBS_37"].isna()
        if "viennarna_P_open_RBS_37" in eva.columns
        else pd.Series(True, index=eva.index)
    )
    if not bool(missing.any()):
        return eva
    try:
        from thermo_sim.vienna_rna import extract_p_open_rbs
    except ImportError:
        return eva
    out = eva.copy()
    if "viennarna_P_open_RBS_37" not in out.columns:
        out["viennarna_P_open_RBS_37"] = np.nan
        out["viennarna_P_open_RBS_55"] = np.nan
    for idx in out.index[missing]:
        seq = str(out.at[idx, "sequence"])
        feats = extract_p_open_rbs({"sequence": seq})
        out.at[idx, "viennarna_P_open_RBS_37"] = feats.get("viennarna_P_open_RBS_37")
        out.at[idx, "viennarna_P_open_RBS_55"] = feats.get("viennarna_P_open_RBS_55")
    if "viennarna_delta_P_RBS" in out.columns:
        need_dp = out["viennarna_delta_P_RBS"].isna()
        out.loc[need_dp, "viennarna_delta_P_RBS"] = (
            out.loc[need_dp, "viennarna_P_open_RBS_55"]
            - out.loc[need_dp, "viennarna_P_open_RBS_37"]
        )
    return out


def _row_p_open_37(row: pd.Series) -> object:
    """Prefer measured P_open(37 °C); fall back to fitted Hill bottom."""
    p_open = row.get("viennarna_P_open_RBS_37")
    if pd.isna(p_open):
        p_open = row.get("viennarna_hill_bottom")
    return p_open


def _row_amplitude_stroke(row: pd.Series) -> object:
    """Return unsigned melting stroke |Δθ| when amplitude is defined."""
    amp = row.get("viennarna_amplitude")
    if pd.isna(amp):
        return amp
    return abs(float(amp))


def tier_gate_flags(row: pd.Series, *, tier: int = 2) -> dict[str, bool]:
    """Score one row against Tier 1 (plausibility) or Tier 2 (Neupert) gates.

    Tier 1 asks whether melting is biologically plausible in a wide heat-shock
    band. Tier 2 asks whether it matches Neupert-style heat-inducible RBS
    unmasking (tight Tm, larger stroke, locked 37 °C baseline).
    """
    if tier == 1:
        n_h_min, tm_lo, tm_hi, amp_min, p_max = (
            TIER1_HILL,
            TIER1_TM_LO,
            TIER1_TM_HI,
            TIER1_AMP_MIN,
            TIER1_POPEN_MAX,
        )
    elif tier == 2:
        n_h_min, tm_lo, tm_hi, amp_min, p_max = (
            TIER2_HILL,
            TIER2_TM_LO,
            TIER2_TM_HI,
            TIER2_AMP_MIN,
            TIER2_POPEN_MAX,
        )
    else:
        raise ValueError(f"tier must be 1 or 2, got {tier}")
    return {
        "gate_nh": gate_hill(row.get("viennarna_hill_coeff"), threshold=n_h_min),
        "gate_tm": gate_tm(row.get("viennarna_Tm"), lo=tm_lo, hi=tm_hi),
        "gate_amp": gate_amplitude(_row_amplitude_stroke(row), minimum=amp_min),
        "gate_base": gate_baseline_repression(_row_p_open_37(row), maximum=p_max),
    }


def four_gate_flags(row: pd.Series) -> dict[str, bool]:
    """Alias for Tier 2 (Neupert / synbio) four-gate checklist flags."""
    return tier_gate_flags(row, tier=2)


def _cohort_gate_counts(sub: pd.DataFrame, *, tier: int) -> dict[str, object]:
    """Aggregate per-gate and all-four pass counts for one cohort table."""
    flags = [tier_gate_flags(row, tier=tier) for _, row in sub.iterrows()]
    table = pd.DataFrame(flags) if flags else pd.DataFrame()
    prefix = f"t{tier}_"
    empty = {
        f"{prefix}passed_nh": 0,
        f"{prefix}passed_tm": 0,
        f"{prefix}passed_amp": 0,
        f"{prefix}passed_base": 0,
        f"{prefix}passed_all_four": 0,
        f"{prefix}frac_all_four": 0.0,
    }
    if table.empty:
        return empty
    all_four = table.all(axis=1)
    return {
        f"{prefix}passed_nh": int(table["gate_nh"].sum()),
        f"{prefix}passed_tm": int(table["gate_tm"].sum()),
        f"{prefix}passed_amp": int(table["gate_amp"].sum()),
        f"{prefix}passed_base": int(table["gate_base"].sum()),
        f"{prefix}passed_all_four": int(all_four.sum()),
        f"{prefix}frac_all_four": float(all_four.mean()),
    }


def checklist_by_cohort(frame: pd.DataFrame) -> pd.DataFrame:
    """Two-tier gate pass rates per cohort (Tier 1 plausibility + Tier 2 Neupert)."""
    rows: list[dict[str, object]] = []
    for cohort, sub in frame.groupby("cohort", sort=False):
        n = int(len(sub))
        row: dict[str, object] = {
            "cohort": cohort,
            "n": n,
            "frac_target_box": _target_box_fraction(sub),
        }
        t1 = _cohort_gate_counts(sub, tier=1)
        t2 = _cohort_gate_counts(sub, tier=2)
        row.update({k: v for k, v in t1.items() if k.startswith("t1_")})
        row.update({k: v for k, v in t2.items() if k.startswith("t2_")})
        # Backward-compatible Tier-2 aliases used by older notebook cells.
        row["passed_nh"] = row["t2_passed_nh"]
        row["passed_tm"] = row["t2_passed_tm"]
        row["passed_amp"] = row["t2_passed_amp"]
        row["passed_base"] = row["t2_passed_base"]
        row["passed_all_four"] = row["t2_passed_all_four"]
        row["frac_all_four"] = row["t2_frac_all_four"]
        rows.append(row)
    return pd.DataFrame(rows)


def _target_box_fraction(frame: pd.DataFrame) -> float | None:
    """Return the fraction of rows with Tier-2 Tm window and n_H > Tier-2 snap."""
    if frame.empty or "viennarna_Tm" not in frame.columns:
        return None
    tm = pd.to_numeric(frame["viennarna_Tm"], errors="coerce")
    n_h = pd.to_numeric(frame["viennarna_hill_coeff"], errors="coerce")
    inside = (tm >= TIER2_TM_LO) & (tm <= TIER2_TM_HI) & (n_h > TIER2_HILL)
    defined = tm.notna() & n_h.notna()
    if int(defined.sum()) == 0:
        return None
    return float(inside.sum() / defined.sum())


def two_sample_tests(
    left: pd.Series,
    right: pd.Series,
    *,
    feature: str,
    comparison: str,
) -> dict[str, object]:
    """Mann–Whitney U and KS tests between two numeric series."""
    a = pd.to_numeric(left, errors="coerce").dropna()
    b = pd.to_numeric(right, errors="coerce").dropna()
    if len(a) < 3 or len(b) < 3:
        return {
            "comparison": comparison,
            "feature": feature,
            "n_left": int(len(a)),
            "n_right": int(len(b)),
            "mannwhitney_u": None,
            "mannwhitney_p": None,
            "ks_statistic": None,
            "ks_p": None,
        }
    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
    ks_stat, ks_p = ks_2samp(a, b)
    return {
        "comparison": comparison,
        "feature": feature,
        "n_left": int(len(a)),
        "n_right": int(len(b)),
        "mannwhitney_u": float(u_stat),
        "mannwhitney_p": float(u_p),
        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_p),
    }


def cohort_stat_tests(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare EVA stream vs Rfam, RefSeq, and EVA pilot on melting scalars."""
    features = [
        "viennarna_hill_coeff",
        "viennarna_Tm",
        "viennarna_delta_P_RBS",
        "viennarna_mfe_zscore",
        "viennarna_amplitude",
        "viennarna_P_open_RBS_37",
    ]
    pairs = [
        (COHORT_EVA_STREAM, COHORT_RFAM, "stream_vs_rfam"),
        (COHORT_EVA_STREAM, COHORT_REFSEQ, "stream_vs_refseq"),
        (COHORT_EVA_STREAM, COHORT_EVA_PILOT, "stream_vs_pilot"),
    ]
    rows: list[dict[str, object]] = []
    for left_name, right_name, tag in pairs:
        left = frame[frame["cohort"] == left_name]
        right = frame[frame["cohort"] == right_name]
        for feature in features:
            if feature not in frame.columns:
                continue
            rows.append(
                two_sample_tests(
                    left[feature],
                    right[feature],
                    feature=feature,
                    comparison=tag,
                )
            )
    return pd.DataFrame(rows)


def engine_concordance(frame: pd.DataFrame) -> dict[str, Any]:
    """Spearman r_s and MAE between Vienna and NUPACK Tm / n_H."""
    out: dict[str, Any] = {}
    for v_col, n_col, name in (
        ("viennarna_Tm", "nupack_Tm", "Tm"),
        ("viennarna_hill_coeff", "nupack_hill_coeff", "hill"),
    ):
        if v_col not in frame.columns or n_col not in frame.columns:
            out[name] = {"n": 0, "r_s": None, "mae": None}
            continue
        # Concat with EVA rows can leave NUPACK columns as object dtype;
        # spearmanr then crashes inside numpy.corrcoef — coerce first.
        left = pd.to_numeric(frame[v_col], errors="coerce")
        right = pd.to_numeric(frame[n_col], errors="coerce")
        mask = left.notna() & right.notna()
        n = int(mask.sum())
        if n < 3:
            out[name] = {"n": n, "r_s": None, "p_value": None, "mae": None}
            continue
        x = left.loc[mask].to_numpy(dtype=float)
        y = right.loc[mask].to_numpy(dtype=float)
        r_s, p_value = spearmanr(x, y)
        mae = float(np.mean(np.abs(x - y)))
        out[name] = {
            "n": n,
            "r_s": None if pd.isna(r_s) else float(r_s),
            "p_value": None if pd.isna(p_value) else float(p_value),
            "mae": mae,
        }
    return out


def flag_software_fragile(frame: pd.DataFrame) -> pd.Series:
    """True when |ΔTm| > 5 °C or the engines disagree on switch vs open coil."""
    if "viennarna_Tm" not in frame.columns or "nupack_Tm" not in frame.columns:
        return pd.Series(False, index=frame.index)
    dtm = (frame["viennarna_Tm"] - frame["nupack_Tm"]).abs()
    v_h = pd.to_numeric(frame.get("viennarna_hill_coeff"), errors="coerce")
    n_h = pd.to_numeric(frame.get("nupack_hill_coeff"), errors="coerce")
    switch_vs_coil = ((v_h > HILL_SNAP) & (n_h <= OPEN_COIL_HILL_MAX)) | (
        (n_h > HILL_SNAP) & (v_h <= OPEN_COIL_HILL_MAX)
    )
    return (dtm > FRAGILE_DELTA_TM) | switch_vs_coil.fillna(False)


def composite_score(
    row: pd.Series,
    weights: dict[str, float] | None = None,
) -> float:
    """Tier-2-aligned lead score with capped $n_H$ and severe stroke penalty.

    $$
    \\mathrm{Score}
    = w_1 \\min(n_H, 3)
    + w_2 |\\Delta\\theta|
    - w_3 |T_m - 43.5|
    - w_4 P_{open}(37)
    - w_5 |\\Delta T_m|
    - \\mathrm{Penalty}_{stroke}
    $$

    $\\mathrm{Penalty}_{stroke}$ is large when $|\\Delta\\theta| < amp\\_floor$
    (default 0.25), so zero-amplitude $n_H=20$ fits leave the top-10.
    """
    w = weights or SCORE_WEIGHTS
    n_h_cap = float(w.get("n_h_cap", 3.0))
    amp_floor = float(w.get("amp_floor", 0.25))
    n_h = min(_finite(row.get("viennarna_hill_coeff"), 0.0), n_h_cap)
    amp = _finite(_row_amplitude_stroke(row), 0.0)
    tm = _finite(row.get("viennarna_Tm"), IDEAL_TM_C)
    p37 = _finite(row.get("viennarna_P_open_RBS_37"), 1.0)
    stroke_pen = float(w.get("w_stroke_penalty", 12.0)) * max(0.0, amp_floor - amp)
    v_tm = row.get("viennarna_Tm")
    n_tm = row.get("nupack_Tm")
    engine_pen = (
        abs(float(v_tm) - float(n_tm)) if pd.notna(v_tm) and pd.notna(n_tm) else 0.0
    )
    return (
        float(w.get("w_hill", 1.0)) * n_h
        + float(w.get("w_amp", 2.0)) * amp
        - float(w.get("w_tm", 0.4)) * abs(tm - IDEAL_TM_C)
        - float(w.get("w_leak", 6.0)) * p37
        - float(w.get("w_engine", 0.25)) * engine_pen
        - stroke_pen
    )


def _finite(value: object, default: float) -> float:
    """Return float(value) when finite, otherwise *default*."""
    if isinstance(value, (int, float, str)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default
    return default


def rank_leads(frame: pd.DataFrame, *, n: int = 10) -> pd.DataFrame:
    """Rank EVA passers by composite score and return the top *n* rows."""
    eva = frame[frame["cohort"].isin({COHORT_EVA_PILOT, COHORT_EVA_STREAM})].copy()
    if eva.empty:
        return eva
    eva["software_fragile"] = flag_software_fragile(eva)
    eva["composite_score"] = eva.apply(composite_score, axis=1)
    eva = eva.sort_values("composite_score", ascending=False)
    return eva.head(n)


def restriction_sites(sequence: str) -> list[str]:
    """Return restriction-enzyme names whose DNA motifs occur in *sequence*."""
    dna = sequence.upper().replace("U", "T")
    hits = [name for name, motif in RESTRICTION_SITES.items() if motif in dna]
    return hits


def mfe_dotbracket(sequence: str) -> tuple[str, float | None]:
    """Return the Vienna MFE dot-bracket and energy, or empty structure if unavailable."""
    try:
        from thermo_sim.vienna_rna import ViennaConfig, _fold_mfe
    except ImportError:
        return "", None
    energy, structure = _fold_mfe(sequence, ViennaConfig(dangles=2))
    return structure, energy


def write_rna_plot_ps(sequence: str, structure: str, output_path: Path) -> bool:
    """Write a ViennaRNA PostScript secondary-structure plot; return False if unavailable."""
    try:
        import RNA
    except ImportError:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = bool(RNA.file_PS_rnaplot(sequence, structure, str(output_path)))
    return ok


def export_leads(
    leads: pd.DataFrame,
    *,
    fasta_path: Path | None = None,
    fig_dir: Path | None = None,
) -> Path:
    """Write the top-lead FASTA plus restriction-site and MFE sidecar metadata."""
    fasta_path = fasta_path or resolve_path(DEFAULT_LEADS_FASTA)
    fig_dir = fig_dir or resolve_path(DEFAULT_FIG_DIR)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    meta_rows: list[dict[str, object]] = []
    for rank, (_, row) in enumerate(leads.iterrows(), start=1):
        seq = str(row.get("sequence") or "")
        sites = restriction_sites(seq)
        structure, energy = mfe_dotbracket(seq) if seq else ("", None)
        if seq and structure:
            write_rna_plot_ps(
                seq, structure, fig_dir / f"lead_{rank:02d}_{row['record_id']}.ps"
            )
        header = (
            f">{row['record_id']} rank={rank} cohort={row['cohort']} "
            f"score={float(row.get('composite_score', 0.0)):.4f} "
            f"n_H={_finite(row.get('viennarna_hill_coeff'), float('nan')):.3f} "
            f"Tm={_finite(row.get('viennarna_Tm'), float('nan')):.2f} "
            f"sites={','.join(sites) if sites else 'none'}"
        )
        lines.append(header)
        lines.append(seq)
        meta_rows.append(
            {
                "rank": rank,
                "record_id": row["record_id"],
                "restriction_sites": ",".join(sites),
                "mfe": energy,
                "dot_bracket": structure,
                "software_fragile": bool(row.get("software_fragile", False)),
            }
        )
    fasta_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    sidecar = fasta_path.with_suffix(".json")
    sidecar.write_text(json.dumps(meta_rows, indent=2))
    return fasta_path


def write_checklist_summary(table: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist two-tier gate pass rates as a lightweight JSON sidecar."""
    path = path or resolve_path(DEFAULT_CHECKLIST_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": {
            "tier1_biological_plausibility": {
                "n_H_min": TIER1_HILL,
                "Tm_C": [TIER1_TM_LO, TIER1_TM_HI],
                "delta_theta_min": TIER1_AMP_MIN,
                "P_open_RBS_37_max": TIER1_POPEN_MAX,
            },
            "tier2_neupert_2008_synbio": {
                "n_H_min": TIER2_HILL,
                "Tm_C": [TIER2_TM_LO, TIER2_TM_HI],
                "delta_theta_min": TIER2_AMP_MIN,
                "delta_theta_ideal": TIER2_AMP_IDEAL,
                "P_open_RBS_37_max": TIER2_POPEN_MAX,
                "citation": "Neupert et al. 2008 (heat-inducible RBS unmasking)",
            },
            "method_benchmark": {
                "citation": "Hoynes-O'Connor et al. 2015",
                "role": (
                    "experimental stress-test layout for synthetic RNA "
                    "temperature regulators (dual controls, orthogonality, "
                    "mechanism rescue) — heat-repressible RNase E class"
                ),
            },
            "target_box": {"Tm_C": [TIER2_TM_LO, TIER2_TM_HI], "n_H_min": TIER2_HILL},
            "ideal_leakiness": {
                "P_open_37_max": LEAKY_POPEN_MAX,
                "delta_P_RBS_min": IDEAL_DELTA_P,
            },
            "score_weights": SCORE_WEIGHTS,
        },
        "cohorts": json.loads(
            table.replace({np.nan: None}).to_json(orient="records") or "[]"
        ),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def plot_ribbon_curves(
    ribbon: pd.DataFrame,
    prototypes: pd.DataFrame | None = None,
    *,
    title: str = "Population-level RBS melting ribbons (30–60 °C)",
) -> Figure:
    """Plot grand-mean θ(T) ribbons with IQR shading for each cohort."""
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for cohort, sub in ribbon.groupby("cohort", sort=False):
        color = COHORT_COLORS.get(str(cohort), "#444444")
        ordered = sub.sort_values("temp_C")
        ax.fill_between(
            ordered["temp_C"],
            ordered["iqr_lo"],
            ordered["iqr_hi"],
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.plot(
            ordered["temp_C"],
            ordered["mean"],
            color=color,
            lw=2.2,
            label=f"{cohort} (n={int(ordered['n'].iloc[0])})",
        )
    if (
        prototypes is not None
        and not prototypes.empty
        and "temps" in prototypes.columns
    ):
        gold = prototypes[
            prototypes["rfam_id"]
            .astype(str)
            .str.contains("FourU", case=False, na=False)
        ]
        if gold.empty:
            gold = prototypes
        row = gold.iloc[0]
        ax.plot(
            row["temps"],
            row["p_open_curve"],
            color=COHORT_COLORS[COHORT_PROTOTYPE],
            lw=2.0,
            ls="--",
            label=f"FourU prototype ({row.get('rfam_id', 'FourU')})",
        )
    ax.axvspan(TM_MIN, TM_MAX, color="0.85", zorder=0, label="heat-shock Tm window")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"$P_{\mathrm{open}}$ / $\theta(T)$ (SD-window unpaired)")
    ax.set_xlim(SWEEP_TEMP_MIN, SWEEP_TEMP_MAX)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_nh_vs_tm(frame: pd.DataFrame) -> Figure:
    """Scatter n_H vs Tm with the gold-standard target box overlay."""
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.axvspan(TM_MIN, TM_MAX, color="#D9E8F5", zorder=0)
    ax.axhspan(HILL_SNAP, 20, xmin=0, xmax=1, color="#E8F5D9", alpha=0.35, zorder=0)
    ax.plot(
        [TM_MIN, TM_MAX, TM_MAX, TM_MIN, TM_MIN],
        [HILL_SNAP, HILL_SNAP, 20, 20, HILL_SNAP],
        color="#2166AC",
        lw=1.4,
        ls="--",
        label=r"target box: $T_m\in[42,45]$, $n_H\geq 1.5$",
    )
    for cohort, sub in frame.groupby("cohort", sort=False):
        ax.scatter(
            sub["viennarna_Tm"],
            sub["viennarna_hill_coeff"],
            s=18 if cohort in {COHORT_RFAM, COHORT_REFSEQ} else 36,
            alpha=0.35 if cohort in {COHORT_RFAM, COHORT_REFSEQ} else 0.85,
            color=COHORT_COLORS.get(str(cohort), "0.4"),
            label=str(cohort),
            edgecolors="none",
        )
    ax.set_xlabel(r"$T_m$ (°C)")
    ax.set_ylabel(r"$n_{\mathrm{H}}$")
    ax.set_ylim(0, 12)
    ax.set_xlim(20, 80)
    ax.set_title(r"Cooperativity vs midpoint ($n_{\mathrm{H}}$ vs $T_m$)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def plot_leakiness(frame: pd.DataFrame) -> Figure:
    """Scatter basal P_open(37 °C) against ΔP_RBS with the ideal quadrant."""
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    ax.axvspan(-0.05, LEAKY_POPEN_MAX, ymin=0, ymax=1, color="#F0F7E8", zorder=0)
    ax.axhspan(IDEAL_DELTA_P, 1.05, color="#F0F7E8", alpha=0.4, zorder=0)
    ax.axvline(LEAKY_POPEN_MAX, color="#1A7F37", ls="--", lw=1)
    ax.axhline(IDEAL_DELTA_P, color="#1A7F37", ls="--", lw=1)
    x_col = "viennarna_P_open_RBS_37"
    y_col = "viennarna_delta_P_RBS"
    for cohort, sub in frame.groupby("cohort", sort=False):
        if x_col not in sub.columns or y_col not in sub.columns:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=18 if cohort in {COHORT_RFAM, COHORT_REFSEQ} else 36,
            alpha=0.35 if cohort in {COHORT_RFAM, COHORT_REFSEQ} else 0.85,
            color=COHORT_COLORS.get(str(cohort), "0.4"),
            label=str(cohort),
            edgecolors="none",
        )
    ax.set_xlabel(r"$P_{\mathrm{open}}^{37^\circ\mathrm{C}}$ (basal leakiness)")
    ax.set_ylabel(r"$\Delta P_{\mathrm{RBS}}$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_title("Leakiness vs dynamic range")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_engine_parity(frame: pd.DataFrame) -> Figure:
    """Vienna vs NUPACK parity plots for Tm and Hill coefficient."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    specs = [
        ("viennarna_Tm", "nupack_Tm", r"$T_m$ (°C)", axes[0]),
        ("viennarna_hill_coeff", "nupack_hill_coeff", r"$n_{\mathrm{H}}$", axes[1]),
    ]
    for x_col, y_col, label, ax in specs:
        pts = (
            frame[[x_col, y_col, "cohort"]].dropna()
            if {x_col, y_col} <= set(frame.columns)
            else pd.DataFrame()
        )
        for cohort, sub in pts.groupby("cohort", sort=False):
            ax.scatter(
                sub[x_col],
                sub[y_col],
                s=22,
                alpha=0.7,
                color=COHORT_COLORS.get(str(cohort), "0.4"),
                label=str(cohort),
                edgecolors="none",
            )
        if not pts.empty:
            lo = float(min(pts[x_col].min(), pts[y_col].min()))
            hi = float(max(pts[x_col].max(), pts[y_col].max()))
            ax.plot([lo, hi], [lo, hi], color="0.3", lw=1, ls="--", label=r"$y=x$")
        ax.set_xlabel(rf"Vienna {label}")
        ax.set_ylabel(rf"NUPACK {label}")
        ax.set_title(f"{label} parity")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Multi-engine concordance (ViennaRNA vs NUPACK)")
    fig.tight_layout()
    return fig


def plot_novelty_vs_nh(frame: pd.DataFrame) -> Figure:
    """Scatter Rfam E-value (log) against switching quality for EVA passers.

    Uses Hill cooperativity when available; falls back to Z-score or ΔP_RBS.
    """
    eva = frame[frame["cohort"].isin({COHORT_EVA_PILOT, COHORT_EVA_STREAM})].copy()
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    if eva.empty or "E_Rfam" not in eva.columns:
        ax.set_title("Novelty vs switching quality (no EVA E-values loaded)")
        fig.tight_layout()
        return fig
    y_col, y_label = _pick_y_axis(eva)
    if y_col is None:
        ax.set_title("Novelty vs switching quality (no numeric columns)")
        fig.tight_layout()
        return fig
    eva["log10_E"] = eva["E_Rfam"].map(_log10_e)
    for cohort, sub in eva.groupby("cohort", sort=False):
        ax.scatter(
            sub["log10_E"],
            sub[y_col],
            s=40,
            alpha=0.85,
            color=COHORT_COLORS.get(str(cohort), "0.4"),
            label=str(cohort),
        )
    ax.axvline(
        math.log10(1e-3), color="0.4", ls=":", label=r"$E_{\mathrm{Rfam}}=10^{-3}$"
    )
    ax.set_xlabel(r"$\log_{10} E_{\mathrm{Rfam}}$ (no-hit plotted at $+8$)")
    ax.set_ylabel(y_label)
    ax.set_title(f"Sequence novelty vs switching quality ({y_col})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _pick_y_axis(
    eva: pd.DataFrame,
) -> tuple[str | None, str]:
    """Choose the best available y-axis column for the novelty scatter."""
    candidates = [
        ("viennarna_hill_coeff", r"$n_{\mathrm{H}}$"),
        ("viennarna_mfe_zscore", r"MFE $Z$-score"),
        ("viennarna_delta_P_RBS", r"$\Delta P_{\mathrm{RBS}}$"),
    ]
    for col, label in candidates:
        if col in eva.columns and eva[col].notna().any():
            return col, label
    return None, ""


def _log10_e(value: object) -> float:
    """Return log10(E), mapping +inf no-hit to +8 for plotting."""
    number = _as_erfam(value)
    if math.isinf(number):
        return 8.0
    return math.log10(max(number, 1e-300))


def plot_violins(frame: pd.DataFrame) -> Figure:
    """Four-panel violin plots of n_H, Tm, ΔP_RBS, and MFE Z-score by cohort."""
    features = [
        ("viennarna_hill_coeff", r"$n_{\mathrm{H}}$"),
        ("viennarna_Tm", r"$T_m$ (°C)"),
        ("viennarna_delta_P_RBS", r"$\Delta P_{\mathrm{RBS}}$"),
        ("viennarna_mfe_zscore", r"$Z$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 4.2))
    order = [COHORT_RFAM, COHORT_REFSEQ, COHORT_EVA_PILOT, COHORT_EVA_STREAM]
    present = [c for c in order if c in set(frame["cohort"])]
    for ax, (col, label) in zip(axes, features, strict=True):
        data = [
            pd.to_numeric(
                frame.loc[frame["cohort"] == c, col], errors="coerce"
            ).dropna()
            for c in present
        ]
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for body, cohort in zip(parts["bodies"], present, strict=True):
            body.set_facecolor(COHORT_COLORS.get(cohort, "0.5"))
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(present) + 1))
        ax.set_xticklabels(
            ["Rfam", "RefSeq", "Pilot", "Stream"][: len(present)],
            rotation=25,
            ha="right",
        )
        ax.set_ylabel(label)
        ax.set_title(label)
    fig.suptitle("EVA vs natural / control parameter distributions")
    fig.tight_layout()
    return fig


# ── Extended ML / synthetic-biology diagnostics (§7–§12) ─────────────────────

INDUCTION_ON_C = 42.0
INDUCTION_OFF_C = 37.0
HEATSHOCK_LO = 42.0
HEATSHOCK_HI = 45.0


def positional_unpairing_matrix(
    sequence: str,
    temps: Sequence[float] | list[int] | None = None,
) -> np.ndarray:
    """Build a (L × T) matrix of per-base unpaired probability vs temperature.

    Purpose: show *where* melting happens along the transcript.
    Why: a true thermoswitch should open mainly around the Shine–Dalgarno /
    RBS hairpin while the scaffold stem stays paired. Global brightening of
    the whole sequence means the RNA is falling apart, not switching.
    Looking for: a bright horizontal band near the SD window as T rises,
    with cooler (still-paired) scaffold positions.
    """
    from thermo_sim.vienna_rna import ViennaConfig, _partition_unpaired

    seq = sequence.upper().replace("T", "U")
    grid = list(temps) if temps is not None else sweep_temps()
    cols: list[np.ndarray] = []
    for temp in grid:
        profile = _partition_unpaired(seq, ViennaConfig(dangles=2, temperature_c=temp))
        cols.append(np.asarray(profile, dtype=float))
    if not cols:
        return np.zeros((len(seq), 0), dtype=float)
    return np.column_stack(cols)


def plot_positional_unpairing_heatmap(
    matrix: np.ndarray,
    temps: list[int],
    *,
    title: str,
    sd_span: tuple[int, int] | None = None,
) -> Figure:
    """Heatmap of unpaired probability (rows = position, columns = temperature).

    Bright = more unpaired. A vertical SD band that lights up only near
    heat-shock temperatures is the visual signature of localized switching.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    if matrix.size == 0:
        ax.text(0.5, 0.5, "No positional matrix", ha="center", va="center")
        ax.axis("off")
        return fig
    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        extent=(temps[0] - 0.5, temps[-1] + 0.5, 0.5, matrix.shape[0] + 0.5),
    )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Sequence position (nt)")
    ax.set_title(title)
    if sd_span is not None:
        lo, hi = sd_span
        ax.axhspan(lo + 0.5, hi + 0.5, color="cyan", alpha=0.18, label="SD window")
        ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$P_{\mathrm{unpaired}}$")
    fig.tight_layout()
    return fig


def sd_span_for_sequence(sequence: str) -> tuple[int, int] | None:
    """Return 0-based inclusive SD-window bounds for heatmap overlays."""
    from thermo_sim.thermo_common import sd_window_indices

    idx = sd_window_indices(sequence)
    if not idx:
        return None
    return int(min(idx)), int(max(idx))


def add_induction_ratio(
    frame: pd.DataFrame,
    *,
    sweeps: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach Fold-Change = P_open(42 °C) / P_open(37 °C) for each row.

    Purpose: measure operational ON/OFF induction the way a cell would care —
    how many times more open the RBS is at heat shock vs body temperature.
    Why: ΔP_RBS is an absolute gap; fold-change is signal-to-noise. A tiny
    absolute rise on a leaky baseline can look large in ΔP but poor as a ratio.
    Looking for: EVA leads with higher median fold-change than RefSeq, ideally
    competitive with Rfam.
    Prefers sweep interpolation, then Hill reconstruction, then 55/37 proxy.
    """
    out = frame.copy()
    ratios: list[float] = []
    sources: list[str] = []
    sweep_map: dict[str, pd.DataFrame] = {}
    if sweeps is not None and not sweeps.empty:
        for rid, sub in sweeps.groupby("record_id"):
            sweep_map[str(rid)] = sub

    for _, row in out.iterrows():
        rid = str(row.get("record_id", ""))
        p_on: float | None = None
        p_off: float | None = None
        source = "missing"
        if rid in sweep_map:
            p_off = _interp_at(sweep_map[rid], "p_open", INDUCTION_OFF_C)
            p_on = _interp_at(sweep_map[rid], "p_open", INDUCTION_ON_C)
            source = "sweep"
        if p_on is None or p_off is None:
            temps = np.asarray([INDUCTION_OFF_C, INDUCTION_ON_C], dtype=float)
            theta = reconstruct_theta(row, temps)
            if np.all(np.isfinite(theta)):
                p_off, p_on = float(theta[0]), float(theta[1])
                source = "hill"
        if (p_on is None or p_off is None) and "viennarna_P_open_RBS_37" in row.index:
            p_off = (
                float(row["viennarna_P_open_RBS_37"])
                if pd.notna(row.get("viennarna_P_open_RBS_37"))
                else None
            )
            p_on = (
                float(row["viennarna_P_open_RBS_55"])
                if pd.notna(row.get("viennarna_P_open_RBS_55"))
                else None
            )
            if p_on is not None and p_off is not None:
                source = "55_over_37_proxy"
        if p_on is None or p_off is None or p_off <= 1e-6:
            ratios.append(float("nan"))
            sources.append("missing")
        else:
            ratios.append(float(p_on / p_off))
            sources.append(source)
    out["fold_change_42_37"] = ratios
    out["fold_change_source"] = sources
    return out


def plot_induction_ratio_violins(frame: pd.DataFrame) -> Figure:
    """Violin comparison of operational fold-change across cohorts.

    Higher fold-change means a cleaner translation ON/OFF switch. RefSeq
    should sit near 1 (little induction); switches should sit well above 1.
    """
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    col = "fold_change_42_37"
    if col not in frame.columns or frame[col].isna().all():
        ax.text(0.5, 0.5, "Fold-change not available yet", ha="center", va="center")
        ax.axis("off")
        return fig
    order = [COHORT_RFAM, COHORT_REFSEQ, COHORT_EVA_PILOT, COHORT_EVA_STREAM]
    present = [c for c in order if c in set(frame["cohort"])]
    data = [
        pd.to_numeric(frame.loc[frame["cohort"] == c, col], errors="coerce")
        .dropna()
        .clip(upper=20)
        for c in present
    ]
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    raw_bodies = parts.get("bodies")
    bodies = list(cast(Iterable[Any], raw_bodies)) if raw_bodies is not None else []
    for body, cohort in zip(bodies, present, strict=True):
        body.set_facecolor(COHORT_COLORS.get(cohort, "0.5"))
        body.set_alpha(0.75)
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0, label="no induction")
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(
        ["Rfam", "RefSeq", "Pilot", "Stream"][: len(present)], rotation=20, ha="right"
    )
    ax.set_ylabel(
        r"$P_{\mathrm{open}}(42^\circ\mathrm{C}) / P_{\mathrm{open}}(37^\circ\mathrm{C})$"
    )
    ax.set_title("Operational induction ratio (Fold-Change ON/OFF)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def hill_dtheta_dT(
    temps: np.ndarray,
    *,
    bottom: float,
    top: float,
    tm: float,
    n_h: float,
) -> np.ndarray:
    """Analytical first derivative of the Hill sigmoid θ(T).

    Purpose: turn cooperativity into a visible peak. Sharp switches make a
    tall, narrow bump near Tm; gradual melts look like low, wide humps.
    Why: n_H alone is a single number; dθ/dT shows *where* the snap happens.
    Looking for: FWHM peak centered in the 42–45 °C heat-shock box.
    """
    amp = float(top) - float(bottom)
    t = np.asarray(temps, dtype=float)
    t = np.clip(t, 1e-6, None)
    tm_n = float(tm) ** float(n_h)
    t_n = t ** float(n_h)
    denom = tm_n + t_n
    return amp * float(n_h) * tm_n * (t ** (float(n_h) - 1.0)) / (denom**2)


def fwhm_of_peak(temps: np.ndarray, values: np.ndarray) -> float | None:
    """Full width at half-maximum (°C) of the tallest peak in *values*."""
    if len(temps) < 3 or not np.any(np.isfinite(values)):
        return None
    y = np.asarray(values, dtype=float)
    peak_i = int(np.nanargmax(y))
    half = 0.5 * float(y[peak_i])
    left = peak_i
    while left > 0 and y[left] >= half:
        left -= 1
    right = peak_i
    while right < len(y) - 1 and y[right] >= half:
        right += 1
    width = float(temps[right] - temps[left])
    return width if width > 0 else None


def derivative_profiles(
    frame: pd.DataFrame,
    temps: np.ndarray | None = None,
    *,
    max_per_cohort: int = 40,
) -> pd.DataFrame:
    """Long-form dθ/dT curves plus FWHM for a subsample of each cohort.

    Purpose: overlay many melting derivatives so sharp vs mushy transitions
    are obvious by eye. Looking for EVA curves that peak inside 42–45 °C
    with narrow FWHM; RefSeq should be flatter/dispersed.
    """
    grid = np.asarray(temps if temps is not None else sweep_temps(), dtype=float)
    rows: list[dict[str, object]] = []
    for cohort, sub in frame.groupby("cohort", sort=False):
        usable = sub.dropna(subset=["viennarna_Tm", "viennarna_hill_coeff"]).head(
            max_per_cohort
        )
        for _, row in usable.iterrows():
            tm = float(row["viennarna_Tm"])
            n_h = float(row["viennarna_hill_coeff"])
            amp = row.get("viennarna_amplitude")
            bottom = row.get("viennarna_hill_bottom")
            if pd.isna(amp):
                continue
            if pd.isna(bottom):
                bottom = 0.0
            top = float(bottom) + float(amp)
            dth = hill_dtheta_dT(grid, bottom=float(bottom), top=top, tm=tm, n_h=n_h)
            width = fwhm_of_peak(grid, dth)
            for temp, val in zip(grid, dth, strict=True):
                rows.append(
                    {
                        "cohort": cohort,
                        "record_id": row.get("record_id", ""),
                        "temp_C": float(temp),
                        "dtheta_dT": float(val),
                        "fwhm_C": width,
                        "peak_Tm": tm,
                    }
                )
    return pd.DataFrame(rows)


def plot_derivative_curves(profiles: pd.DataFrame) -> Figure:
    """Plot first-derivative melting curves with the heat-shock window shaded.

    Clean switches: tall, narrow peaks inside the cyan band. Non-switches:
    flat or smeared lines that never form a clear snap.
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    if profiles.empty:
        ax.text(0.5, 0.5, "No Hill parameters to differentiate", ha="center")
        ax.axis("off")
        return fig
    ax.axvspan(
        HEATSHOCK_LO, HEATSHOCK_HI, color="#7fdbda", alpha=0.25, label="42–45 °C"
    )
    for cohort, sub in profiles.groupby("cohort", sort=False):
        color = COHORT_COLORS.get(str(cohort), "0.5")
        for rid, curve in sub.groupby("record_id"):
            ax.plot(
                curve["temp_C"],
                curve["dtheta_dT"],
                color=color,
                alpha=0.18,
                linewidth=0.9,
            )
        mean = sub.groupby("temp_C", sort=True)["dtheta_dT"].mean()
        ax.plot(
            mean.index,
            mean.values,
            color=color,
            linewidth=2.2,
            label=str(cohort).replace(" Top Passers", "").replace("Natural ", ""),
        )
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(r"$\mathrm{d}\theta/\mathrm{d}T$")
    ax.set_title("First-derivative transition sharpness")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def tree_edit_distance(struct_a: str, struct_b: str) -> float:
    """Pairwise secondary-structure distance (Vienna tree-edit, BP fallback).

    Purpose: quantify how different two folds are as topologies, not just
    sequence strings. Why: generative models can mode-collapse into tiny
    variants of one hairpin; tree-edit distance catches that.
    """
    if not struct_a or not struct_b:
        return float("nan")
    try:
        import RNA

        return float(
            RNA.tree_edit_distance(RNA.expand_Full(struct_a), RNA.expand_Full(struct_b))
        )
    except Exception:  # noqa: BLE001 — UI fallback when Vienna tree API differs
        n = min(len(struct_a), len(struct_b))
        if n == 0:
            return float("nan")
        return float(
            sum(a != b for a, b in zip(struct_a[:n], struct_b[:n], strict=True))
        )


def structure_distance_matrix(sequences: list[str]) -> tuple[np.ndarray, list[str]]:
    """MFE-structure distance matrix for a list of RNA sequences."""
    structures: list[str] = []
    for seq in sequences:
        struct, _ = mfe_dotbracket(seq)
        structures.append(struct or ("." * len(seq)))
    n = len(structures)
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = tree_edit_distance(structures[i], structures[j])
            dist[i, j] = dist[j, i] = d if np.isfinite(d) else 0.0
    return dist, structures


def embed_structure_manifold(
    dist: np.ndarray,
    *,
    method: str = "MDS",
) -> tuple[np.ndarray, str]:
    """Embed a structure-distance matrix into 2D (MDS preferred, t-SNE fallback).

    Looking for: a cloud with multiple islands (diverse scaffolds). A tight
    single blob means mode collapse — many sequences, one fold idea.
    """
    if dist.shape[0] < 3:
        return np.zeros((dist.shape[0], 2)), "too few points"
    method_u = method.upper()
    if method_u == "TSNE":
        try:
            from sklearn.manifold import TSNE

            coords = TSNE(
                n_components=2,
                metric="precomputed",
                init="random",
                random_state=42,
                perplexity=min(30, max(2, dist.shape[0] // 4)),
            ).fit_transform(dist)
            return coords, "t-SNE"
        except Exception:  # noqa: BLE001 — fall through to MDS
            method_u = "MDS"
    from sklearn.manifold import MDS

    coords = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    ).fit_transform(dist)
    return coords, "MDS"


def plot_structure_manifold(
    coords: np.ndarray,
    labels: pd.Series | list[str],
    *,
    method_name: str,
) -> Figure:
    """2D scatter of structure-space embedding colored by cohort."""
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if coords.shape[0] == 0:
        ax.text(0.5, 0.5, "No structures to embed", ha="center")
        ax.axis("off")
        return fig
    label_series = pd.Series(labels)
    for cohort in label_series.unique():
        mask = (label_series == cohort).to_numpy()
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=36,
            alpha=0.75,
            color=COHORT_COLORS.get(str(cohort), "0.4"),
            label=str(cohort).replace(" Top Passers", ""),
        )
    ax.set_xlabel(f"{method_name} 1")
    ax.set_ylabel(f"{method_name} 2")
    ax.set_title("Structural diversity (Tree-edit / MFE manifold)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def gate_survival_funnel(frame: pd.DataFrame) -> pd.DataFrame:
    """Sequential four-gate attrition counts and fractions per cohort.

    Purpose: show how many sequences survive each *extra* physical rule in
    order (n_H → Tm → Δθ → baseline). Why: independent gate counts hide that
    almost nobody clears all four at once. Looking for: natural cohorts drop
    to ~0 by the last step; EVA should retain a measurable yield if the
    generator learned cooperative switching.
    """
    gate_order = [
        ("start", None),
        ("1_nH", "gate_nh"),
        ("2_nH+Tm", "gate_tm"),
        ("3_+amp", "gate_amp"),
        ("4_all", "gate_base"),
    ]
    rows: list[dict[str, object]] = []
    for cohort, sub in frame.groupby("cohort", sort=False):
        flags = pd.DataFrame([four_gate_flags(row) for _, row in sub.iterrows()])
        n0 = int(len(sub))
        alive = (
            pd.Series(True, index=flags.index)
            if not flags.empty
            else pd.Series(dtype=bool)
        )
        for step, key in gate_order:
            if key is not None and not flags.empty:
                alive = alive & flags[key]
            n_alive = int(n0 if key is None else alive.sum()) if n0 else 0
            rows.append(
                {
                    "cohort": cohort,
                    "step": step,
                    "n_alive": n_alive,
                    "frac_alive": float(n_alive / n0) if n0 else 0.0,
                    "n_start": n0,
                }
            )
    return pd.DataFrame(rows)


def plot_gate_survival_funnel(funnel: pd.DataFrame) -> Figure:
    """Step-chart of consecutive gate survival fractions by cohort.

    Each drop is sequences lost when the next physical gate is applied on
    top of the previous ones. Flat near zero on Rfam/RefSeq with a non-zero
    EVA tail means the generative model found rare all-four survivors.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    if funnel.empty:
        ax.text(0.5, 0.5, "No gate funnel data", ha="center")
        ax.axis("off")
        return fig
    step_order = ["start", "1_nH", "2_nH+Tm", "3_+amp", "4_all"]
    x = list(range(len(step_order)))
    for cohort, sub in funnel.groupby("cohort", sort=False):
        sub = sub.set_index("step").reindex(step_order)
        ax.plot(
            x,
            sub["frac_alive"].to_numpy(),
            marker="o",
            linewidth=2.0,
            color=COHORT_COLORS.get(str(cohort), "0.4"),
            label=str(cohort).replace(" Top Passers", "").replace("Natural ", ""),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Start", r"$n_H$", r"$+T_m$", r"$+\Delta\theta$", "+baseline"],
        fontsize=9,
    )
    ax.set_ylabel("Fraction still alive")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("4-gate cohort survival funnel (consecutive gates)")
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def _paired_positions(dot_bracket: str) -> dict[int, int]:
    """Map each paired index to its partner from a dot-bracket string."""
    stack: list[int] = []
    pairs: dict[int, int] = {}
    for i, ch in enumerate(dot_bracket):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            j = stack.pop()
            pairs[i] = j
            pairs[j] = i
    return pairs


def _wc_partner(base: str) -> str:
    """Return the Watson–Crick RNA partner of *base*."""
    return {"A": "U", "U": "A", "G": "C", "C": "G"}.get(base.upper(), "A")


def _mismatch_against(partner: str) -> str:
    """Return a base that cannot Watson–Crick pair with *partner*."""
    forbid = _wc_partner(partner)
    for base in ("A", "C", "G", "U"):
        if base != forbid:
            return base
    return "A"


def design_stem_mutations(sequence: str) -> dict[str, Any]:
    """Build wild-type, RBS-stem mismatch, and compensatory rescue sequences.

    Purpose: test whether switching depends on a specific base pair, not just
    overall %GC. Why: if a mismatch kills the snap and a compensatory mutation
    restores it, the phenotype is structure-driven. Looking for: disrupt curve
    loses amplitude / rises in leak; rescue returns toward wild-type.
    """
    seq = sequence.upper().replace("T", "U")
    structure, _ = mfe_dotbracket(seq)
    pairs = _paired_positions(structure)
    from thermo_sim.thermo_common import sd_window_indices

    sd_idx = set(sd_window_indices(seq))
    candidate = None
    for i in sorted(sd_idx):
        if i in pairs:
            candidate = i
            break
    if candidate is None:
        for i in range(len(seq)):
            if i in pairs:
                candidate = i
                break
    if candidate is None:
        return {
            "ok": False,
            "reason": "no paired stem positions found",
            "wildtype": seq,
            "disrupt": seq,
            "rescue": seq,
            "i": None,
            "j": None,
        }
    i = int(candidate)
    j = int(pairs[i])
    disrupt = list(seq)
    disrupt[i] = _mismatch_against(seq[j])
    rescue = disrupt.copy()
    rescue[j] = _wc_partner(disrupt[i])
    return {
        "ok": True,
        "reason": "ok",
        "wildtype": seq,
        "disrupt": "".join(disrupt),
        "rescue": "".join(rescue),
        "i": i,
        "j": j,
        "structure": structure,
    }


def characterize_mutation_trio(
    sequence: str,
    *,
    temps: list[int] | None = None,
) -> pd.DataFrame:
    """Sweep wild-type / disrupt / rescue variants and return long-form curves.

    Each variant gets a Vienna SD-window P_open(T) curve so we can overlay
    melting. Expect disrupt to flatten or leak; rescue to recover toward WT.
    """
    grid = temps or sweep_temps()
    design = design_stem_mutations(sequence)
    rows: list[dict[str, object]] = []
    if not design["ok"]:
        return pd.DataFrame(rows)
    for label in ("wildtype", "disrupt", "rescue"):
        curve = _vienna_hill_curve(str(design[label]), grid)
        for temp, val in zip(grid, curve, strict=True):
            rows.append(
                {
                    "variant": label,
                    "temp_C": int(temp),
                    "p_open": None if val is None else float(val),
                    "mutate_i": design["i"],
                    "mutate_j": design["j"],
                }
            )
    return pd.DataFrame(rows)


def plot_mutation_rescue_curves(
    curves_by_lead: dict[str, pd.DataFrame],
) -> Figure:
    """Overlay WT / mismatch / rescue melting curves for up to three leads.

    Solid = wild-type, dashed = disrupt, dotted = rescue. A real structure-
    driven switch shows disrupt failing and rescue recovering.
    """
    n = max(1, len(curves_by_lead))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    styles = {
        "wildtype": ("-", "#1B9E77"),
        "disrupt": ("--", "#D95F02"),
        "rescue": (":", "#7570B3"),
    }
    for ax, (lead_id, curve) in zip(axes, curves_by_lead.items(), strict=False):
        if curve.empty:
            ax.text(
                0.5, 0.5, "mutation design failed", ha="center", transform=ax.transAxes
            )
            ax.set_title(lead_id)
            continue
        for variant, sub in curve.groupby("variant"):
            ls, color = styles.get(str(variant), ("-", "0.4"))
            ax.plot(
                sub["temp_C"],
                sub["p_open"],
                linestyle=ls,
                color=color,
                linewidth=2.0,
                label=str(variant),
            )
        i = curve["mutate_i"].iloc[0]
        j = curve["mutate_j"].iloc[0]
        ax.set_title(f"{lead_id}\ni={i}, j={j}")
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel(r"$P_{\mathrm{open}}$ (SD)")
        ax.legend(fontsize=7)
    fig.suptitle("In silico stem disruption and compensatory rescue")
    fig.tight_layout()
    return fig


def _normalize_rna(sequence: str) -> str:
    """Uppercase RNA alphabet (T→U)."""
    return str(sequence or "").upper().replace("T", "U")


def scaffold_au_pairs_far_from_sd(
    sequence: str,
    *,
    n_pairs: int = 3,
) -> list[tuple[int, int, str, str]]:
    """Pick the farthest non-RBS A–U / U–A MFE pairs as G–C clamp sites.

    Purpose: strengthen scaffold stems that are *not* the Shine–Dalgarno
    duplex, so RBS opening can stay local. Why: EVA lead 315 melts globally;
    clamping distant AU stems tests whether instability is modular.

    Returns
    -------
    list of ``(i, j, from_pair, to_pair)`` with 0-based indices and labels
    like ``'A-U'`` → ``'G-C'``. Empty if no eligible pairs exist.
    """
    seq = _normalize_rna(sequence)
    if not seq:
        return []
    struct, _ = mfe_dotbracket(seq)
    pairs = _paired_positions(struct)
    sd = set(sd_window_indices(seq))
    if not sd:
        return []
    sd_lo, sd_hi = min(sd), max(sd)
    candidates: list[tuple[int, int, int, str, str]] = []
    for i, j in pairs.items():
        if i >= j:
            continue
        if i in sd or j in sd:
            continue
        a, b = seq[i], seq[j]
        if {a, b} != {"A", "U"}:
            continue
        dist = min(abs(i - sd_lo), abs(j - sd_lo), abs(i - sd_hi), abs(j - sd_hi))
        candidates.append((dist, i, j, a, b))
    candidates.sort(key=lambda t: t[0], reverse=True)
    edits: list[tuple[int, int, str, str]] = []
    for _, i, j, a, b in candidates[: max(0, int(n_pairs))]:
        if a == "A" and b == "U":
            edits.append((i, j, "A-U", "G-C"))
        elif a == "U" and b == "A":
            edits.append((i, j, "U-A", "C-G"))
    return edits


def apply_pair_clamp_edits(
    sequence: str,
    edits: list[tuple[int, int, str, str]],
) -> str:
    """Apply A–U→G–C (or U–A→C–G) clamp edits; leave other bases untouched.

    Purpose: build the engineered variant for before/after heatmaps.
    """
    mut = list(_normalize_rna(sequence))
    for i, j, _fr, to in edits:
        left, right = to.split("-")
        if not (0 <= i < len(mut) and 0 <= j < len(mut)):
            continue
        mut[i] = left
        mut[j] = right
    return "".join(mut)


def positional_heatmap_metrics(
    mat: np.ndarray,
    temps: Sequence[float],
    sd_span: tuple[int, int],
    *,
    focus_temps: Sequence[float] = (37.0, 42.0, 43.0, 45.0),
) -> dict[str, float]:
    """Summarize global / scaffold / SD unpairing from a positional matrix.

    Purpose: score rescue scenarios without eyeballing every pixel.
    ``stroke_sd_43_37`` is Δθ on the SD window between 37 and 43 °C.
    """
    t_idx = {float(t): k for k, t in enumerate(temps)}
    lo, hi = int(sd_span[0]), int(sd_span[1])
    out: dict[str, float] = {}
    for T in focus_temps:
        key = float(T)
        if key not in t_idx:
            continue
        col = mat[:, t_idx[key]]
        mask = np.ones(len(col), dtype=bool)
        mask[lo : hi + 1] = False
        out[f"sd_{int(T)}"] = float(col[lo : hi + 1].mean())
        out[f"global_{int(T)}"] = float(col.mean())
        out[f"scaffold_{int(T)}"] = (
            float(col[mask].mean()) if mask.any() else float("nan")
        )
    if "sd_43" in out and "sd_37" in out:
        out["stroke_sd_43_37"] = float(out["sd_43"] - out["sd_37"])
    return out


def classify_scaffold_clamp_rescue(
    wildtype: dict[str, float],
    clamped: dict[str, float],
) -> int:
    """Map before/after metrics to rescue scenario 1 / 2 / 3.

    1 — Ideal modular rescue (scaffold quiet, SD stroke large).
    2 — Negative design trap (SD locked / no stroke).
    3 — Partial / ineffective (scaffold still frays).
    """
    g43 = float(clamped.get("global_43", 1.0))
    sd43 = float(clamped.get("sd_43", 0.0))
    stroke = float(clamped.get("stroke_sd_43_37", 0.0))
    if g43 <= 0.25 and stroke >= 0.40 and sd43 >= 0.35:
        return 1
    if sd43 <= 0.10 and abs(stroke) < 0.05:
        return 2
    _ = wildtype  # retained for API symmetry / future deltas
    return 3


def run_scaffold_gc_clamp_poc(
    sequence: str,
    *,
    record_id: str = "eva_sample_315",
    n_pairs: int = 3,
    temps: Sequence[float] | None = None,
) -> dict[str, Any]:
    """One-sequence PoC: clamp distant scaffold A–U pairs and re-heatmap.

    Hypothesis
    ----------
    Targeted G–C clamps on non-RBS stems of ``eva_sample_315`` suppress
    global background melting at 42–45 °C while preserving SD unpairing.
    """
    seq = _normalize_rna(sequence)
    temps_list = [float(t) for t in (temps if temps is not None else sweep_temps())]
    edits = scaffold_au_pairs_far_from_sd(seq, n_pairs=n_pairs)
    mut = apply_pair_clamp_edits(seq, edits) if edits else seq
    span = sd_span_for_sequence(seq) or (0, 0)
    mat_wt = positional_unpairing_matrix(seq, temps_list)
    mat_mut = positional_unpairing_matrix(mut, temps_list)
    m_wt = positional_heatmap_metrics(mat_wt, temps_list, span)
    m_mut = positional_heatmap_metrics(mat_mut, temps_list, span)
    scenario = classify_scaffold_clamp_rescue(m_wt, m_mut)
    return {
        "record_id": record_id,
        "edits": [{"i": i, "j": j, "from": fr, "to": to} for i, j, fr, to in edits],
        "wildtype_sequence": seq,
        "clamped_sequence": mut,
        "temps": temps_list,
        "sd_span": [int(span[0]), int(span[1])],
        "mat_wildtype": mat_wt,
        "mat_clamped": mat_mut,
        "wildtype_metrics": m_wt,
        "clamped_metrics": m_mut,
        "scenario": int(scenario),
    }


def plot_before_after_positional_heatmaps(
    poc: dict[str, Any],
    *,
    title: str = "§13 PoC: scaffold G–C clamps",
) -> Figure:
    """Side-by-side Before vs After positional unpairing heatmaps.

    Purpose: visual case-study box for the manual rescue attempt.
    Cyan band = SD window; white band = 42–45 °C design range.
    """
    temps = [float(t) for t in poc["temps"]]
    mat_wt = np.asarray(poc["mat_wildtype"], dtype=float)
    mat_mut = np.asarray(poc["mat_clamped"], dtype=float)
    lo, hi = int(poc["sd_span"][0]), int(poc["sd_span"][1])
    rid = str(poc.get("record_id") or "lead")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    panels = (
        (axes[0], mat_wt, f"Before: {rid} WT"),
        (
            axes[1],
            mat_mut,
            f"After: +{len(poc.get('edits') or [])} scaffold G–C clamps",
        ),
    )
    im = None
    for ax, mat, panel_title in panels:
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            extent=(temps[0] - 0.5, temps[-1] + 0.5, 0.5, mat.shape[0] + 0.5),
        )
        ax.axhspan(lo + 0.5, hi + 0.5, color="cyan", alpha=0.22, label="SD")
        ax.axvspan(42.0, 45.0, color="white", alpha=0.08)
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Sequence position (nt)")
    axes[0].legend(loc="upper right", fontsize=8)
    if im is not None:
        fig.colorbar(
            im, ax=axes, fraction=0.03, pad=0.02, label=r"$P_{\mathrm{unpaired}}$"
        )
    fig.suptitle(title, y=1.02)
    fig.subplots_adjust(right=0.88, wspace=0.12)
    return fig


def save_figure(fig: Figure, path: Path) -> Path:
    """Save a matplotlib figure to *path* and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path
