import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path

DEFAULT_MODEL_PATH = "data/processed/models/rf_thermoswitch.joblib"
DEFAULT_NONCIRCULAR_MODEL_PATH = (
    "data/processed/models/rf_thermoswitch_noncircular.joblib"
)
DEFAULT_LENGTH_MATCHED_MODEL_PATH = (
    "data/processed/models/rf_thermoswitch_length_matched.joblib"
)
DEFAULT_TRAINING_FUSED = "data/processed/fused_features.csv"
DEFAULT_LENGTH_MATCHED_FUSED = "data/processed/fused_features_length_matched.csv"
DEFAULT_DENOVO_FUSED = "data/processed/denovo_fused_features.csv"
DEFAULT_PREDICTIONS = "data/processed/denovo_predictions.csv"
DEFAULT_FEATURE_LOG = "data/processed/rf_noncircular_feature_log.json"

# Legacy extensive-feature set (raw MFE + absolute stem/loop). Kept for comparison.
LEGACY_PHYSICS_FEATURE_COLUMNS = [
    "nupack_MFE",
    "nupack_max_stem_length",
    "nupack_max_loop_length",
    "nupack_mean_exposure",
    "nupack_gc_content",
    "nupack_Tm",
    "nupack_hill_coeff",
    "nupack_amplitude",
    "viennarna_MFE",
    "viennarna_max_loop_length",
    "viennarna_gc_content",
    "viennarna_mean_unpaired_prob",
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
]

# Previous intensive+dynamic 20-column set. Circular: Tm/Hill/amplitude/Z/ΔP_RBS/ΔΔG
# are the melting phenotype and must not be RF inputs. Kept for XGBoost / comparison.
CIRCULAR_PHYSICS_FEATURE_COLUMNS = [
    "nupack_MFE_per_nt",
    "nupack_max_stem_frac",
    "nupack_max_loop_frac",
    "nupack_mean_exposure",
    "nupack_gc_content",
    "nupack_Tm",
    "nupack_hill_coeff",
    "nupack_amplitude",
    "viennarna_MFE_per_nt",
    "viennarna_max_loop_frac",
    "viennarna_gc_content",
    "viennarna_mean_unpaired_prob",
    "viennarna_Tm",
    "viennarna_hill_coeff",
    "viennarna_amplitude",
    "viennarna_mfe_zscore",
    "viennarna_delta_P_RBS",
    "viennarna_delta_delta_G",
    "viennarna_ensemble_diversity",
    "viennarna_mean_positional_entropy",
]

# Alias retained so older scripts that import PHYSICS_FEATURE_COLUMNS keep the 20-col set.
PHYSICS_FEATURE_COLUMNS = CIRCULAR_PHYSICS_FEATURE_COLUMNS

# XGBoost monotone_constraints: -1 inverse, +1 direct, 0 unconstrained (vs P(positive)).
MONOTONE_CONSTRAINTS_BY_FEATURE = {
    "viennarna_mfe_zscore": -1,
    "viennarna_delta_P_RBS": 1,
    "viennarna_delta_delta_G": -1,
    "viennarna_ensemble_diversity": 1,
    "viennarna_mean_positional_entropy": 1,
    "viennarna_MFE_per_nt": -1,
    "nupack_MFE_per_nt": -1,
    "nupack_amplitude": 1,
    "nupack_hill_coeff": 1,
    "nupack_max_stem_frac": 0,
    "nupack_max_loop_frac": 0,
    "viennarna_max_loop_frac": 0,
    "nupack_Tm": 0,
    # Remaining intensive cols default to unconstrained (0) via .get(..., 0)
}

DEFAULT_XGB_MODEL_PATH = "data/processed/models/xgb_thermoswitch_refseq_dynamic.joblib"
DEFAULT_REFSEQ_DYNAMIC_FUSED = "data/processed/fused_features_refseq_dynamic.csv"


def monotone_constraints_tuple(feature_cols: list[str]) -> tuple[int, ...]:
    """Build XGBoost monotone_constraints aligned to feature column order."""
    return tuple(int(MONOTONE_CONSTRAINTS_BY_FEATURE.get(c, 0)) for c in feature_cols)


def build_xgboost_monotonic(
    feature_cols: list[str],
    *,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[object, tuple[int, ...]]:
    """Construct an XGBClassifier with physical monotone_constraints."""
    from xgboost import XGBClassifier

    constraints = monotone_constraints_tuple(feature_cols)
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=random_state,
        n_jobs=-1,
        monotone_constraints=constraints,
    ), constraints


def add_intensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MFE/N and stem/loop fraction columns derived from raw fused features."""
    out = df.copy()
    if "seq_length" not in out.columns:
        raise ValueError("seq_length required to build intensive features")
    length = out["seq_length"].replace(0, pd.NA)
    if "viennarna_MFE" in out.columns:
        out["viennarna_MFE_per_nt"] = out["viennarna_MFE"] / length
    if "nupack_MFE" in out.columns:
        out["nupack_MFE_per_nt"] = out["nupack_MFE"] / length
    if "nupack_max_stem_length" in out.columns:
        out["nupack_max_stem_frac"] = out["nupack_max_stem_length"] / length
    if "nupack_max_loop_length" in out.columns:
        out["nupack_max_loop_frac"] = out["nupack_max_loop_length"] / length
    if "viennarna_max_loop_length" in out.columns:
        out["viennarna_max_loop_frac"] = out["viennarna_max_loop_length"] / length
    return out


def _available_features(
    df: pd.DataFrame, feature_columns: list[str] | None = None
) -> list[str]:
    """Return physics feature columns that are present on the frame."""
    cols = feature_columns if feature_columns is not None else PHYSICS_FEATURE_COLUMNS
    return [col for col in cols if col in df.columns]


def train_random_forest(
    fused_csv: str = DEFAULT_TRAINING_FUSED,
    model_path: str | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
    intensive: bool = True,
    feature_set: str | None = None,
    dataset_csv: str | None = None,
    dataset_fasta: str | None = None,
    denovo_fasta: str | None = None,
    feature_log_json: str = DEFAULT_FEATURE_LOG,
) -> Path:
    """Train a Random Forest on fused physics features and persist the model."""
    from thermo_sim.noncircular_features import (
        DEFAULT_DATASET_CSV,
        DEFAULT_DATASET_FASTA,
        build_noncircular_matrix,
        noncircular_feature_columns,
        physics_dropna_columns,
        write_feature_log,
    )

    if feature_set is None:
        feature_set = "legacy" if not intensive else "noncircular"
    if model_path is None:
        model_path = (
            DEFAULT_NONCIRCULAR_MODEL_PATH
            if feature_set == "noncircular"
            else DEFAULT_MODEL_PATH
        )

    fused_csv = resolve_path(fused_csv)
    df = pd.read_csv(fused_csv)
    if "label" not in df.columns:
        raise ValueError(f"{fused_csv} must contain a label column for training.")

    dataset_csv = dataset_csv or DEFAULT_DATASET_CSV
    dataset_fasta = dataset_fasta or DEFAULT_DATASET_FASTA

    if feature_set == "noncircular":
        df = add_intensive_features(df)
        df = build_noncircular_matrix(
            df,
            dataset_csv=dataset_csv,
            dataset_fasta=dataset_fasta,
            denovo_fasta=denovo_fasta,
            already_intensive=True,
        )
        feature_cols = noncircular_feature_columns(df)
        dropna_cols = physics_dropna_columns(df)
    elif feature_set == "circular":
        df = add_intensive_features(df)
        feature_cols = _available_features(df, CIRCULAR_PHYSICS_FEATURE_COLUMNS)
        dropna_cols = feature_cols
    elif feature_set == "legacy":
        feature_cols = _available_features(df, LEGACY_PHYSICS_FEATURE_COLUMNS)
        dropna_cols = feature_cols
    else:
        raise ValueError(f"Unknown feature_set={feature_set!r}")

    if not feature_cols:
        raise ValueError("No feature columns found in fused CSV.")

    train_df = df.dropna(subset=["label"] + dropna_cols)
    x = train_df[feature_cols]
    y = train_df["label"].astype(int)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x, y)

    model_path = resolve_path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "feature_columns": feature_cols,
        "feature_set": feature_set,
        "n_rows": int(len(train_df)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "circular_excluded": [
            "nupack_Tm",
            "nupack_hill_coeff",
            "nupack_amplitude",
            "viennarna_Tm",
            "viennarna_hill_coeff",
            "viennarna_amplitude",
            "viennarna_mfe_zscore",
            "viennarna_delta_P_RBS",
            "viennarna_delta_delta_G",
        ]
        if feature_set == "noncircular"
        else [],
    }
    joblib.dump(payload, model_path)
    sidecar = model_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({k: v for k, v in payload.items() if k != "model"}, indent=2)
    )

    if feature_set == "noncircular":
        write_feature_log(
            train_df,
            feature_cols,
            output_json=feature_log_json,
            dataset_fasta=dataset_fasta,
            extra={"model_path": str(model_path), "n_fit": int(len(train_df))},
        )

    print(f"Trained RandomForest on {len(train_df)} rows, {len(feature_cols)} features")
    print(f"  feature_set: {feature_set}")
    print(f"  features: {', '.join(feature_cols)}")
    print(f"  model:    {model_path}")
    return model_path


def train_xgboost_monotonic(
    fused_csv: str = DEFAULT_REFSEQ_DYNAMIC_FUSED,
    model_path: str = DEFAULT_XGB_MODEL_PATH,
    n_estimators: int = 300,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Path:
    """Train monotonic XGBoost on intensive+dynamic fused features."""
    fused_csv = resolve_path(fused_csv)
    df = pd.read_csv(fused_csv)
    if "label" not in df.columns:
        raise ValueError(f"{fused_csv} must contain a label column for training.")
    df = add_intensive_features(df)
    feature_cols = _available_features(df, PHYSICS_FEATURE_COLUMNS)
    if not feature_cols:
        raise ValueError("No physics feature columns found in fused CSV.")

    train_df = df.dropna(subset=["label"] + feature_cols)
    x = train_df[feature_cols]
    y = train_df["label"].astype(int)

    model, constraints = build_xgboost_monotonic(
        feature_cols,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )
    model.fit(x, y)

    model_path = resolve_path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "monotone_constraints": constraints,
            "monotone_constraints_by_feature": {
                c: int(MONOTONE_CONSTRAINTS_BY_FEATURE.get(c, 0)) for c in feature_cols
            },
            "intensive": True,
        },
        model_path,
    )
    print(
        f"Trained monotonic XGBoost on {len(train_df)} rows, {len(feature_cols)} features"
    )
    print(f"  features: {', '.join(feature_cols)}")
    print(f"  monotone_constraints: {constraints}")
    print(f"  model:    {model_path}")
    return model_path


def _resolve_id_column(df: pd.DataFrame) -> str:
    """Pick the first available join/id column for prediction output."""
    from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS

    for col in ("record_id", *JOIN_COLUMNS):
        if col in df.columns:
            return col
    raise ValueError("No join column found for predictions output.")


def predict_thermoswitches(
    fused_csv: str = DEFAULT_DENOVO_FUSED,
    model_path: str = DEFAULT_MODEL_PATH,
    predictions_csv: str = DEFAULT_PREDICTIONS,
    join_column: str = "record_id",
    dataset_csv: str | None = None,
    dataset_fasta: str | None = None,
    denovo_fasta: str | None = None,
) -> Path:
    """Score fused sequences with a saved classifier and write predictions."""
    from thermo_sim.noncircular_features import (
        DEFAULT_DATASET_CSV,
        DEFAULT_DATASET_FASTA,
        build_noncircular_matrix,
        physics_dropna_columns,
    )

    fused_csv = resolve_path(fused_csv)
    model_path = resolve_path(model_path)
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_columns"]
    feature_set = payload.get("feature_set", "circular")

    df = pd.read_csv(fused_csv)
    if feature_set == "noncircular":
        df = add_intensive_features(df) if "seq_length" in df.columns else df
        df = build_noncircular_matrix(
            df,
            dataset_csv=dataset_csv or DEFAULT_DATASET_CSV,
            dataset_fasta=dataset_fasta or DEFAULT_DATASET_FASTA,
            denovo_fasta=denovo_fasta,
            already_intensive=True,
        )
        dropna_cols = physics_dropna_columns(df)
    else:
        if any(col not in df.columns for col in feature_cols):
            df = add_intensive_features(df)
        dropna_cols = feature_cols

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Fused CSV missing feature columns: {missing}")

    predict_df = df.dropna(subset=dropna_cols)
    proba = model.predict_proba(predict_df[feature_cols])
    classes = list(model.classes_)
    if 1 in classes:
        probs = proba[:, classes.index(1)]
    else:
        # Smoke / degenerate train sets may only contain label 0.
        probs = [0.0] * len(predict_df)
    labels = model.predict(predict_df[feature_cols])

    id_col = (
        join_column
        if join_column in predict_df.columns
        else _resolve_id_column(predict_df)
    )
    results = predict_df[[id_col]].copy()
    results["prob_positive"] = probs
    results["predicted_label"] = labels

    predictions_csv = resolve_path(predictions_csv)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(predictions_csv, index=False)
    print(f"Wrote {len(results)} predictions to {predictions_csv}")
    return predictions_csv


def _build_parser() -> argparse.ArgumentParser:
    """Build the train/predict/posthoc CLI parser."""
    parser = argparse.ArgumentParser(
        description="Train or run thermoswitch classifiers (RF / monotonic XGBoost)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser(
        "train", help="Train RF on labeled fused_features.csv"
    )
    train_parser.add_argument("--fused-csv", default=DEFAULT_REFSEQ_DYNAMIC_FUSED)
    train_parser.add_argument("--model-path", default=DEFAULT_NONCIRCULAR_MODEL_PATH)
    train_parser.add_argument(
        "--legacy-features",
        action="store_true",
        help="Use raw MFE + absolute stem/loop features instead of intensive set.",
    )
    train_parser.add_argument(
        "--circular-features",
        action="store_true",
        help="Use the previous 20-column intensive+dynamic set (includes Tm/Hill/Z/ΔP_RBS).",
    )
    train_parser.add_argument(
        "--dataset-csv",
        default="data/processed/balanced/length_gc_matched_refseq_dataset.csv",
    )
    train_parser.add_argument(
        "--dataset-fasta",
        default="data/processed/balanced/length_gc_matched_refseq_dataset.fasta",
    )
    train_parser.add_argument("--denovo-fasta", default=None)
    train_parser.add_argument("--feature-log-json", default=DEFAULT_FEATURE_LOG)

    train_xgb = sub.add_parser(
        "train-xgb",
        help="Train monotonic XGBoost on intensive+dynamic fused features",
    )
    train_xgb.add_argument("--fused-csv", default=DEFAULT_REFSEQ_DYNAMIC_FUSED)
    train_xgb.add_argument("--model-path", default=DEFAULT_XGB_MODEL_PATH)
    train_xgb.add_argument("--n-estimators", type=int, default=300)
    train_xgb.add_argument("--max-depth", type=int, default=4)
    train_xgb.add_argument("--learning-rate", type=float, default=0.05)

    predict_parser = sub.add_parser("predict", help="Predict on de novo fused features")
    predict_parser.add_argument("--fused-csv", default=DEFAULT_DENOVO_FUSED)
    predict_parser.add_argument("--model-path", default=DEFAULT_NONCIRCULAR_MODEL_PATH)
    predict_parser.add_argument("--predictions-csv", default=DEFAULT_PREDICTIONS)
    predict_parser.add_argument(
        "--dataset-csv",
        default="data/processed/balanced/length_gc_matched_refseq_dataset.csv",
    )
    predict_parser.add_argument(
        "--dataset-fasta",
        default="data/processed/balanced/length_gc_matched_refseq_dataset.fasta",
    )
    predict_parser.add_argument("--denovo-fasta", default=None)

    posthoc = sub.add_parser(
        "posthoc",
        help="OOF confidence bins, melting gates, Spearman, MW/KS (not RF inputs)",
    )
    posthoc.add_argument("--fused-csv", default=DEFAULT_REFSEQ_DYNAMIC_FUSED)
    posthoc.add_argument(
        "--output-json", default="data/processed/rf_posthoc_report.json"
    )
    posthoc.add_argument("--group-col", default="rfam_acc")
    posthoc.add_argument(
        "--dataset-csv",
        default="data/processed/balanced/length_gc_matched_refseq_dataset.csv",
    )
    posthoc.add_argument(
        "--dataset-fasta",
        default="data/processed/balanced/length_gc_matched_refseq_dataset.fasta",
    )
    posthoc.add_argument(
        "--diagnostics-json",
        default="data/processed/rf_noncircular_diagnostics.json",
    )
    return parser


def _run_posthoc_cli(args: argparse.Namespace) -> None:
    """Run grouped CV diagnostics and post-hoc melting gates from CLI args."""
    import json as _json

    from sklearn.ensemble import RandomForestClassifier as _RF
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        StratifiedKFold,
        cross_val_predict,
    )

    from thermo_sim.noncircular_features import (
        build_noncircular_matrix,
        grouped_permutation_importance,
        noncircular_feature_columns,
        physics_dropna_columns,
        write_feature_log,
    )
    from thermo_sim.rf_posthoc import evaluate_posthoc

    df = pd.read_csv(resolve_path(args.fused_csv))
    df = add_intensive_features(df)
    df = build_noncircular_matrix(
        df,
        dataset_csv=args.dataset_csv,
        dataset_fasta=args.dataset_fasta,
        already_intensive=True,
    )
    feature_cols = noncircular_feature_columns(df)
    dropna_cols = physics_dropna_columns(df)
    clean = df.dropna(subset=["label"] + dropna_cols).copy()
    y = clean["label"].astype(int)
    X = clean[feature_cols]
    groups = (
        clean[args.group_col].astype(str) if args.group_col in clean.columns else None
    )

    def _eval_matrix(
        matrix: pd.DataFrame, cv: object, grp: pd.Series | None = None
    ) -> tuple[dict[str, float], object]:
        """Score ROC-AUC and accuracy from out-of-fold RF probabilities."""
        rf = _RF(n_estimators=200, random_state=42, n_jobs=-1)
        kwargs = {"cv": cv, "method": "predict_proba"}
        if grp is not None:
            kwargs["groups"] = grp
        proba = cross_val_predict(rf, matrix, y, **kwargs)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "roc_auc": float(roc_auc_score(y, proba)),
            "accuracy": float(accuracy_score(y, pred)),
        }, proba

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified, _ = _eval_matrix(X, skf)
    length_alone, _ = _eval_matrix(clean[["seq_length"]], skf)
    group_metrics = None
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        group_metrics, yhat = _eval_matrix(X, sgkf, groups)
    else:
        _, yhat = _eval_matrix(X, skf)

    fitted = _RF(n_estimators=200, random_state=42, n_jobs=-1)
    fitted.fit(X, y)
    grouped_imp = grouped_permutation_importance(fitted, X, y, n_repeats=5)

    diagnostics = {
        "n": int(len(clean)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "length_alone_stratified": length_alone,
        "noncircular_stratified": stratified,
        "noncircular_stratified_group": group_metrics,
        "grouped_permutation_importance": grouped_imp,
    }
    diag_path = resolve_path(args.diagnostics_json)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(_json.dumps(diagnostics, indent=2))
    print(f"Wrote {diag_path}")

    write_feature_log(
        clean,
        feature_cols,
        dataset_fasta=args.dataset_fasta,
        extra={"diagnostics_json": str(diag_path)},
    )
    evaluate_posthoc(
        clean,
        feature_cols,
        group_col=args.group_col,
        yhat=yhat,
        output_json=args.output_json,
    )


def main() -> None:
    """Dispatch thermoswitch train, predict, or post-hoc CLI commands."""
    args = _build_parser().parse_args()
    if args.command == "train":
        if args.legacy_features:
            feature_set = "legacy"
        elif args.circular_features:
            feature_set = "circular"
        else:
            feature_set = "noncircular"
        train_random_forest(
            fused_csv=args.fused_csv,
            model_path=args.model_path,
            feature_set=feature_set,
            dataset_csv=args.dataset_csv,
            dataset_fasta=args.dataset_fasta,
            denovo_fasta=args.denovo_fasta,
            feature_log_json=args.feature_log_json,
        )
    elif args.command == "train-xgb":
        train_xgboost_monotonic(
            fused_csv=args.fused_csv,
            model_path=args.model_path,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
        )
    elif args.command == "predict":
        predict_thermoswitches(
            fused_csv=args.fused_csv,
            model_path=args.model_path,
            predictions_csv=args.predictions_csv,
            dataset_csv=args.dataset_csv,
            dataset_fasta=args.dataset_fasta,
            denovo_fasta=args.denovo_fasta,
        )
    elif args.command == "posthoc":
        _run_posthoc_cli(args)


if __name__ == "__main__":
    main()
