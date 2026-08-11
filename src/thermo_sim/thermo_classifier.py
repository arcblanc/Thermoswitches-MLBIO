import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_engineering.paths import resolve_path

DEFAULT_MODEL_PATH = "data/processed/models/rf_thermoswitch.joblib"
DEFAULT_LENGTH_MATCHED_MODEL_PATH = "data/processed/models/rf_thermoswitch_length_matched.joblib"
DEFAULT_TRAINING_FUSED = "data/processed/fused_features.csv"
DEFAULT_LENGTH_MATCHED_FUSED = "data/processed/fused_features_length_matched.csv"
DEFAULT_DENOVO_FUSED = "data/processed/denovo_fused_features.csv"
DEFAULT_PREDICTIONS = "data/processed/denovo_predictions.csv"

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

# Intensive features: length-normalized MFE and stem/loop fractions (no raw MFE).
PHYSICS_FEATURE_COLUMNS = [
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
    # Dynamic / composition-relative Vienna features
    "viennarna_mfe_zscore",
    "viennarna_delta_P_RBS",
    "viennarna_delta_delta_G",
    "viennarna_ensemble_diversity",
    "viennarna_mean_positional_entropy",
]

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
):
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


def _available_features(df, feature_columns=None):
    cols = feature_columns if feature_columns is not None else PHYSICS_FEATURE_COLUMNS
    return [col for col in cols if col in df.columns]


def train_random_forest(
    fused_csv=DEFAULT_TRAINING_FUSED,
    model_path=DEFAULT_MODEL_PATH,
    n_estimators=200,
    random_state=42,
    intensive=True,
):
    fused_csv = resolve_path(fused_csv)
    df = pd.read_csv(fused_csv)
    if "label" not in df.columns:
        raise ValueError(f"{fused_csv} must contain a label column for training.")
    if intensive:
        df = add_intensive_features(df)
        feature_cols = _available_features(df, PHYSICS_FEATURE_COLUMNS)
    else:
        feature_cols = _available_features(df, LEGACY_PHYSICS_FEATURE_COLUMNS)
    if not feature_cols:
        raise ValueError("No physics feature columns found in fused CSV.")

    train_df = df.dropna(subset=["label"] + feature_cols)
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
    joblib.dump({"model": model, "feature_columns": feature_cols}, model_path)
    print(f"Trained RandomForest on {len(train_df)} rows, {len(feature_cols)} features")
    print(f"  features: {', '.join(feature_cols)}")
    print(f"  model:    {model_path}")
    return model_path


def train_xgboost_monotonic(
    fused_csv=DEFAULT_REFSEQ_DYNAMIC_FUSED,
    model_path=DEFAULT_XGB_MODEL_PATH,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
):
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
    print(f"Trained monotonic XGBoost on {len(train_df)} rows, {len(feature_cols)} features")
    print(f"  features: {', '.join(feature_cols)}")
    print(f"  monotone_constraints: {constraints}")
    print(f"  model:    {model_path}")
    return model_path


def _resolve_id_column(df):
    from data_engineering.cd_hit_sequence_similarity import JOIN_COLUMNS

    for col in ("record_id", *JOIN_COLUMNS):
        if col in df.columns:
            return col
    raise ValueError("No join column found for predictions output.")


def predict_thermoswitches(
    fused_csv=DEFAULT_DENOVO_FUSED,
    model_path=DEFAULT_MODEL_PATH,
    predictions_csv=DEFAULT_PREDICTIONS,
    join_column="record_id",
):
    fused_csv = resolve_path(fused_csv)
    model_path = resolve_path(model_path)
    payload = joblib.load(model_path)
    model = payload["model"]
    feature_cols = payload["feature_columns"]

    df = pd.read_csv(fused_csv)
    if any(col not in df.columns for col in feature_cols):
        df = add_intensive_features(df)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Fused CSV missing feature columns: {missing}")

    predict_df = df.dropna(subset=feature_cols)
    proba = model.predict_proba(predict_df[feature_cols])
    classes = list(model.classes_)
    if 1 in classes:
        probs = proba[:, classes.index(1)]
    else:
        # Smoke / degenerate train sets may only contain label 0.
        probs = [0.0] * len(predict_df)
    labels = model.predict(predict_df[feature_cols])

    id_col = join_column if join_column in predict_df.columns else _resolve_id_column(predict_df)
    results = predict_df[[id_col]].copy()
    results["prob_positive"] = probs
    results["predicted_label"] = labels

    predictions_csv = resolve_path(predictions_csv)
    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(predictions_csv, index=False)
    print(f"Wrote {len(results)} predictions to {predictions_csv}")
    return predictions_csv


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Train or run thermoswitch classifiers (RF / monotonic XGBoost)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train RF on labeled fused_features.csv")
    train_parser.add_argument("--fused-csv", default=DEFAULT_TRAINING_FUSED)
    train_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    train_parser.add_argument(
        "--legacy-features",
        action="store_true",
        help="Use raw MFE + absolute stem/loop features instead of intensive set.",
    )

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
    predict_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--predictions-csv", default=DEFAULT_PREDICTIONS)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "train":
        train_random_forest(
            fused_csv=args.fused_csv,
            model_path=args.model_path,
            intensive=not args.legacy_features,
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
        )


if __name__ == "__main__":
    main()
