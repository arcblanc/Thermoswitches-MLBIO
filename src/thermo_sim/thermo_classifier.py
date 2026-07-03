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
DEFAULT_TRAINING_FUSED = "data/processed/fused_features.csv"
DEFAULT_DENOVO_FUSED = "data/processed/denovo_fused_features.csv"
DEFAULT_PREDICTIONS = "data/processed/denovo_predictions.csv"

PHYSICS_FEATURE_COLUMNS = [
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


def _available_features(df):
    return [col for col in PHYSICS_FEATURE_COLUMNS if col in df.columns]


def train_random_forest(
    fused_csv=DEFAULT_TRAINING_FUSED,
    model_path=DEFAULT_MODEL_PATH,
    n_estimators=200,
    random_state=42,
):
    fused_csv = resolve_path(fused_csv)
    df = pd.read_csv(fused_csv)
    if "label" not in df.columns:
        raise ValueError(f"{fused_csv} must contain a label column for training.")
    feature_cols = _available_features(df)
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
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Fused CSV missing feature columns: {missing}")

    predict_df = df.dropna(subset=feature_cols)
    probs = model.predict_proba(predict_df[feature_cols])[:, 1]
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
    parser = argparse.ArgumentParser(description="Train or run Random Forest thermoswitch classifier.")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="Train RF on labeled fused_features.csv")
    train_parser.add_argument("--fused-csv", default=DEFAULT_TRAINING_FUSED)
    train_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)

    predict_parser = sub.add_parser("predict", help="Predict on de novo fused features")
    predict_parser.add_argument("--fused-csv", default=DEFAULT_DENOVO_FUSED)
    predict_parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    predict_parser.add_argument("--predictions-csv", default=DEFAULT_PREDICTIONS)
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "train":
        train_random_forest(fused_csv=args.fused_csv, model_path=args.model_path)
    elif args.command == "predict":
        predict_thermoswitches(
            fused_csv=args.fused_csv,
            model_path=args.model_path,
            predictions_csv=args.predictions_csv,
        )


if __name__ == "__main__":
    main()
