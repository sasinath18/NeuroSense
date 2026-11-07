
"""
NeuroSense - Stress Model Training and Evaluation

Input:
    data/stress_dataset.csv

Output:
    ml_model/neurosense_stress_model.joblib
    ml_model/model_evaluation.json

IMPORTANT:
    Use genuine labelled physiological data.
    Do not train or report accuracy using fabricated sample data.
"""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    train_test_split,
)
from sklearn.pipeline import Pipeline


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "stress_dataset.csv"

MODEL_PATH = (
    ROOT_DIR / "ml_model" /
    "neurosense_stress_model.joblib"
)

REPORT_PATH = (
    ROOT_DIR / "ml_model" /
    "model_evaluation.json"
)


# ============================================================
# 2. MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "heart_rate_bpm",
    "hrv_ms",
    "gsr_raw",
    "skin_temperature_c",
    "heart_rate_delta",
    "hrv_delta",
    "gsr_delta",
    "skin_temperature_delta",
]

RAW_COLUMNS = [
    "heart_rate_bpm",
    "hrv_ms",
    "gsr_raw",
    "skin_temperature_c",
]

BASELINE_COLUMNS = [
    "baseline_heart_rate_bpm",
    "baseline_hrv_ms",
    "baseline_gsr_raw",
    "baseline_skin_temperature_c",
]

REQUIRED_COLUMNS = (
    RAW_COLUMNS
    + BASELINE_COLUMNS
    + ["stress_label"]
)


# ============================================================
# 3. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"\nDataset not found: {DATA_PATH}\n"
            "Add a genuine labelled CSV dataset before training."
        )

    df = pd.read_csv(DATA_PATH)

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if len(df) < 20:
        raise ValueError(
            "At least 20 labelled records are required "
            "for this initial training workflow. "
            "This does not guarantee a reliable model."
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=["stress_label"]
    ).copy()

    if df.empty:
        raise ValueError(
            "No labelled records found."
        )

    if df["stress_label"].nunique() < 2:
        raise ValueError(
            "Dataset must contain at least two stress classes."
        )

    print("\nDataset loaded successfully.")
    print(f"Records: {len(df)}")
    print(
        "Classes:",
        df["stress_label"].value_counts().to_dict()
    )

    return df


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

def prepare_features(df):

    X = pd.DataFrame(index=df.index)

    for column in RAW_COLUMNS:
        X[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    X["heart_rate_delta"] = (
        X["heart_rate_bpm"]
        - pd.to_numeric(
            df["baseline_heart_rate_bpm"],
            errors="coerce"
        )
    )

    X["hrv_delta"] = (
        X["hrv_ms"]
        - pd.to_numeric(
            df["baseline_hrv_ms"],
            errors="coerce"
        )
    )

    X["gsr_delta"] = (
        X["gsr_raw"]
        - pd.to_numeric(
            df["baseline_gsr_raw"],
            errors="coerce"
        )
    )

    X["skin_temperature_delta"] = (
        X["skin_temperature_c"]
        - pd.to_numeric(
            df["baseline_skin_temperature_c"],
            errors="coerce"
        )
    )

    X = X[FEATURE_COLUMNS]

    if X.dropna(how="all").empty:
        raise ValueError(
            "All physiological features are missing."
        )

    if X.isna().all(axis=0).any():
        empty_columns = X.columns[
            X.isna().all(axis=0)
        ].tolist()

        raise ValueError(
            f"Entire feature columns are missing: {empty_columns}"
        )

    return X


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

def split_dataset(X, y, df):

    # Prefer participant-level split to reduce data leakage.
    if (
        "participant_id" in df.columns
        and df["participant_id"].nunique() >= 2
    ):

        groups = df["participant_id"].fillna(
            "unknown-participant"
        )

        splitter = GroupShuffleSplit(
            n_splits=100,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        for train_idx, test_idx in splitter.split(
            X, y, groups
        ):

            if (
                y.iloc[train_idx].nunique() >= 2
                and y.iloc[test_idx].nunique() >= 2
            ):

                print(
                    "\nParticipant-level split selected."
                )

                return (
                    X.iloc[train_idx],
                    X.iloc[test_idx],
                    y.iloc[train_idx],
                    y.iloc[test_idx]
                )

        raise ValueError(
            "Could not create a participant-level split "
            "containing all required classes. "
            "Collect more participants or labelled data."
        )

    # Fallback: record-level split.
    print(
        "\nWARNING: participant_id is unavailable "
        "or contains only one participant."
    )

    print(
        "Using stratified record-level split. "
        "This is weaker validation and may overestimate performance."
    )

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


# ============================================================
# 6. MODEL PIPELINE
# ============================================================

def create_model():

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ])


# ============================================================
# 7. TRAIN MODEL
# ============================================================

def train_model():

    df = load_dataset()

    X = prepare_features(df)

    y = df["stress_label"]

    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        df
    )

    print("\nTraining NeuroSense stress classifier...")

    model = create_model()

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # ========================================================
    # 8. MODEL EVALUATION
    # ========================================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n========== MODEL EVALUATION ==========")

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1: {macro_f1:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")

    print(matrix)

    # ========================================================
    # 9. SAVE MODEL
    # ========================================================

    joblib.dump(
        model,
        MODEL_PATH
    )

    print(
        f"\nModel saved: {MODEL_PATH}"
    )

    # ========================================================
    # 10. SAVE EVALUATION REPORT
    # ========================================================

    results = {
        "project": "NeuroSense",

        "model": "RandomForestClassifier",

        "dataset": str(DATA_PATH),

        "total_records": int(len(df)),

        "training_records": int(len(X_train)),

        "testing_records": int(len(X_test)),

        "participant_level_split": (
            "participant_id" in df.columns
            and df["participant_id"].nunique() >= 2
        ),

        "features": FEATURE_COLUMNS,

        "classes": [
            str(value)
            for value in model.classes_
        ],

        "accuracy": float(accuracy),

        "macro_f1": float(macro_f1),

        "confusion_matrix": matrix.tolist(),

        "classification_report": report,

        "evaluation_note": (
            "Performance on this test split only. "
            "Independent validation is still required."
        )
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print(
        f"Evaluation report saved: {REPORT_PATH}"
    )

    print(
        "\nTraining and evaluation finished."
    )


# ============================================================
# 11. MAIN
# ============================================================

if __name__ == "__main__":

    try:
        train_model()

    except Exception as error:

        print(
            "\nMODEL TRAINING FAILED:"
        )

        print(error)

        sys.exit(1)
