#!/usr/bin/env python3
"""Train an offline, face-feature-only stress regression model from EmoWork.

The script reads only:
  * CSV/VIDEO/<participant>/<condition>.csv (facial landmark-derived features)
  * LABEL/<participant>/<condition>/stress.csv (self-reported stress)

It deliberately ignores SENSORS, AUDIO, pupil, EDA/GSR, BVP, temperature,
heart-rate, EEG, and all other physiological/acoustic modalities. The dataset
is access-controlled; obtain and follow its Data Usage Agreement before use.
No participant data is uploaded by this script.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "emowork"
DEFAULT_MODEL = ROOT_DIR / "ml_model" / "emowork_face_stress_rf.joblib"
DEFAULT_REPORT = ROOT_DIR / "ml_model" / "emowork_face_stress_evaluation.json"
DEFAULT_WINDOWS = ROOT_DIR / "data" / "emowork_face_stress_windows.csv"

TARGET_MIN = 1.0
TARGET_MAX = 20.0
DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_STEP_SECONDS = 2.5
DEFAULT_LABEL_TOLERANCE_MS = 250.0
DEFAULT_MIN_FRAME_RATIO = 0.5


def normalize_column_name(value: Any) -> str:
    """Normalize a provider column name without guessing its semantic meaning."""
    token = str(value).strip().lower()
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def find_column(columns: list[str], accepted_names: set[str], description: str) -> str:
    normalized = {normalize_column_name(column): column for column in columns}
    matches = [normalized[name] for name in accepted_names if name in normalized]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {description} column from "
            f"{sorted(accepted_names)}; found {matches}. "
            f"Available columns: {columns}"
        )
    return matches[0]


def select_face_feature_columns(frame: pd.DataFrame) -> tuple[str, list[str]]:
    """Select only the documented face-landmark feature contract.

    Required numeric fields are head rotation/translation, total translation,
    blink, gaze and left/right lip-corner movement. error_type is used only to
    filter failed landmark frames; it is never a model feature.
    """
    original = {normalize_column_name(column): str(column) for column in frame.columns}

    required = [
        "rotation_x", "rotation_y", "rotation_z",
        "translation_x", "translation_y", "translation_z",
        "total_translation", "blink", "gaze_x", "gaze_y",
    ]
    missing = [name for name in required if name not in original]
    if missing:
        raise ValueError(
            "EmoWork video CSV is missing documented facial columns: "
            f"{missing}. Refusing to substitute non-face data. "
            f"Available columns: {list(frame.columns)}"
        )

    lip_columns = [
        original[name]
        for name in original
        if "lip" in name and ("left" in name or "right" in name)
    ]
    has_left = any("left" in normalize_column_name(column) for column in lip_columns)
    has_right = any("right" in normalize_column_name(column) for column in lip_columns)
    if not has_left or not has_right:
        raise ValueError(
            "Video CSV must contain both left and right lip-corner movement "
            f"features; found {lip_columns}. Inspect the authorized CSV schema."
        )

    error_column = find_column(
        list(frame.columns), {"error_type"}, "face landmark detection status"
    )
    selected = [original[name] for name in required] + sorted(set(lip_columns))
    # The strict whitelist prevents accidental use of pupil, audio, or wearable data.
    forbidden = re.compile(
        r"(pupil|eda|gsr|bvp|ecg|eeg|heart|hrv|temperature|temp|audio|mfcc|sensor)",
        re.IGNORECASE,
    )
    selected = [column for column in selected if not forbidden.search(column)]
    if len(selected) < len(required) + 2:
        raise ValueError("Insufficient facial feature columns after modality filtering.")
    return error_column, selected


def timestamp_to_seconds(values: pd.Series, name: str) -> pd.Series:
    """Convert ISO/UTC or numeric provider timestamps into comparable seconds."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        valid = numeric.dropna()
        if valid.empty:
            raise ValueError(f"{name} contains no valid timestamps.")
        median = float(valid.abs().median())
        if median >= 1e17:      # Unix nanoseconds
            divisor = 1e9
        elif median >= 1e11:    # Unix milliseconds
            divisor = 1e3
        elif median >= 1e8:     # Unix seconds
            divisor = 1.0
        elif median >= 1e4:     # relative milliseconds
            divisor = 1e3
        else:                   # relative seconds
            divisor = 1.0
        return numeric.astype(float) / divisor

    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.notna().mean() < 0.95:
        raise ValueError(
            f"{name} timestamp column is not consistently parseable as ISO/UTC "
            "or numeric time values."
        )
    return parsed.astype("int64").astype(float) / 1e9


def identify_participant_condition(path: Path, root: Path) -> tuple[str, str] | None:
    """Extract participant and condition from LABEL/<P>/<C>/stress.csv layout."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        return None
    parts = list(relative.parts)
    condition_index = next(
        (i for i, part in enumerate(parts[:-1])
         if re.fullmatch(r"c[1-3]", Path(part).stem, re.IGNORECASE)),
        None,
    )
    if condition_index is None or condition_index == 0:
        return None
    condition = Path(parts[condition_index]).stem.lower()
    participant_part = parts[condition_index - 1]
    digits = re.sub(r"[^0-9]", "", participant_part)
    if not digits:
        return None
    return str(int(digits)), condition


def participant_matches(path: Path, participant_id: str) -> bool:
    expected = str(int(participant_id))
    for part in path.parts:
        digits = re.sub(r"[^0-9]", "", part)
        if digits and str(int(digits)) == expected:
            return True
    return False


def condition_matches(path: Path, condition: str) -> bool:
    expected = condition.lower()
    return any(Path(part).stem.lower() == expected for part in path.parts)


def discover_dataset_pairs(root: Path) -> list[tuple[str, str, Path, Path]]:
    """Find matching facial-video-feature and stress-label CSVs.

    The expected hierarchy is LABEL/<participant>/<condition>/stress.csv and
    CSV/VIDEO/<participant>/<condition>.csv (or a nested CSV in that folder).
    Ambiguous pairs fail closed instead of selecting a wrong participant file.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"EmoWork dataset folder not found: {root}. Request access from the "
            "dataset provider, accept its Data Usage Agreement, then extract "
            "the authorized archive locally."
        )

    video_files = [
        path for path in root.rglob("*.csv")
        if any(part.lower() == "video" for part in path.parts)
    ]
    pairs = []
    label_files = sorted(
        path for path in root.rglob("stress.csv")
        if any(part.lower() == "label" for part in path.parts)
    )
    for label_path in label_files:
        identity = identify_participant_condition(label_path, root)
        if identity is None:
            continue
        participant, condition = identity
        candidates = [
            path for path in video_files
            if participant_matches(path, participant)
            and condition_matches(path, condition)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one video-feature CSV for participant {participant}, "
                f"{condition}; found {len(candidates)}: {candidates}"
            )
        pairs.append((participant, condition, candidates[0], label_path))

    if not pairs:
        raise FileNotFoundError(
            f"No LABEL/<participant>/<condition>/stress.csv and "
            f"CSV/VIDEO/<participant>/<condition>.csv pairs found under {root}."
        )
    return pairs


def load_aligned_face_stress(
    video_csv: Path,
    stress_csv: Path,
    label_tolerance_ms: float = DEFAULT_LABEL_TOLERANCE_MS,
) -> tuple[pd.DataFrame, list[str]]:
    """Align face frames to nearest self-reported stress timestamp.

    Only valid facial frames (error_type == 0) and the exact stress label are
    retained. No physiological or audio files are read.
    """
    if label_tolerance_ms <= 0:
        raise ValueError("label_tolerance_ms must be greater than zero.")

    video = pd.read_csv(video_csv)
    labels = pd.read_csv(stress_csv)
    if video.empty or labels.empty:
        raise ValueError("Video feature CSV and stress label CSV must be non-empty.")

    video_time = find_column(
        list(video.columns),
        {"timestamp", "utc_timestamp", "timestamp_utc", "time"},
        "video timestamp",
    )
    label_time = find_column(
        list(labels.columns),
        {"timestamp", "utc_timestamp", "timestamp_utc", "time"},
        "stress label timestamp",
    )
    stress_column = find_column(
        list(labels.columns),
        {"stress", "stress_level", "perceived_stress"},
        "self-reported stress target",
    )
    error_column, face_columns = select_face_feature_columns(video)

    video_work = video[[video_time, error_column, *face_columns]].copy()
    label_work = labels[[label_time, stress_column]].copy()
    video_work["_time_seconds"] = timestamp_to_seconds(video_work[video_time], "video")
    label_work["_time_seconds"] = timestamp_to_seconds(label_work[label_time], "label")
    video_work["_error_code"] = pd.to_numeric(
        video_work[error_column], errors="coerce"
    )
    label_work["_stress_score"] = pd.to_numeric(
        label_work[stress_column], errors="coerce"
    )

    # Keep only successful facial landmark extraction attempts.
    video_work = video_work[video_work["_error_code"] == 0].copy()
    label_work = label_work[
        label_work["_stress_score"].between(TARGET_MIN, TARGET_MAX, inclusive="both")
    ].copy()
    if video_work.empty or label_work.empty:
        raise ValueError("No valid face frames or 1–20 self-reported stress labels.")

    for column in face_columns:
        video_work[column] = pd.to_numeric(video_work[column], errors="coerce")
    video_work = video_work.dropna(subset=face_columns, how="all")
    if video_work.empty:
        raise ValueError("No numeric facial feature values remain after validation.")

    video_work = video_work.sort_values("_time_seconds")
    label_work = label_work.sort_values("_time_seconds")
    aligned = pd.merge_asof(
        video_work,
        label_work[["_time_seconds", "_stress_score"]],
        on="_time_seconds",
        direction="nearest",
        tolerance=label_tolerance_ms / 1000.0,
    )
    aligned = aligned.dropna(subset=["_stress_score"]).copy()
    if aligned.empty:
        raise ValueError(
            "No facial frames could be aligned to stress labels within "
            f"{label_tolerance_ms} ms. Check that both streams use the same UTC clock."
        )
    return aligned, face_columns


def build_face_windows(
    aligned: pd.DataFrame,
    face_columns: list[str],
    participant_id: str,
    condition: str,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    step_seconds: float = DEFAULT_STEP_SECONDS,
    min_frame_ratio: float = DEFAULT_MIN_FRAME_RATIO,
) -> list[dict[str, Any]]:
    """Aggregate facial features and continuous stress labels by fixed windows."""
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Window and step durations must be positive.")
    if not 0 < min_frame_ratio <= 1:
        raise ValueError("min_frame_ratio must be in (0, 1].")
    if aligned.empty:
        return []

    times = aligned["_time_seconds"].to_numpy(dtype=float)
    start_time = float(np.min(times))
    end_time = float(np.max(times))
    expected_frames = max(1, int(round(window_seconds * 30.0 * min_frame_ratio)))
    rows: list[dict[str, Any]] = []

    for start in np.arange(
        start_time,
        max(start_time, end_time - window_seconds) + 1e-9,
        step_seconds,
    ):
        stop = float(start + window_seconds)
        mask = (times >= start) & (times < stop)
        selected = aligned.loc[mask]
        if len(selected) < expected_frames:
            continue
        row: dict[str, Any] = {
            "participant_id": str(participant_id),
            "condition": str(condition).lower(),
            "window_start_seconds": float(start),
            "stress_score": float(selected["_stress_score"].median()),
        }
        for column in face_columns:
            values = pd.to_numeric(selected[column], errors="coerce").dropna()
            row[f"{column}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{column}_std"] = float(values.std(ddof=0)) if not values.empty else np.nan
        rows.append(row)
    return rows


def create_regressor() -> Pipeline:
    """A modest random forest for a small, participant-grouped research dataset."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate_loso(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Evaluate continuous stress using Leave-One-Participant-Out validation."""
    x = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = frame["stress_score"].astype(float)
    groups = frame["participant_id"].astype(str)
    participant_count = int(groups.nunique())
    if participant_count < 4:
        raise ValueError(
            f"Only {participant_count} participants have usable windows; "
            "at least four are required for exploratory LOSO validation."
        )
    predictions = np.full(len(frame), np.nan, dtype=float)
    folds = []
    for train_index, test_index in LeaveOneGroupOut().split(x, y, groups):
        model = create_regressor()
        model.fit(x.iloc[train_index], y.iloc[train_index])
        predicted = model.predict(x.iloc[test_index])
        predictions[test_index] = predicted
        truth = y.iloc[test_index]
        folds.append({
            "held_out_participant": str(groups.iloc[test_index[0]]),
            "test_windows": int(len(test_index)),
            "mae": float(mean_absolute_error(truth, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
            "r2": (
                float(r2_score(truth, predicted))
                if len(test_index) >= 2 and truth.nunique() >= 2 else None
            ),
        })

    valid = np.isfinite(predictions)
    if not valid.any():
        raise ValueError("LOSO validation produced no held-out predictions.")
    truth = y.iloc[np.flatnonzero(valid)]
    predicted = predictions[valid]
    return {
        "validation": "Leave-One-Participant-Out (LOSO)",
        "participants": participant_count,
        "windows": int(len(frame)),
        "evaluated_windows": int(valid.sum()),
        "mae": float(mean_absolute_error(truth, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predicted))),
        "r2": float(r2_score(truth, predicted)) if truth.nunique() >= 2 else None,
        "folds": folds,
    }


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    pairs = discover_dataset_pairs(dataset_root)
    print(f"Found {len(pairs)} facial-video/stress-label pairs.")

    rows: list[dict[str, Any]] = []
    failures = []
    used_feature_columns: list[str] | None = None
    for index, (participant, condition, video_path, label_path) in enumerate(pairs, 1):
        print(f"[{index}/{len(pairs)}] participant={participant} condition={condition}")
        try:
            aligned, face_columns = load_aligned_face_stress(
                video_path, label_path, args.label_tolerance_ms
            )
            if used_feature_columns is None:
                used_feature_columns = face_columns
            elif face_columns != used_feature_columns:
                raise ValueError(
                    "Facial CSV feature schemas differ between participants; "
                    "normalise the authorized dataset export before training."
                )
            task_rows = build_face_windows(
                aligned,
                face_columns,
                participant,
                condition,
                args.window_seconds,
                args.step_seconds,
                args.min_frame_ratio,
            )
            rows.extend(task_rows)
            print(f"  aligned frames={len(aligned)}; windows={len(task_rows)}")
        except Exception as exc:
            failures.append({
                "participant_id": participant,
                "condition": condition,
                "error": str(exc),
            })
            print(f"  skipped: {exc}", file=sys.stderr)

    if not rows:
        raise ValueError(
            "No valid face-only stress windows were produced. Check the authorized "
            "CSV hierarchy, feature columns, UTC alignment and stress labels."
        )

    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    metadata = {
        "participant_id", "condition", "window_start_seconds", "stress_score"
    }
    feature_columns = [column for column in frame.columns if column not in metadata]
    frame = frame.dropna(subset=feature_columns, how="all")
    if len(frame) < 50:
        raise ValueError(
            f"Only {len(frame)} face-only windows were produced; at least 50 are "
            "required for this exploratory run."
        )

    evaluation = evaluate_loso(frame, feature_columns)
    model = create_regressor()
    model.fit(
        frame[feature_columns].replace([np.inf, -np.inf], np.nan),
        frame["stress_score"].astype(float),
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": feature_columns,
        "target": "self-reported stress score (1–20)",
        "input_modality": "facial landmark-derived features only",
        "dataset": "EmoWork",
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "label_tolerance_ms": args.label_tolerance_ms,
        "limitations": [
            "Research prototype, not a medical measurement or diagnosis.",
            "Training context is simulated call-centre interactions.",
            "Participant-held-out validation is not a substitute for independent student-context validation.",
            "Do not enable a user-facing stress score without matching live facial features and independent validation.",
        ],
    }, args.model_path)

    report = {
        "project": "NeuroSense",
        "model": "RandomForestRegressor",
        "input_modality": "facial landmark-derived features only",
        "dataset": "EmoWork",
        "dataset_doi": "10.5281/zenodo.17513978",
        "target": "self-reported stress score (1–20)",
        "participants": evaluation["participants"],
        "windows": evaluation["windows"],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "class_or_score_distribution": {
            "minimum": float(frame["stress_score"].min()),
            "median": float(frame["stress_score"].median()),
            "maximum": float(frame["stress_score"].max()),
        },
        "evaluation": evaluation,
        "failed_pairs": failures,
        "important_limitations": [
            "This training path uses only video-derived facial features and self-reported stress labels.",
            "Physiological, wearable, pupil and audio data are not model inputs.",
            "The dataset context is simulated call-centre interactions, not student study sessions.",
            "Published EmoWork multimodal performance is not the performance of this face-only model.",
            "New-user and student-context validation are required before any user-facing score.",
            "The output is a research estimate, not diagnosis, screening or treatment advice.",
        ],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== EMOWORK FACE-ONLY STRESS EVALUATION ==========")
    print(f"Participants: {evaluation['participants']}")
    print(f"Evaluated windows: {evaluation['evaluated_windows']}")
    print(f"LOSO MAE: {evaluation['mae']:.3f}")
    print(f"LOSO RMSE: {evaluation['rmse']:.3f}")
    print(f"LOSO R²: {evaluation['r2'] if evaluation['r2'] is not None else 'n/a'}")
    print(f"Model saved: {args.model_path}")
    print(f"Report saved: {args.report_path}")
    print("This offline artifact is not enabled in NeuroSense FaceScan.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an offline face-feature-only EmoWork stress regressor."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--window-seconds", type=float, default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=float, default=DEFAULT_STEP_SECONDS)
    parser.add_argument(
        "--label-tolerance-ms", type=float, default=DEFAULT_LABEL_TOLERANCE_MS
    )
    parser.add_argument(
        "--min-frame-ratio", type=float, default=DEFAULT_MIN_FRAME_RATIO
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"\nEMOWORK FACE-ONLY TRAINING FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
