#!/usr/bin/env python3
"""Train NeuroSense's camera-only facial blendshape stress classifier.

Only MediaPipe Face Landmarker blendshapes are used. Wearable signals are not
loaded. Video is decoded locally and is never uploaded by this script.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT_DIR / "data" / "empathicschool"
DEFAULT_MODEL = ROOT_DIR / "ml_model" / "face_stress_model.joblib"
DEFAULT_REPORT = ROOT_DIR / "ml_model" / "face_stress_evaluation.json"
DEFAULT_LANDMARKER = ROOT_DIR / "ml_model" / "assets" / "face_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

WINDOW_SECONDS = 20.0
STEP_SECONDS = 5.0
DEFAULT_SAMPLE_FPS = 5.0
MIN_WINDOW_COVERAGE = 0.60
LABEL_NAMES = {0: "no_stress", 1: "low_stress", 2: "high_stress"}


def parse_stress_label(value: Any) -> int | None:
    """Map only the dataset's documented 0/1/2 labels; skip unknown values."""
    if value is None or pd.isna(value):
        return None
    token = re.sub(r"[_\s-]+", " ", str(value).strip().lower())
    try:
        number = float(token)
        if number.is_integer() and int(number) in LABEL_NAMES:
            return int(number)
    except (TypeError, ValueError):
        pass
    return {
        "no stress": 0, "nostress": 0, "not stressed": 0,
        "low stress": 1, "low": 1,
        "high stress": 2, "high": 2,
    }.get(token)


def read_stress_labels(path: Path) -> list[int | None]:
    """Choose the CSV column with the largest count of recognized labels."""
    try:
        frame = pd.read_csv(path, header=None, dtype=str, sep=None, engine="python")
    except Exception as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"Stress label file is empty: {path}")

    candidates = []
    for column in frame.columns:
        parsed = [parse_stress_label(value) for value in frame[column].tolist()]
        valid_count = sum(value is not None for value in parsed)
        if valid_count:
            candidates.append((valid_count, parsed))
    if not candidates:
        raise ValueError(
            f"No documented 0/1/2 labels found in {path}. "
            "Inspect the provider's CSV schema rather than guessing a mapping."
        )
    return max(candidates, key=lambda item: item[0])[1]


def build_label_intervals(
    labels: list[int | None], duration_seconds: float
) -> list[tuple[float, float, int | None]]:
    """Use the paper's 90/120/120/120/90-second pattern when five labels exist."""
    if not labels:
        return []
    if len(labels) == 5 and duration_seconds >= 480:
        boundaries = [0.0, 90.0, 210.0, 330.0, 450.0, float(duration_seconds)]
    else:
        boundaries = np.linspace(
            0.0, float(duration_seconds), num=len(labels) + 1
        ).tolist()
    return [
        (float(boundaries[i]), float(boundaries[i + 1]), labels[i])
        for i in range(len(labels))
    ]


def label_for_window(
    start: float,
    end: float,
    intervals: list[tuple[float, float, int | None]],
) -> int | None:
    """Return a label only if the full feature window lies in one label interval."""
    for left, right, label in intervals:
        if start >= left - 1e-6 and end <= right + 1e-6:
            return label
    return None


def find_subject_task(video_path: Path, root: Path) -> tuple[str, str] | None:
    """Read subject/task IDs from folder names, not video filenames."""
    subject_id = None
    task_id = None
    for parent in (video_path.parent, *video_path.parents):
        if subject_id is None and re.fullmatch(r"S\d+", parent.name, re.I):
            subject_id = parent.name.upper()
        if task_id is None and re.fullmatch(r"T\d+", parent.name, re.I):
            task_id = parent.name.upper()
        if parent.resolve() == root.resolve():
            break
    return (subject_id, task_id) if subject_id and task_id else None


def find_label_file(task_dir: Path) -> Path | None:
    matches = [
        p for p in task_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
        and "stress" in p.name.lower() and "level" in p.name.lower()
    ]
    if matches:
        return sorted(matches)[0]
    parent = task_dir.parent
    matches = [
        p for p in parent.iterdir()
        if p.is_file() and p.suffix.lower() == ".csv"
        and "stress" in p.name.lower() and "level" in p.name.lower()
        and task_dir.name.lower() in p.name.lower()
    ]
    return sorted(matches)[0] if matches else None


def discover_video_tasks(root: Path) -> list[tuple[str, str, Path, Path]]:
    """Find released video tasks that also have their own stress-label file."""
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {root}. Request EmpathicSchool access, "
            "then extract the authorized copy under data/empathicschool/."
        )
    tasks = []
    for video in sorted(root.rglob("*.mp4")):
        identity = find_subject_task(video, root)
        if identity is None:
            continue
        label_file = find_label_file(video.parent)
        if label_file is not None:
            tasks.append((identity[0], identity[1], video, label_file))
    if not tasks:
        raise FileNotFoundError(
            f"No eligible video/label pairs found under {root}. Expected folders "
            "such as S7/T2/V.mp4 and S7/T2/Stress_level.csv. Only consented videos "
            "included in the authorized dataset release can be used."
        )
    return tasks


def ensure_landmarker_model(path: Path) -> Path:
    """Download the same Face Landmarker task asset used by the browser client."""
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    print(f"Downloading MediaPipe Face Landmarker asset to {path}")
    try:
        urllib.request.urlretrieve(LANDMARKER_URL, temporary)
        if temporary.stat().st_size <= 1_000_000:
            raise RuntimeError("Downloaded Face Landmarker asset is unexpectedly small.")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def create_landmarker(model_path: Path):
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as exc:
        raise RuntimeError(
            "MediaPipe Tasks API is unavailable. Install the project's requirements."
        ) from exc
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp, vision.FaceLandmarker.create_from_options(options)


def extract_blendshape_samples(
    video_path: Path, model_path: Path, sample_fps: float
) -> tuple[list[tuple[float, dict[str, float]]], float]:
    """Extract timestamps and blendshape scores without saving decoded frames."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if source_fps <= 0 or frame_count <= 0:
        capture.release()
        raise ValueError(f"Invalid frame rate or frame count in {video_path}")
    duration = frame_count / source_fps
    stride = max(1, int(round(source_fps / sample_fps)))
    mp, landmarker = create_landmarker(model_path)
    samples: list[tuple[float, dict[str, float]]] = []
    frame_index = 0
    last_timestamp_ms = -1
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % stride == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int(round(frame_index * 1000.0 / source_fps))
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)
                if result.face_blendshapes:
                    categories = result.face_blendshapes[0]
                    scores = {
                        str(item.category_name): float(item.score)
                        for item in categories if item.category_name
                    }
                    if scores:
                        samples.append((timestamp_ms / 1000.0, scores))
            frame_index += 1
    finally:
        capture.release()
        landmarker.close()
    return samples, duration


def build_feature_windows(
    samples: list[tuple[float, dict[str, float]]],
    duration_seconds: float,
    labels: list[int | None],
    subject_id: str,
    task_id: str,
    window_seconds: float = WINDOW_SECONDS,
    step_seconds: float = STEP_SECONDS,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
) -> list[dict[str, Any]]:
    """Compute mean/std of each blendshape over labelled 20-second windows."""
    if not samples:
        return []
    if window_seconds <= 0 or step_seconds <= 0:
        raise ValueError("Window and step durations must be positive.")

    names = sorted({name for _, values in samples for name in values})
    times = np.asarray([timestamp for timestamp, _ in samples], dtype=float)
    intervals = build_label_intervals(labels, duration_seconds)
    minimum_samples = max(5, int(window_seconds * sample_fps * MIN_WINDOW_COVERAGE))
    rows = []

    for start in np.arange(
        0.0,
        max(0.0, duration_seconds - window_seconds) + 1e-9,
        step_seconds,
    ):
        end = float(start + window_seconds)
        label = label_for_window(float(start), end, intervals)
        if label is None:
            continue
        mask = (times >= start) & (times < end)
        if int(mask.sum()) < minimum_samples:
            continue

        row: dict[str, Any] = {
            "participant_id": subject_id,
            "task_id": task_id,
            "window_start_seconds": round(float(start), 3),
            "stress_label": int(label),
        }
        selected = [score_map for keep, (_, score_map) in zip(mask, samples) if keep]
        for name in names:
            values = [scores[name] for scores in selected if name in scores]
            row[f"blendshape_{name}_mean"] = (
                float(np.mean(values)) if values else np.nan
            )
            row[f"blendshape_{name}_std"] = (
                float(np.std(values)) if values else np.nan
            )
        rows.append(row)
    return rows


def create_classifier() -> Pipeline:
    """Use a conservative, class-weighted tree model for this small dataset."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def evaluate_loso(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, Any]:
    """Evaluate using participant-held-out folds, never random window splits."""
    x = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = frame["stress_label"].astype(int)
    groups = frame["participant_id"].astype(str)
    if groups.nunique() < 4:
        raise ValueError(
            f"Only {groups.nunique()} participants are usable. At least four are "
            "required for exploratory LOSO evaluation; more are strongly preferred."
        )
    if y.nunique() < 2:
        raise ValueError("At least two stress classes are required.")

    predictions = np.full(len(frame), -1, dtype=int)
    folds = []
    for train_idx, test_idx in LeaveOneGroupOut().split(x, y, groups):
        y_train = y.iloc[train_idx]
        if y_train.nunique() < 2:
            continue
        model = create_classifier()
        model.fit(x.iloc[train_idx], y_train)
        fold_predictions = model.predict(x.iloc[test_idx])
        predictions[test_idx] = fold_predictions
        folds.append({
            "held_out_participant": str(groups.iloc[test_idx[0]]),
            "train_windows": int(len(train_idx)),
            "test_windows": int(len(test_idx)),
            "macro_f1": float(f1_score(
                y.iloc[test_idx], fold_predictions,
                average="macro", zero_division=0,
            )),
        })

    valid = predictions >= 0
    if not valid.any():
        raise ValueError("LOSO validation produced no valid held-out predictions.")
    y_true = y.iloc[np.flatnonzero(valid)]
    y_pred = predictions[valid]
    ordered_labels = [0, 1, 2]
    return {
        "validation": "Leave-One-Subject-Out (LOSO)",
        "participants": int(groups.nunique()),
        "windows": int(len(frame)),
        "evaluated_windows": int(valid.sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=ordered_labels,
            average="macro", zero_division=0,
        )),
        "confusion_matrix_labels": ordered_labels,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=ordered_labels
        ).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=ordered_labels,
            target_names=[LABEL_NAMES[index] for index in ordered_labels],
            zero_division=0, output_dict=True,
        ),
        "folds": folds,
        "not_evaluated_windows": int((~valid).sum()),
    }


def run(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    model_path = args.model_path.resolve()
    report_path = args.report_path.resolve()
    landmarker_path = ensure_landmarker_model(args.landmarker_model.resolve())
    tasks = discover_video_tasks(dataset_root)
    print(f"Found {len(tasks)} labelled video tasks.")

    all_rows: list[dict[str, Any]] = []
    failures = []
    for index, (subject, task, video, label_path) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {subject} {task}: {video.name}")
        try:
            labels = read_stress_labels(label_path)
            samples, duration = extract_blendshape_samples(
                video, landmarker_path, args.sample_fps
            )
            rows = build_feature_windows(
                samples, duration, labels, subject, task,
                args.window_seconds, args.step_seconds, args.sample_fps,
            )
            all_rows.extend(rows)
            print(f"  face samples: {len(samples)}; labelled windows: {len(rows)}")
        except Exception as exc:
            failures.append({
                "participant_id": subject, "task_id": task, "error": str(exc)
            })
            print(f"  skipped: {exc}", file=sys.stderr)

    if not all_rows:
        raise ValueError(
            "No labelled windows were created. Check access, consented video files, "
            "label schema and folder layout."
        )

    frame = pd.DataFrame(all_rows).replace([np.inf, -np.inf], np.nan)
    metadata = {"participant_id", "task_id", "window_start_seconds", "stress_label"}
    feature_columns = [column for column in frame.columns if column not in metadata]
    frame = frame.dropna(subset=feature_columns, how="all")
    if len(frame) < 40:
        raise ValueError(
            f"Only {len(frame)} labelled windows were produced; collect more "
            "released participant videos before training."
        )

    output_csv = args.output_csv.resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    print(f"Saved local feature windows: {output_csv}")

    evaluation = evaluate_loso(frame, feature_columns)
    final_model = create_classifier()
    final_model.fit(
        frame[feature_columns].replace([np.inf, -np.inf], np.nan),
        frame["stress_label"].astype(int),
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_columns": feature_columns,
        "class_names": LABEL_NAMES,
        "feature_extractor": "MediaPipe Face Landmarker blendshapes",
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "dataset": "EmpathicSchool",
        "training_note": "Research prototype; camera-only; not a medical model.",
    }, model_path)

    report = {
        "project": "NeuroSense",
        "model": "RandomForestClassifier",
        "input_modality": "camera-only MediaPipe facial blendshapes",
        "feature_columns": feature_columns,
        "class_names": LABEL_NAMES,
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sample_fps": args.sample_fps,
        "dataset_name": "EmpathicSchool",
        "dataset_source": "https://doi.org/10.5281/zenodo.7372497",
        "dataset_rows": int(len(frame)),
        "participant_count": int(frame["participant_id"].nunique()),
        "task_count": int(frame[["participant_id", "task_id"]].drop_duplicates().shape[0]),
        "class_distribution": {
            str(key): int(value)
            for key, value in frame["stress_label"].value_counts().sort_index().items()
        },
        "evaluation": evaluation,
        "failed_tasks": failures,
        "important_limitations": [
            "The dataset has limited participant diversity and laboratory task conditions.",
            "Facial movement is not a direct or universal measure of psychological stress.",
            "This model uses facial blendshapes only, not HR, BVP, EDA, temperature or wearable data.",
            "Published emotion-only results from EmpathicSchool are not the performance of this separate MediaPipe model.",
            "Independent testing on unseen participants and real-world camera conditions is required.",
            "Not intended for diagnosis, treatment, screening or clinical decisions.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== FACE-ONLY MODEL EVALUATION ==========")
    print(f"Participants: {evaluation['participants']}")
    print(f"Evaluated windows: {evaluation['evaluated_windows']}")
    print(f"LOSO accuracy: {evaluation['accuracy']:.3f}")
    print(f"LOSO balanced accuracy: {evaluation['balanced_accuracy']:.3f}")
    print(f"LOSO macro F1: {evaluation['macro_f1']:.3f}")
    print(f"Model saved: {model_path}")
    print(f"Report saved: {report_path}")
    print("Do not enable a user-facing stress score before independent validation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NeuroSense's camera-only facial stress model."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--landmarker-model", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument(
        "--output-csv", type=Path,
        default=ROOT_DIR / "data" / "face_stress_windows.csv",
    )
    parser.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--step-seconds", type=float, default=STEP_SECONDS)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except Exception as error:
        print(f"\nFACE-ONLY TRAINING FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
