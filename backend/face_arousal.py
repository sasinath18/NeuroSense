"""Experimental face-only felt-arousal inference for the NeuroSense webcam scan.

The model was trained from AFFEC OpenFace video-phase statistics to predict
self-reported felt arousal (f_emotion_a, 1-9). It is NOT a stress detector.
Each live request uses a short, explicitly consented webcam clip; files and
OpenFace CSV output are kept in a TemporaryDirectory and deleted on return.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("neurosense.face_arousal")

WINDOW_SECONDS = 3.8
MIN_VIDEO_SECONDS = 3.7
MAX_VIDEO_SECONDS = 5.0
MIN_VALID_FRAMES = 40
MIN_CONFIDENCE = 0.8
MAX_VIDEO_BYTES = 12 * 1024 * 1024
OPENFACE_TIMEOUT_SECONDS = 90

RAW_FEATURES = (
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r",
    "AU07_r", "AU09_r", "AU10_r", "AU12_r", "AU14_r",
    "AU15_r", "AU17_r", "AU20_r", "AU23_r", "AU25_r",
    "AU26_r", "AU45_r",
    "gaze_angle_x", "gaze_angle_y",
    "pose_Rx", "pose_Ry", "pose_Rz",
)

# Match the Colab training table's exact insertion order:
# for each raw feature, its mean is followed by its population std.
FEATURE_NAMES = tuple(
    name + suffix
    for name in RAW_FEATURES
    for suffix in ("__mean", "__std")
)


def load_face_arousal_model(
    model_path: Path,
    model_url: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load the user-trained arousal bundle, optionally downloading from Drive."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file() and model_url:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "gdown is required to download the face-arousal model."
            ) from exc

        temporary_path = model_path.with_suffix(model_path.suffix + ".download")
        try:
            downloaded = gdown.download(
                url=model_url,
                output=str(temporary_path),
                quiet=False,
                fuzzy=True,
            )
            if not downloaded or not temporary_path.is_file():
                raise RuntimeError(
                    "Could not download the face-arousal model. "
                    "Share the model file as Anyone with the link - Viewer."
                )
            if temporary_path.stat().st_size < 10_000:
                raise RuntimeError(
                    "Downloaded face-arousal model is unexpectedly small."
                )
            if expected_sha256:
                digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
                if digest.lower() != expected_sha256.strip().lower():
                    raise RuntimeError(
                        "Face-arousal model SHA-256 does not match."
                    )
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face-arousal model not found at {model_path}. "
            "Configure NEUROSENSE_FACE_AROUSAL_MODEL_URL in Render."
        )

    if model_path.stat().st_size < 10_000:
        raise RuntimeError("Face-arousal model file is unexpectedly small.")

    if expected_sha256:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest.lower() != expected_sha256.strip().lower():
            raise RuntimeError("Face-arousal model SHA-256 does not match.")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise RuntimeError("Face-arousal artifact is not a model bundle.")

    saved_features = bundle.get("feature_columns")
    if not isinstance(saved_features, (list, tuple)):
        raise RuntimeError("Face-arousal artifact has no feature_columns schema.")

    if tuple(saved_features) != FEATURE_NAMES:
        raise RuntimeError(
            "Face-arousal artifact feature schema does not match the live "
            "OpenFace extractor. Refusing mismatched inference."
        )

    if bundle.get("target") != "f_emotion_a":
        raise RuntimeError(
            "The loaded artifact target is not the expected felt-arousal label."
        )

    if len(saved_features) != 44:
        raise RuntimeError(
            f"Expected 44 face features; artifact contains {len(saved_features)}."
        )

    return bundle


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_live_feature_frame(
    rows: Iterable[Mapping[str, Any]],
    feature_columns: Iterable[str] = FEATURE_NAMES,
) -> tuple[pd.DataFrame | None, int, str | None]:
    """Aggregate OpenFace AU/gaze/pose rows using the offline feature contract."""
    valid: list[list[float]] = []

    for row in rows:
        timestamp = _as_float(row.get("timestamp"))
        success = _as_float(row.get("success"))
        confidence = _as_float(row.get("confidence"))

        if (
            not np.isfinite(timestamp)
            or timestamp < 0
            or timestamp > WINDOW_SECONDS + 0.15
        ):
            continue
        if success != 1:
            continue
        if not np.isfinite(confidence) or confidence < MIN_CONFIDENCE:
            continue

        valid.append([_as_float(row.get(name)) for name in RAW_FEATURES])

    valid_count = len(valid)
    if valid_count < MIN_VALID_FRAMES:
        return (
            None,
            valid_count,
            f"Only {valid_count} valid face frames; at least "
            f"{MIN_VALID_FRAMES} are required.",
        )

    values = np.asarray(valid, dtype=np.float64)
    computed: dict[str, float] = {}

    for index, raw_name in enumerate(RAW_FEATURES):
        column = values[:, index]
        finite = column[np.isfinite(column)]
        computed[raw_name + "__mean"] = (
            float(np.mean(finite)) if finite.size else float("nan")
        )
        computed[raw_name + "__std"] = (
            float(np.std(finite, ddof=0)) if finite.size else float("nan")
        )

    columns = list(feature_columns)
    if len(columns) != 44 or set(columns) != set(FEATURE_NAMES):
        raise RuntimeError("Invalid face-arousal feature schema.")

    frame = pd.DataFrame(
        [[computed[name] for name in columns]],
        columns=columns,
        dtype=np.float64,
    )
    return frame, valid_count, None


def predict_from_openface_rows(
    bundle: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    feature_columns = bundle["feature_columns"]
    frame, valid_count, reason = build_live_feature_frame(
        rows, feature_columns
    )
    if frame is None:
        return {
            "status": "no_prediction",
            "reason": reason,
            "valid_frames": valid_count,
            "required_frames": MIN_VALID_FRAMES,
        }

    model = bundle["model"]
    prediction = float(np.asarray(model.predict(frame)).reshape(-1)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("The face-arousal model returned a non-finite value.")

    # The training target is a bounded self-report scale from 1 through 9.
    prediction = float(np.clip(prediction, 1.0, 9.0))

    return {
        "status": "prediction_available",
        "estimated_arousal": prediction,
        "scale_min": 1,
        "scale_max": 9,
        "valid_frames": valid_count,
        "features_used": 44,
        "window_seconds": WINDOW_SECONDS,
        "target": "f_emotion_a",
        "label": "felt_arousal",
        "research_only": True,
        "stress_inference": False,
    }


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return float(result.stdout.strip())


def predict_from_video_bytes(
    bundle: Mapping[str, Any],
    video_bytes: bytes,
    openface_executable: Path,
    openface_root: Path,
) -> dict[str, Any]:
    """Convert a short webcam clip to OpenFace rows and predict felt arousal."""
    if not video_bytes:
        raise ValueError("The uploaded camera clip is empty.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError("The camera clip exceeds the 12 MB upload limit.")

    executable = Path(openface_executable)
    root = Path(openface_root)

    if not executable.is_file():
        raise FileNotFoundError(
            f"OpenFace FeatureExtraction executable not found: {executable}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"OpenFace root directory not found: {root}")

    with tempfile.TemporaryDirectory(prefix="neurosense-arousal-") as temporary:
        work_dir = Path(temporary)
        source_path = work_dir / "camera_capture.webm"
        video_path = work_dir / "camera_capture.mp4"
        output_dir = work_dir / "openface"
        output_dir.mkdir()

        source_path.write_bytes(video_bytes)
        duration = _probe_duration(source_path)

        if duration < MIN_VIDEO_SECONDS:
            return {
                "status": "no_prediction",
                "reason": (
                    f"Capture lasted {duration:.2f}s; at least "
                    f"{MIN_VIDEO_SECONDS:.1f}s is required."
                ),
                "valid_frames": 0,
                "required_frames": MIN_VALID_FRAMES,
            }
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"Clip duration {duration:.2f}s exceeds "
                f"the {MAX_VIDEO_SECONDS:.1f}s limit."
            )

        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(source_path),
                "-t", str(WINDOW_SECONDS),
                "-an",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-pix_fmt", "yuv420p",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

        command = [
            str(executable),
            "-f", str(video_path),
            "-aus",
            "-gaze",
            "-pose",
            "-out_dir", str(output_dir),
            "-q",
        ]

        try:
            process = subprocess.run(
                command,
                cwd=str(executable.parent),
                check=False,
                capture_output=True,
                text=True,
                timeout=OPENFACE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "OpenFace arousal feature extraction exceeded its time limit."
            ) from error

        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "")[-1800:]
            raise RuntimeError("OpenFace extraction failed. " + detail)

        csv_candidates = list(output_dir.glob("*.csv"))
        if not csv_candidates:
            raise RuntimeError("OpenFace did not produce a feature CSV.")

        csv_path = max(csv_candidates, key=lambda path: path.stat().st_mtime)
        with csv_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError("OpenFace feature CSV has no header.")

            reader.fieldnames = [
                str(name).strip() for name in reader.fieldnames
            ]
            rows = [
                {
                    str(name).strip(): (
                        value.strip() if isinstance(value, str) else value
                    )
                    for name, value in row.items()
                    if name is not None
                }
                for row in reader
            ]

        return predict_from_openface_rows(bundle, rows)
