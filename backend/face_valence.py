"""AFFEC face-only felt-valence inference for NeuroSense.

The model contract is OpenFace AU intensity, gaze-angle and head-pose
features over the first 4.8 seconds of a video. It is not a stress model.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger("neurosense.face_valence")

WINDOW_SECONDS = 4.8
MIN_VALID_FRAMES = 40
MIN_CONFIDENCE = 0.8
MAX_VIDEO_BYTES = 12 * 1024 * 1024
MIN_VIDEO_SECONDS = 4.70
MAX_VIDEO_SECONDS = 8.0
OPENFACE_TIMEOUT_SECONDS = 90

RAW_FEATURES = (
    "gaze_angle_x",
    "gaze_angle_y",
    "pose_Tx",
    "pose_Ty",
    "pose_Tz",
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
)

# Exact 50-feature sequence from the saved training schema.
FEATURE_NAMES = tuple(
    [f"face__{name}__mean" for name in RAW_FEATURES]
    + [f"face__{name}__std" for name in RAW_FEATURES]
)


class ClipQ(BaseEstimator, TransformerMixin):
    """Training-compatible 1st-99th percentile feature clipping."""

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=np.float64)
        self.lo_ = np.nanpercentile(values, 1, axis=0)
        self.hi_ = np.nanpercentile(values, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lo_, self.hi_)


class KB(SelectKBest):
    """Training-compatible SelectKBest with a capped feature count."""

    def __init__(self, score_func=f_regression, k_max=60):
        super().__init__(score_func=score_func, k=k_max)
        self.k_max = k_max

    def fit(self, X, y):
        self.k = min(self.k_max, X.shape[1])
        return super().fit(X, y)


def load_face_valence_model(
    model_path: Path,
    model_url: str | None = None,
    expected_sha256: str | None = None,
):
    """Load the saved sklearn Pipeline, including its __main__ transformers.

    Colab serialized ClipQ and KB with __module__ == "__main__".
    Register the compatible definitions under __main__ before joblib.load.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file() and model_url:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "gdown is required to download the private model from a "
                "Google Drive sharing URL. Install requirements-render.txt."
            ) from exc

        temporary_path = model_path.with_suffix(model_path.suffix + ".download")
        downloaded = gdown.download(
            url=model_url,
            output=str(temporary_path),
            quiet=False,
            fuzzy=True,
        )
        if not downloaded or not temporary_path.is_file():
            raise RuntimeError(
                "Could not download the face-valence model. Check that the "
                "Google Drive file is shared as 'Anyone with the link – Viewer'."
            )
        temporary_path.replace(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Face-valence model not found at {model_path}; configure "
            "NEUROSENSE_FACE_VALENCE_MODEL_URL or provide the model file."
        )

    if model_path.stat().st_size < 1_000_000:
        raise RuntimeError("Face-valence model file is unexpectedly small.")

    if expected_sha256:
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest.lower() != expected_sha256.strip().lower():
            raise RuntimeError("Face-valence model SHA-256 does not match.")

    main_module = sys.modules["__main__"]
    setattr(main_module, "ClipQ", ClipQ)
    setattr(main_module, "KB", KB)

    model = joblib.load(model_path)
    feature_count = getattr(model, "n_features_in_", None)
    if feature_count is not None and int(feature_count) != len(FEATURE_NAMES):
        raise RuntimeError(
            f"Loaded model expects {feature_count} inputs; "
            f"the live schema provides {len(FEATURE_NAMES)}."
        )
    return model


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_live_feature_frame(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[pd.DataFrame | None, int, str | None]:
    """Compute the model input using float32 frame values and ddof=0 std."""
    valid_values: list[list[float]] = []

    for row in rows:
        timestamp = _as_float(row.get("timestamp"))
        success = _as_float(row.get("success"))
        confidence = _as_float(row.get("confidence"))

        if not np.isfinite(timestamp) or timestamp < 0 or timestamp > WINDOW_SECONDS:
            continue
        if success != 1:
            continue
        if not np.isfinite(confidence) or confidence < MIN_CONFIDENCE:
            continue

        valid_values.append([_as_float(row.get(name)) for name in RAW_FEATURES])

    valid_count = len(valid_values)
    if valid_count < MIN_VALID_FRAMES:
        return (
            None,
            valid_count,
            f"Only {valid_count} valid face frames; at least "
            f"{MIN_VALID_FRAMES} are required.",
        )

    # Cast every frame measurement to float32 BEFORE aggregation, matching
    # the successful parity experiment.
    frame_matrix = np.asarray(valid_values, dtype=np.float32)
    computed: dict[str, np.float32] = {}

    for index, raw_name in enumerate(RAW_FEATURES):
        values = frame_matrix[:, index]
        if not np.isfinite(values).any():
            mean_value = np.float32(np.nan)
            std_value = np.float32(np.nan)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_value = np.nanmean(values, dtype=np.float32)
                std_value = np.nanstd(values, ddof=0, dtype=np.float32)
        computed[f"face__{raw_name}__mean"] = np.float32(mean_value)
        computed[f"face__{raw_name}__std"] = np.float32(std_value)

    # Build in the saved schema order, not alphabetical or dictionary order.
    ordered = [computed[name] for name in FEATURE_NAMES]
    feature_frame = pd.DataFrame(
        [ordered],
        columns=list(FEATURE_NAMES),
        dtype=np.float32,
    )

    if feature_frame.shape != (1, 50):
        raise RuntimeError(
            f"Feature extraction produced {feature_frame.shape}, expected (1, 50)."
        )
    return feature_frame, valid_count, None


def predict_from_openface_rows(
    model,
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    feature_frame, valid_count, reason = build_live_feature_frame(rows)
    if feature_frame is None:
        return {
            "status": "no_prediction",
            "reason": reason,
            "valid_frames": valid_count,
            "required_frames": MIN_VALID_FRAMES,
        }

    prediction = float(np.asarray(model.predict(feature_frame)).reshape(-1)[0])
    if not np.isfinite(prediction):
        raise RuntimeError("The model returned a non-finite felt-valence value.")

    return {
        "status": "prediction_available",
        "felt_valence": prediction,
        "valid_frames": valid_count,
        "features_used": len(FEATURE_NAMES),
        "window_seconds": WINDOW_SECONDS,
        "numeric_type": "float32",
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
    model,
    video_bytes: bytes,
    openface_executable: Path,
    openface_root: Path,
) -> dict[str, Any]:
    """Transcode an explicitly submitted short clip, run OpenFace, predict.

    All uploaded media and OpenFace CSV output live in a TemporaryDirectory
    and are deleted automatically after the request completes.
    """
    if not video_bytes:
        raise ValueError("The uploaded video is empty.")
    if len(video_bytes) > MAX_VIDEO_BYTES:
        raise ValueError("The video exceeds the 12 MB upload limit.")

    executable = Path(openface_executable)
    root = Path(openface_root)
    if not executable.is_file():
        raise FileNotFoundError(
            f"OpenFace FeatureExtraction executable not found: {executable}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"OpenFace root directory not found: {root}")

    with tempfile.TemporaryDirectory(prefix="neurosense-valence-") as temporary:
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
                    f"Capture lasted {duration:.2f}s; a complete "
                    f"{WINDOW_SECONDS:.1f}s video window is required."
                ),
                "valid_frames": 0,
                "required_frames": MIN_VALID_FRAMES,
            }
        if duration > MAX_VIDEO_SECONDS:
            raise ValueError(
                f"Video duration {duration:.2f}s exceeds the "
                f"{MAX_VIDEO_SECONDS:.1f}s upload limit."
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
        process = subprocess.run(
            command,
            cwd=str(root),
            check=False,
            capture_output=True,
            text=True,
            timeout=OPENFACE_TIMEOUT_SECONDS,
        )
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "")[-1800:]
            raise RuntimeError(
                "OpenFace feature extraction failed. " + detail
            )

        csv_candidates = list(output_dir.glob("*.csv"))
        if not csv_candidates:
            raise RuntimeError("OpenFace did not produce a feature CSV.")

        csv_path = max(csv_candidates, key=lambda path: path.stat().st_mtime)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise RuntimeError("OpenFace feature CSV has no header.")
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
            rows = [
                {
                    str(name).strip(): (value.strip() if isinstance(value, str) else value)
                    for name, value in row.items()
                    if name is not None
                }
                for row in reader
            ]

        return predict_from_openface_rows(model, rows)
