"""Experimental AFFEC face-only felt-valence feature extraction for NeuroSense.

The currently implemented live extractor truncates a submitted clip to its
first 4.8 seconds. The supplied model metadata says training used a
stimulus-locked, variable-duration video phase ending at last_frame_video
(median 4.61 seconds; observed 3.53 to about 5.3 seconds). These windows are
not equivalent. Keep the API inference route disabled until numerical parity
and new-user validation are independently completed. This is not a stress model.
"""

from __future__ import annotations

import csv
import json
import hashlib
import logging
import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger("neurosense.face_valence")

# Live prototype capture length only; not a validated reproduction of the event-anchored training window.
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


def load_face_valence_model(
    model_path: Path,
    model_url: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Ensure the AFFEC artifact exists and passes integrity checks.

    The model was serialized with scikit-learn 1.6.1, while the wearable
    stress model uses 1.9.1. Do not unpickle AFFEC in the main API process.
    Inference is delegated to the isolated /opt/affec-venv runtime.
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
        try:
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
            if temporary_path.stat().st_size < 1_000_000:
                raise RuntimeError("Downloaded face-valence model is unexpectedly small.")
            if expected_sha256:
                digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
                if digest.lower() != expected_sha256.strip().lower():
                    raise RuntimeError("Downloaded face-valence model SHA-256 does not match.")
            temporary_path.replace(model_path)
        finally:
            temporary_path.unlink(missing_ok=True)

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

    return model_path


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

        if not np.isfinite(timestamp) or timestamp < 0 or timestamp > (WINDOW_SECONDS + 1e-9):
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


def _predict_with_affec_worker(
    model_path: Path,
    feature_frame: pd.DataFrame,
) -> float:
    """Run the serialized AFFEC pipeline in its pinned isolated environment."""
    python_path = Path(os.getenv(
        "NEUROSENSE_AFFEC_PYTHON",
        "/opt/affec-venv/bin/python",
    ))
    worker_path = Path(__file__).with_name("face_valence_worker.py")
    if not python_path.is_file():
        raise FileNotFoundError(
            f"Isolated AFFEC Python runtime not found: {python_path}. "
            "Build the Docker image with requirements-affec-worker.txt."
        )
    if not worker_path.is_file():
        raise FileNotFoundError(f"AFFEC worker script not found: {worker_path}")

    row = feature_frame.iloc[0].to_dict()
    payload = {
        "feature_names": list(FEATURE_NAMES),
        "features": {
            name: (float(value) if np.isfinite(value) else None)
            for name, value in row.items()
        },
    }
    try:
        process = subprocess.run(
            [str(python_path), str(worker_path), str(Path(model_path))],
            input=json.dumps(payload, allow_nan=False),
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError("AFFEC isolated model inference exceeded its time limit.") from error

    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "No worker details returned.")[-1800:]
        raise RuntimeError("AFFEC model worker failed. " + detail)

    try:
        response = json.loads(process.stdout)
        prediction = float(response["felt_valence"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("AFFEC model worker returned an invalid response.") from error
    if not np.isfinite(prediction):
        raise RuntimeError("The AFFEC worker returned a non-finite felt-valence value.")
    return prediction


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

    if isinstance(model, (str, Path)):
        prediction = _predict_with_affec_worker(Path(model), feature_frame)
    else:
        # Retained for unit tests and explicitly injected in-process estimators.
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
        try:
            process = subprocess.run(
                command,
                # OpenFace resolves its bundled model files relative to this directory.
                cwd=str(executable.parent),
                check=False,
                capture_output=True,
                text=True,
                timeout=OPENFACE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(
                "OpenFace feature extraction exceeded its processing time limit."
            ) from error
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
