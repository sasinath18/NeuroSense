
"""
NeuroSense - WESAD Dataset Preprocessing

Reads:
    data/wesad/S2/S2.pkl
    data/wesad/S3/S3.pkl
    ...

Uses wrist signals only:
    BVP  - 64 Hz
    EDA  - 4 Hz
    TEMP - 4 Hz
    LABELS - 700 Hz

Labels:
    1 = Baseline
    2 = Stress

Output:
    data/stress_dataset.csv

The CSV schema matches train_model.py.

IMPORTANT:
    WESAD EDA is measured in microsiemens (uS).
    The training column is temporarily named gsr_raw
    for compatibility with the existing trainer.

    Before using the model with the physical GSR module,
    calibrate ESP32 ADC readings into the same EDA unit.
"""

from __future__ import annotations

import pickle
import re
import sys

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import butter, sosfiltfilt, find_peaks


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "data" / "wesad"

OUTPUT_PATH = ROOT_DIR / "data" / "stress_dataset.csv"

BVP_RATE = 64
EDA_RATE = 4
TEMP_RATE = 4
LABEL_RATE = 700

WINDOW_SECONDS = 30
STEP_SECONDS = 15

BASELINE_LABEL = 1
STRESS_LABEL = 2

MIN_PEAK_DISTANCE_SECONDS = 60.0 / 200.0

BVP_LOW_HZ = 0.7
BVP_HIGH_HZ = 3.5


# ============================================================
# 2. DISCOVER SUBJECT FILES
# ============================================================

def discover_subject_files():

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"WESAD directory not found: {DATASET_DIR}"
        )

    files = []

    for path in DATASET_DIR.rglob("*.pkl"):

        if re.fullmatch(r"S\d+", path.stem):

            if path.parent.name == path.stem:
                files.append(path)

    files.sort(
        key=lambda item: int(
            item.stem[1:]
        )
    )

    if not files:

        raise FileNotFoundError(
            "\nNo WESAD subject pickle files found.\n"
            "Expected structure:\n"
            "data/wesad/S2/S2.pkl\n"
            "data/wesad/S3/S3.pkl\n"
            "Extract the downloaded dataset first."
        )

    print(
        f"\nFound {len(files)} subject files."
    )

    for file in files:
        print("  ", file.relative_to(ROOT_DIR))

    return files


# ============================================================
# 3. LOAD PICKLE SAFELY FOR WESAD COMPATIBILITY
# ============================================================

def load_subject_file(file_path):

    print(
        f"\nLoading participant {file_path.stem}..."
    )

    with open(file_path, "rb") as file:

        try:
            data = pickle.load(
                file,
                encoding="latin1"
            )

        except TypeError:

            file.seek(0)
            data = pickle.load(file)

    required_keys = [
        "signal",
        "label"
    ]

    for key in required_keys:

        if key not in data:

            raise ValueError(
                f"{file_path.name}: missing '{key}' key."
            )

    if "wrist" not in data["signal"]:

        raise ValueError(
            f"{file_path.name}: wrist signals unavailable."
        )

    wrist = data["signal"]["wrist"]

    required_signals = [
        "BVP",
        "EDA",
        "TEMP"
    ]

    for signal_name in required_signals:

        if signal_name not in wrist:

            raise ValueError(
                f"{file_path.name}: missing wrist {signal_name}."
            )

    bvp = np.asarray(
        wrist["BVP"],
        dtype=float
    ).reshape(-1)

    eda = np.asarray(
        wrist["EDA"],
        dtype=float
    ).reshape(-1)

    temp = np.asarray(
        wrist["TEMP"],
        dtype=float
    ).reshape(-1)

    labels = np.asarray(
        data["label"]
    ).reshape(-1)

    if min(
        len(bvp),
        len(eda),
        len(temp),
        len(labels)
    ) == 0:

        raise ValueError(
            f"{file_path.name}: one or more signals are empty."
        )

    return {
        "bvp": bvp,
        "eda": eda,
        "temp": temp,
        "labels": labels
    }


# ============================================================
# 4. SIGNAL QUALITY AND FILTERING
# ============================================================

def filter_bvp(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    valid = np.isfinite(signal)

    if not np.all(valid):

        signal = pd.Series(
            signal
        ).interpolate(
            limit_direction="both"
        ).to_numpy()

    if len(signal) < 100:

        raise ValueError(
            "BVP recording is too short for filtering."
        )

    if np.std(signal) < 1e-8:

        raise ValueError(
            "BVP signal is nearly constant."
        )

    nyquist = BVP_RATE / 2.0

    low = BVP_LOW_HZ / nyquist
    high = BVP_HIGH_HZ / nyquist

    sos = butter(
        3,
        [low, high],
        btype="bandpass",
        output="sos"
    )

    filtered = sosfiltfilt(
        sos,
        signal
    )

    return filtered


# ============================================================
# 5. PPG FEATURE EXTRACTION
# ============================================================

def extract_bvp_features(
    filtered_bvp,
    start_seconds,
    end_seconds
):

    start_index = int(
        start_seconds * BVP_RATE
    )

    end_index = int(
        end_seconds * BVP_RATE
    )

    segment = filtered_bvp[
        start_index:end_index
    ]

    if len(segment) < BVP_RATE * 10:

        return {
            "heart_rate_bpm": np.nan,
            "hrv_ms": np.nan
        }

    prominence = max(
        float(np.std(segment)) * 0.25,
        1e-8
    )

    peaks, _ = find_peaks(
        segment,
        distance=int(
            BVP_RATE * MIN_PEAK_DISTANCE_SECONDS
        ),
        prominence=prominence
    )

    if len(peaks) < 3:

        return {
            "heart_rate_bpm": np.nan,
            "hrv_ms": np.nan
        }

    intervals_ms = (
        np.diff(peaks) /
        BVP_RATE *
        1000.0
    )

    # Exclude implausible beat intervals.
    intervals_ms = intervals_ms[
        (intervals_ms >= 300)
        & (intervals_ms <= 2000)
    ]

    if len(intervals_ms) == 0:

        return {
            "heart_rate_bpm": np.nan,
            "hrv_ms": np.nan
        }

    median_ibi = float(
        np.median(intervals_ms)
    )

    heart_rate = (
        60000.0 / median_ibi
    )

    if len(intervals_ms) >= 3:

        difference = np.diff(
            intervals_ms
        )

        hrv = float(
            np.sqrt(
                np.mean(
                    difference ** 2
                )
            )
        )

    else:

        hrv = np.nan

    return {
        "heart_rate_bpm": round(
            heart_rate, 3
        ),

        "hrv_ms": (
            round(hrv, 3)
            if np.isfinite(hrv)
            else np.nan
        )
    }


# ============================================================
# 6. EDA AND TEMPERATURE FEATURES
# ============================================================

def extract_low_rate_features(
    signal,
    sampling_rate,
    start_seconds,
    end_seconds
):

    start_index = int(
        start_seconds * sampling_rate
    )

    end_index = int(
        end_seconds * sampling_rate
    )

    segment = np.asarray(
        signal[start_index:end_index],
        dtype=float
    )

    segment = segment[
        np.isfinite(segment)
    ]

    if len(segment) == 0:
        return np.nan

    return float(
        np.mean(segment)
    )


# ============================================================
# 7. CREATE PURE CONDITION WINDOWS
# ============================================================

def extract_subject_windows(
    subject_id,
    signals
):

    bvp = signals["bvp"]
    eda = signals["eda"]
    temp = signals["temp"]
    labels = signals["labels"]

    filtered_bvp = filter_bvp(bvp)

    # Use the shortest common recording duration.
    common_duration = min(
        len(bvp) / BVP_RATE,
        len(eda) / EDA_RATE,
        len(temp) / TEMP_RATE,
        len(labels) / LABEL_RATE
    )

    window_samples = int(
        WINDOW_SECONDS * LABEL_RATE
    )

    step_samples = int(
        STEP_SECONDS * LABEL_RATE
    )

    rows = []

    skipped_windows = 0

    for start_label_index in range(
        0,
        len(labels) - window_samples + 1,
        step_samples
    ):

        end_label_index = (
            start_label_index + window_samples
        )

        start_seconds = (
            start_label_index / LABEL_RATE
        )

        end_seconds = (
            end_label_index / LABEL_RATE
        )

        if end_seconds > common_duration:
            continue

        label_segment = labels[
            start_label_index:end_label_index
        ]

        unique_labels = np.unique(
            label_segment
        )

        # Only complete Baseline or Stress windows.
        if len(unique_labels) != 1:

            skipped_windows += 1
            continue

        label_value = int(
            unique_labels[0]
        )

        if label_value == BASELINE_LABEL:

            label_name = "baseline"

        elif label_value == STRESS_LABEL:

            label_name = "stress"

        else:

            continue

        bvp_features = extract_bvp_features(
            filtered_bvp,
            start_seconds,
            end_seconds
        )

        eda_mean = extract_low_rate_features(
            eda,
            EDA_RATE,
            start_seconds,
            end_seconds
        )

        temp_mean = extract_low_rate_features(
            temp,
            TEMP_RATE,
            start_seconds,
            end_seconds
        )

        row = {

            "participant_id": subject_id,

            "window_id": len(rows),

            "start_time": round(
                start_seconds, 3
            ),

            "end_time": round(
                end_seconds, 3
            ),

            "heart_rate_bpm": (
                bvp_features["heart_rate_bpm"]
            ),

            "hrv_ms": (
                bvp_features["hrv_ms"]
            ),

            # WESAD wrist EDA in microsiemens.
            # See the unit warning in this file.
            "gsr_raw": eda_mean,

            # WESAD wrist temperature in Celsius.
            "skin_temperature_c": temp_mean,

            "stress_label": label_name
        }

        rows.append(row)

    print(
        f"{subject_id}: {len(rows)} windows created; "
        f"{skipped_windows} mixed-label windows skipped."
    )

    return rows


# ============================================================
# 8. PERSONAL BASELINE CALCULATION
# ============================================================

def add_personal_baselines(df):

    baseline_map = {

        "heart_rate_bpm":
            "baseline_heart_rate_bpm",

        "hrv_ms":
            "baseline_hrv_ms",

        "gsr_raw":
            "baseline_gsr_raw",

        "skin_temperature_c":
            "baseline_skin_temperature_c"
    }

    for subject_id, indices in df.groupby(
        "participant_id"
    ).groups.items():

        participant_rows = df.loc[
            indices
        ]

        baseline_rows = participant_rows[
            participant_rows["stress_label"]
            == "baseline"
        ]

        if baseline_rows.empty:

            raise ValueError(
                f"{subject_id}: no baseline windows available."
            )

        for feature, baseline_column in baseline_map.items():

            baseline_value = pd.to_numeric(
                baseline_rows[feature],
                errors="coerce"
            ).median()

            df.loc[
                indices,
                baseline_column
            ] = baseline_value

    return df


# ============================================================
# 9. MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_dataset():

    subject_files = discover_subject_files()

    all_rows = []

    failed_subjects = []

    for file_path in subject_files:

        subject_id = file_path.stem

        try:

            signals = load_subject_file(
                file_path
            )

            rows = extract_subject_windows(
                subject_id,
                signals
            )

            all_rows.extend(rows)

        except Exception as error:

            failed_subjects.append(
                {
                    "subject": subject_id,
                    "error": str(error)
                }
            )

            print(
                f"\nWARNING: {subject_id} skipped: {error}"
            )

    if not all_rows:

        raise RuntimeError(
            "No valid Baseline/Stress windows were generated."
        )

    df = pd.DataFrame(
        all_rows
    )

    if df["participant_id"].nunique() < 2:

        raise RuntimeError(
            "At least two valid participants are required."
        )

    df = add_personal_baselines(
        df
    )

    # Drop records without essential signal features.
    df = df.dropna(
        subset=[
            "heart_rate_bpm",
            "gsr_raw",
            "skin_temperature_c"
        ]
    ).copy()

    if df.empty:

        raise RuntimeError(
            "No complete physiological feature rows remain."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n======================================")
    print("WESAD PREPROCESSING COMPLETED")
    print("======================================")

    print(
        f"Total feature windows: {len(df)}"
    )

    print(
        f"Participants: {df['participant_id'].nunique()}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["stress_label"].value_counts()
    )

    print(
        "\nOutput file:"
    )

    print(OUTPUT_PATH)

    if failed_subjects:

        print(
            "\nParticipants skipped:"
        )

        for item in failed_subjects:
            print(item)

    print(
        "\nIMPORTANT: BVP-derived HRV at 64 Hz is "
        "an exploratory estimate, not a validated clinical HRV measurement."
    )

    print(
        "IMPORTANT: WESAD EDA is in microsiemens. "
        "Calibrate physical GSR readings to compatible units "
        "before model deployment."
    )

    print(
        "\nNext: run train_model.py only after "
        "reviewing the generated CSV."
    )


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        preprocess_dataset()

    except Exception as error:

        print(
            "\nPREPROCESSING FAILED:"
        )

        print(error)

        sys.exit(1)
