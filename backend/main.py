
"""NeuroSense single-file FastAPI backend.

Local MVP backend for a wearable wellness-monitoring prototype.
- SQLite persistence
- Firebase ID-token verification (optional until configured)
- Secure ESP32 device pairing with one-time device tokens
- Sensor ingestion, PPG/IBI feature extraction and quality gating
- Personal baseline calibration
- Optional trained-model inference (no fabricated score)
- Three-hour summaries, high-stress event records
- Haptic command queue and recovery-session history

This is a development/MVP backend, not a medical device backend.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import sqlite3
import uuid

from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np

from dotenv import load_dotenv
from fastapi import (
    Depends, FastAPI, Header, HTTPException, Query, status
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import (
    BaseModel, ConfigDict, Field, model_validator
)

from scipy.signal import butter, find_peaks, sosfiltfilt

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials, firestore


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

APP_ENV = os.getenv(
    "NEUROSENSE_ENV", "development"
).strip().lower()

DB_PATH = Path(
    os.getenv(
        "NEUROSENSE_DB_PATH",
        str(BASE_DIR / "data" / "local" / "neurosense.db")
    )
)

FIREBASE_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH", ""
).strip()

# Keep this JSON value in the hosting provider's secret environment.
# It is never returned by any public API.
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_JSON", ""
).strip()

FIREBASE_WEB_CONFIG = {
    "apiKey": os.getenv("FIREBASE_WEB_API_KEY", "").strip(),
    "authDomain": os.getenv("FIREBASE_WEB_AUTH_DOMAIN", "").strip(),
    "projectId": os.getenv("FIREBASE_WEB_PROJECT_ID", "").strip(),
    "appId": os.getenv("FIREBASE_WEB_APP_ID", "").strip(),
    "messagingSenderId": os.getenv(
        "FIREBASE_WEB_MESSAGING_SENDER_ID", ""
    ).strip(),
}

ML_MODEL_PATH = Path(
    os.getenv(
        "NEUROSENSE_MODEL_PATH",
        str(BASE_DIR / "ml_model" / "neurosense_stress_model.joblib")
    )
)

HIGH_STRESS_THRESHOLD = float(
    os.getenv("NEUROSENSE_HIGH_STRESS_THRESHOLD", "0.75")
)

SUMMARY_HOURS = 3
MAX_READING_AGE_HOURS = 24
MAX_FUTURE_CLOCK_SKEW_MINUTES = 5

ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurosense")

FIREBASE_APP = None
FIRESTORE_CLIENT = None
STRESS_MODEL = None


# ============================================================
# 2. COMMON HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc(value: datetime | None) -> datetime:

    if value is None:
        return utc_now()

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


def iso(value: datetime | None = None) -> str:
    return to_utc(value).isoformat()


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return to_utc(parsed)


def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def median_or_none(values: list[float]) -> float | None:

    return (
        round(float(np.median(values)), 3)
        if values else None
    )


def row_dict(row: sqlite3.Row | None) -> dict | None:

    return dict(row) if row is not None else None


# ============================================================
# 3. SQLITE DATABASE
# ============================================================

@contextmanager
def db_connection():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        str(DB_PATH),
        timeout=20
    )

    connection.row_factory = sqlite3.Row

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialize_database() -> None:

    with db_connection() as db:

        db.executescript("""

        CREATE TABLE IF NOT EXISTS devices (
            device_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            device_name TEXT NOT NULL,
            token_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen TEXT,
            active INTEGER NOT NULL DEFAULT 1
        );

        CREATE INDEX IF NOT EXISTS idx_devices_user
        ON devices(user_id, active);


        CREATE TABLE IF NOT EXISTS sensor_readings (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            received_at TEXT NOT NULL,
            heart_rate_bpm REAL,
            hrv_ms REAL,
            gsr_raw INTEGER,
            skin_temperature_c REAL,
            signal_quality TEXT NOT NULL,
            source TEXT NOT NULL,
            ibi_ms_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_readings_user_time
        ON sensor_readings(user_id, timestamp);

        CREATE INDEX IF NOT EXISTS idx_readings_device_time
        ON sensor_readings(device_id, timestamp);


        CREATE TABLE IF NOT EXISTS baselines (
            user_id TEXT PRIMARY KEY,
            calibrated_at TEXT NOT NULL,
            sample_count INTEGER NOT NULL,
            heart_rate_bpm REAL,
            hrv_ms REAL,
            gsr_raw REAL,
            skin_temperature_c REAL
        );


        CREATE TABLE IF NOT EXISTS user_settings (
            user_id TEXT PRIMARY KEY,
            haptic_enabled INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );


        CREATE TABLE IF NOT EXISTS haptic_commands (
            command_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            command_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            delivered_at TEXT,
            acknowledged_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_haptic_device_status
        ON haptic_commands(
            device_id, command_status, created_at
        );


        CREATE TABLE IF NOT EXISTS recovery_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            activity TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            duration_seconds INTEGER NOT NULL,
            completed INTEGER NOT NULL,
            before_reading_json TEXT,
            after_reading_json TEXT
        );


        CREATE TABLE IF NOT EXISTS summaries (
            summary_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            reading_count INTEGER NOT NULL,
            reliable_reading_count INTEGER NOT NULL,
            avg_heart_rate REAL,
            avg_hrv REAL,
            avg_gsr REAL,
            avg_skin_temperature REAL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, period_end)
        );


        CREATE TABLE IF NOT EXISTS high_stress_events (
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            reading_id TEXT NOT NULL UNIQUE,
            stress_probability REAL NOT NULL,
            occurred_at TEXT NOT NULL,
            FOREIGN KEY(reading_id)
                REFERENCES sensor_readings(id)
        );

        """)


# ============================================================
# 4. OPTIONAL FIREBASE INITIALIZATION
# ============================================================

def initialize_firebase() -> None:

    global FIREBASE_APP, FIRESTORE_CLIENT

    certificate_source = None
    service_account_json = FIREBASE_SERVICE_ACCOUNT_JSON

    if service_account_json:
        try:
            certificate_source = json.loads(service_account_json)
            if not isinstance(certificate_source, dict):
                raise ValueError("Service-account JSON must be an object.")
        except Exception:
            logger.exception(
                "FIREBASE_SERVICE_ACCOUNT_JSON is invalid JSON."
            )
            return

    elif FIREBASE_SERVICE_ACCOUNT_PATH:
        credential_path = Path(FIREBASE_SERVICE_ACCOUNT_PATH)

        if not credential_path.is_absolute():
            credential_path = BASE_DIR / credential_path

        if not credential_path.is_file():
            logger.error(
                "Firebase service-account file does not exist: %s",
                credential_path
            )
            return

        certificate_source = str(credential_path)

    else:
        logger.info(
            "Firebase Admin is not configured; production user APIs remain protected."
        )
        return

    try:

        try:
            FIREBASE_APP = firebase_admin.get_app()

        except ValueError:

            FIREBASE_APP = firebase_admin.initialize_app(
                credentials.Certificate(certificate_source)
            )

        FIRESTORE_CLIENT = firestore.client(
            app=FIREBASE_APP
        )

        logger.info(
            "Firebase Authentication and Firestore initialized."
        )

    except Exception:

        logger.exception(
            "Firebase initialization failed."
        )


def firebase_web_configured() -> bool:
    required_fields = (
        "apiKey",
        "authDomain",
        "projectId",
        "appId",
        "messagingSenderId"
    )
    return all(FIREBASE_WEB_CONFIG.get(field) for field in required_fields)


# ============================================================
# 5. OPTIONAL MACHINE LEARNING MODEL
# ============================================================

def initialize_ml_model() -> None:

    global STRESS_MODEL

    if not ML_MODEL_PATH.is_file():

        logger.info(
            "No trained stress model found. "
            "Stress predictions will remain unavailable."
        )

        STRESS_MODEL = None
        return

    try:

        STRESS_MODEL = joblib.load(ML_MODEL_PATH)

        logger.info(
            "Stress model loaded from %s",
            ML_MODEL_PATH
        )

    except Exception:

        logger.exception(
            "Could not load the stress model."
        )

        STRESS_MODEL = None


# ============================================================
# 6. FIRESTORE STORAGE HELPER
# ============================================================

def firestore_write(
    collection_path: tuple[str, ...],
    document_id: str,
    payload: dict
) -> str:

    """Write to users/{uid}/{subcollection}/{document_id}."""

    if FIRESTORE_CLIENT is None:
        return "not_configured"

    if not collection_path or len(collection_path) % 2 == 0:

        logger.error(
            "Invalid Firestore collection path: %s",
            collection_path
        )

        return "failed"

    try:

        reference = FIRESTORE_CLIENT.collection(
            collection_path[0]
        )

        for index, segment in enumerate(
            collection_path[1:], start=1
        ):

            if index % 2 == 1:
                reference = reference.document(segment)
            else:
                reference = reference.collection(segment)

        reference.document(document_id).set(payload)

        return "synced"

    except Exception:

        logger.exception(
            "Firestore sync failed for %s",
            document_id
        )

        return "failed"


# ============================================================
# 7. REQUEST VALIDATION MODELS
# ============================================================

class StrictModel(BaseModel):

    model_config = ConfigDict(
        extra="forbid"
    )


class SensorReadingIn(StrictModel):

    timestamp: datetime | None = None

    heart_rate_bpm: float | None = Field(
        default=None, ge=30, le=220
    )

    hrv_ms: float | None = Field(
        default=None, ge=0, le=300
    )

    gsr_raw: int | None = Field(
        default=None, ge=0, le=4095
    )

    skin_temperature_c: float | None = Field(
        default=None, ge=20, le=45
    )

    signal_quality: Literal[
        "good", "moderate", "poor"
    ] = "poor"

    source: Literal[
        "wearable", "simulated"
    ] = "wearable"

    ibi_ms: list[float] | None = Field(
        default=None, max_length=120
    )

    ppg_samples: list[int] | None = Field(
        default=None, max_length=5000
    )

    sampling_rate_hz: float | None = Field(
        default=None, ge=25, le=200
    )

    @model_validator(mode="after")
    def check_measurements(self):

        if all(value is None for value in (
            self.heart_rate_bpm,
            self.hrv_ms,
            self.gsr_raw,
            self.skin_temperature_c,
            self.ibi_ms,
            self.ppg_samples
        )):

            raise ValueError(
                "At least one sensor measurement is required."
            )

        if (
            self.ppg_samples is not None
            and self.sampling_rate_hz is None
        ):

            raise ValueError(
                "sampling_rate_hz is required when ppg_samples are supplied."
            )

        if self.ibi_ms is not None:

            if len(self.ibi_ms) < 3:

                raise ValueError(
                    "At least 3 IBI intervals are required for HRV extraction."
                )

            if any(
                value < 300 or value > 2000
                for value in self.ibi_ms
            ):

                raise ValueError(
                    "IBI intervals must be between 300 and 2000 ms."
                )

        return self


class DevicePairIn(StrictModel):

    device_id: str = Field(
        min_length=3,
        max_length=80,
        pattern=r"^[A-Za-z0-9_.-]+$"
    )

    device_name: str = Field(
        default="NeuroSense ESP32",
        min_length=1,
        max_length=80
    )


class HapticSettingIn(StrictModel):

    haptic_enabled: bool


class HapticCommandIn(StrictModel):

    device_id: str = Field(
        min_length=3,
        max_length=80
    )

    duration_ms: int = Field(
        default=500,
        ge=100,
        le=3000
    )


class RecoverySessionIn(StrictModel):

    activity: Literal[
        "breathing", "music", "haptic"
    ]

    started_at: datetime

    completed_at: datetime | None = None

    duration_seconds: int = Field(
        ge=1,
        le=7200
    )

    completed: bool = True


# ============================================================
# 8. USER AUTHENTICATION
# ============================================================

async def current_user_id(
    authorization: Annotated[
        str | None, Header()
    ] = None,

    x_demo_user: Annotated[
        str | None, Header()
    ] = None,
) -> str:

    if FIREBASE_APP is not None:

        if not authorization:

            raise HTTPException(
                status_code=401,
                detail="Firebase Bearer token required."
            )

        parts = authorization.split()

        if (
            len(parts) != 2
            or parts[0].lower() != "bearer"
        ):

            raise HTTPException(
                status_code=401,
                detail="Use Authorization: Bearer <Firebase ID token>."
            )

        try:

            claims = firebase_auth.verify_id_token(
                parts[1],
                check_revoked=True,
                app=FIREBASE_APP
            )

            return str(claims["uid"])

        except Exception:

            raise HTTPException(
                status_code=401,
                detail="Invalid or expired Firebase ID token."
            )

    if APP_ENV != "development":

        raise HTTPException(
            status_code=503,
            detail="Firebase authentication is not configured."
        )

    demo_id = x_demo_user or "local-demo-user"

    if (
        len(demo_id) > 80
        or not all(
            c.isalnum() or c in "_.-"
            for c in demo_id
        )
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid development user ID."
        )

    return demo_id


# ============================================================
# 9. ESP32 DEVICE AUTHENTICATION
# ============================================================

def authenticated_device(
    device_id: str,
    device_token: str
) -> sqlite3.Row:

    if not device_token or len(device_token) < 24:

        raise HTTPException(
            status_code=401,
            detail="Valid X-Device-Token header required."
        )

    with db_connection() as db:

        row = db.execute(
            """
            SELECT *
            FROM devices
            WHERE device_id = ?
            AND active = 1
            """,
            (device_id,)
        ).fetchone()

    if (
        row is None
        or not secrets.compare_digest(
            row["token_hash"],
            token_digest(device_token)
        )
    ):

        raise HTTPException(
            status_code=401,
            detail="Device authentication failed."
        )

    return row


# ============================================================
# 10. HRV AND PPG SIGNAL PROCESSING
# ============================================================

def hrv_from_intervals(
    intervals: list[float]
) -> dict:

    values = np.asarray(
        intervals,
        dtype=float
    )

    values = values[np.isfinite(values)]

    values = values[
        (values >= 300) & (values <= 2000)
    ]

    if len(values) < 3:

        return {
            "hrv_ms": None,
            "sdnn_ms": None,
            "ibi_count": int(len(values))
        }

    differences = np.diff(values)

    rmssd = float(
        np.sqrt(np.mean(np.square(differences)))
    )

    sdnn = float(
        np.std(values, ddof=1)
    )

    return {
        "hrv_ms": round(rmssd, 2),
        "sdnn_ms": round(sdnn, 2),
        "ibi_count": int(len(values)),
        "heart_rate_from_ibi": round(
            60000 / float(np.median(values)),
            2
        )
    }


def extract_ibi_from_ppg(
    samples: list[int],
    sampling_rate_hz: float
) -> list[float]:

    signal = np.asarray(
        samples,
        dtype=float
    )

    if len(signal) < max(
        80,
        int(sampling_rate_hz * 3)
    ):
        return []

    signal = signal - np.mean(signal)

    nyquist = sampling_rate_hz / 2.0

    low = 0.5 / nyquist
    high = min(4.0 / nyquist, 0.95)

    if low >= high:
        return []

    try:

        sos = butter(
            2,
            [low, high],
            btype="bandpass",
            output="sos"
        )

        filtered = sosfiltfilt(
            sos,
            signal
        )

        min_distance = max(
            1,
            int(sampling_rate_hz * 60 / 220)
        )

        prominence = max(
            float(np.std(filtered)) * 0.35,
            1e-9
        )

        peaks, _ = find_peaks(
            filtered,
            distance=min_distance,
            prominence=prominence
        )

        if len(peaks) < 4:
            return []

        intervals = (
            np.diff(peaks) /
            sampling_rate_hz *
            1000.0
        )

        return [
            float(value)
            for value in intervals
            if 300 <= value <= 2000
        ]

    except Exception:

        logger.exception(
            "PPG feature extraction failed."
        )

        return []


def prepare_sensor_payload(
    reading: SensorReadingIn
) -> dict:

    observed_at = to_utc(reading.timestamp)
    now = utc_now()

    if observed_at > now + timedelta(
        minutes=MAX_FUTURE_CLOCK_SKEW_MINUTES
    ):

        raise HTTPException(
            status_code=422,
            detail="Sensor timestamp is too far in the future."
        )

    if observed_at < now - timedelta(
        hours=MAX_READING_AGE_HOURS
    ):

        raise HTTPException(
            status_code=422,
            detail="Sensor reading is older than the allowed 24-hour window."
        )

    intervals = reading.ibi_ms or []

    if (
        not intervals
        and reading.ppg_samples is not None
    ):

        intervals = extract_ibi_from_ppg(
            reading.ppg_samples,
            reading.sampling_rate_hz or 100.0
        )

    hrv_features = (
        hrv_from_intervals(intervals)
        if intervals else {}
    )

    heart_rate = reading.heart_rate_bpm
    hrv_ms = reading.hrv_ms

    if hrv_features:

        hrv_ms = hrv_features["hrv_ms"]

        if heart_rate is None:

            heart_rate = hrv_features.get(
                "heart_rate_from_ibi"
            )

    quality = reading.signal_quality

    if (
        reading.ppg_samples is not None
        and not intervals
    ):

        quality = "poor"

    if (
        reading.ibi_ms is not None
        and hrv_features.get("hrv_ms") is None
    ):

        quality = "poor"

    return {
        "timestamp": iso(observed_at),
        "received_at": iso(now),

        "heart_rate_bpm": (
            round(heart_rate, 2)
            if heart_rate is not None
            else None
        ),

        "hrv_ms": (
            round(hrv_ms, 2)
            if hrv_ms is not None
            else None
        ),

        "gsr_raw": reading.gsr_raw,

        "skin_temperature_c": (
            round(reading.skin_temperature_c, 2)
            if reading.skin_temperature_c is not None
            else None
        ),

        "signal_quality": quality,
        "source": reading.source,
        "ibi_features": hrv_features or None
    }


# ============================================================
# 11. PERSONAL BASELINE
# ============================================================

def baseline_for_user(
    user_id: str
) -> dict | None:

    with db_connection() as db:

        row = db.execute(
            """
            SELECT *
            FROM baselines
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

    return row_dict(row)


def calibrate_baseline(
    user_id: str,
    lookback_minutes: int = 10
) -> dict:

    since = iso(
        utc_now() - timedelta(
            minutes=lookback_minutes
        )
    )

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE user_id = ?
            AND source = 'wearable'
            AND signal_quality IN ('good', 'moderate')
            AND timestamp >= ?
            ORDER BY timestamp DESC
            """,
            (user_id, since)
        ).fetchall()

    if len(rows) < 10:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Need at least 10 good/moderate wearable readings "
                f"from the last {lookback_minutes} minutes; "
                f"found {len(rows)}."
            )
        )

    def values(key):

        return [
            float(row[key])
            for row in rows
            if row[key] is not None
        ]

    baseline = {
        "user_id": user_id,
        "calibrated_at": iso(),
        "sample_count": len(rows),

        "heart_rate_bpm": median_or_none(
            values("heart_rate_bpm")
        ),

        "hrv_ms": median_or_none(
            values("hrv_ms")
        ),

        "gsr_raw": median_or_none(
            values("gsr_raw")
        ),

        "skin_temperature_c": median_or_none(
            values("skin_temperature_c")
        )
    }

    if (
        baseline["heart_rate_bpm"] is None
        and baseline["hrv_ms"] is None
        and baseline["gsr_raw"] is None
    ):

        raise HTTPException(
            status_code=422,
            detail="Not enough usable physiological features to calibrate baseline."
        )

    with db_connection() as db:

        db.execute(
            """
            INSERT INTO baselines (
                user_id,
                calibrated_at,
                sample_count,
                heart_rate_bpm,
                hrv_ms,
                gsr_raw,
                skin_temperature_c
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(user_id)
            DO UPDATE SET
                calibrated_at=excluded.calibrated_at,
                sample_count=excluded.sample_count,
                heart_rate_bpm=excluded.heart_rate_bpm,
                hrv_ms=excluded.hrv_ms,
                gsr_raw=excluded.gsr_raw,
                skin_temperature_c=excluded.skin_temperature_c
            """,
            (
                user_id,
                baseline["calibrated_at"],
                baseline["sample_count"],
                baseline["heart_rate_bpm"],
                baseline["hrv_ms"],
                baseline["gsr_raw"],
                baseline["skin_temperature_c"]
            )
        )

    firestore_write(
        ("users", user_id, "baselines"),
        "current",
        baseline
    )

    return baseline


# ============================================================
# 12. MACHINE LEARNING FEATURE PREPARATION
# ============================================================

def build_model_features(
    reading: dict,
    baseline: dict
) -> dict:

    def delta(field: str) -> float:

        current = reading.get(field)
        base = baseline.get(field)

        if current is None or base is None:
            return 0.0

        return float(current) - float(base)

    return {
        "heart_rate_bpm": float(
            reading.get("heart_rate_bpm") or 0
        ),

        "hrv_ms": float(
            reading.get("hrv_ms") or 0
        ),

        "gsr_raw": float(
            reading.get("gsr_raw") or 0
        ),

        "skin_temperature_c": float(
            reading.get("skin_temperature_c") or 0
        ),

        "heart_rate_delta": delta(
            "heart_rate_bpm"
        ),

        "hrv_delta": delta(
            "hrv_ms"
        ),

        "gsr_delta": delta(
            "gsr_raw"
        ),

        "skin_temperature_delta": delta(
            "skin_temperature_c"
        )
    }


def predict_stress(
    user_id: str,
    reading: dict
) -> dict:

    if STRESS_MODEL is None:

        return {
            "status": "model_not_available",
            "stress_score": None,
            "message": "A trained and validated model is not installed."
        }

    if reading["signal_quality"] == "poor":

        return {
            "status": "poor_signal_quality",
            "stress_score": None,
            "message": "Reading excluded because signal quality is poor."
        }

    baseline = baseline_for_user(user_id)

    if baseline is None:

        return {
            "status": "baseline_required",
            "stress_score": None,
            "message": "Calibrate a personal resting baseline before prediction."
        }

    features = build_model_features(
        reading,
        baseline
    )

    feature_order = [
        "heart_rate_bpm",
        "hrv_ms",
        "gsr_raw",
        "skin_temperature_c",
        "heart_rate_delta",
        "hrv_delta",
        "gsr_delta",
        "skin_temperature_delta"
    ]

    matrix = np.asarray(
        [[features[name] for name in feature_order]],
        dtype=float
    )

    try:

        if hasattr(STRESS_MODEL, "predict_proba"):

            probabilities = STRESS_MODEL.predict_proba(
                matrix
            )[0]

            classes = list(
                getattr(
                    STRESS_MODEL,
                    "classes_",
                    range(len(probabilities))
                )
            )

            stress_index = next(
                (
                    index
                    for index, label in enumerate(classes)
                    if str(label).strip().lower()
                    in {"1", "stress", "stressed", "high"}
                ),
                len(probabilities) - 1
            )

            probability = float(
                probabilities[stress_index]
            )

            score = round(
                float(np.clip(probability, 0, 1) * 100),
                1
            )

            result = {
                "status": "prediction_available",
                "stress_score": score,
                "stress_probability": round(
                    probability, 4
                ),
                "model_validated": False
            }

            if probability >= HIGH_STRESS_THRESHOLD:

                event_id = str(uuid.uuid4())

                with db_connection() as db:

                    db.execute(
                        """
                        INSERT OR IGNORE INTO high_stress_events (
                            event_id,
                            user_id,
                            reading_id,
                            stress_probability,
                            occurred_at
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            event_id,
                            user_id,
                            reading["id"],
                            probability,
                            reading["timestamp"]
                        )
                    )

            return result

        label = STRESS_MODEL.predict(matrix)[0]

        return {
            "status": "label_only",
            "stress_score": None,
            "predicted_label": str(label),
            "message": "Model does not expose calibrated class probabilities."
        }

    except Exception:

        logger.exception("Stress prediction failed.")

        return {
            "status": "prediction_error",
            "stress_score": None,
            "message": "Model inference failed; see backend log."
        }


# ============================================================
# 13. FIRESTORE STORAGE AND SENSOR READING
# ============================================================

def store_sensor_reading(
    user_id: str,
    device_id: str,
    reading: SensorReadingIn
) -> dict:

    payload = prepare_sensor_payload(reading)

    reading_id = str(uuid.uuid4())

    ibi_json = (
        json.dumps(reading.ibi_ms)
        if reading.ibi_ms
        else None
    )

    with db_connection() as db:

        db.execute(
            """
            INSERT INTO sensor_readings (
                id,
                user_id,
                device_id,
                timestamp,
                received_at,
                heart_rate_bpm,
                hrv_ms,
                gsr_raw,
                skin_temperature_c,
                signal_quality,
                source,
                ibi_ms_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                reading_id,
                user_id,
                device_id,
                payload["timestamp"],
                payload["received_at"],
                payload["heart_rate_bpm"],
                payload["hrv_ms"],
                payload["gsr_raw"],
                payload["skin_temperature_c"],
                payload["signal_quality"],
                payload["source"],
                ibi_json
            )
        )

        db.execute(
            """
            UPDATE devices
            SET last_seen = ?
            WHERE device_id = ?
            AND user_id = ?
            """,
            (
                payload["received_at"],
                device_id,
                user_id
            )
        )

    stored = {
        "id": reading_id,
        "user_id": user_id,
        "device_id": device_id,
        **payload
    }

    cloud_status = firestore_write(
        ("users", user_id, "sensor_readings"),
        reading_id,
        stored
    )

    if payload["source"] == "wearable":

        prediction = predict_stress(
            user_id,
            stored
        )

    else:

        prediction = {
            "status": "simulated_data",
            "stress_score": None
        }

    return {
        "reading": stored,
        "cloud_sync": cloud_status,
        "stress": prediction
    }


# ============================================================
# 14. THREE-HOUR SUMMARY GENERATION
# ============================================================

def average(
    rows: list[sqlite3.Row],
    key: str
) -> float | None:

    vals = [
        float(row[key])
        for row in rows
        if row[key] is not None
    ]

    return (
        round(float(np.mean(vals)), 2)
        if vals else None
    )


def create_due_summaries() -> int:

    created = 0
    now = utc_now()

    with db_connection() as db:

        users = db.execute(
            """
            SELECT DISTINCT user_id
            FROM sensor_readings
            WHERE source = 'wearable'
            """
        ).fetchall()

        for user_row in users:

            user_id = user_row["user_id"]

            last = db.execute(
                """
                SELECT MAX(period_end) AS end_time
                FROM summaries
                WHERE user_id = ?
                """,
                (user_id,)
            ).fetchone()

            if last["end_time"]:

                period_start = parse_iso(
                    last["end_time"]
                )

            else:

                first = db.execute(
                    """
                    SELECT MIN(timestamp) AS start_time
                    FROM sensor_readings
                    WHERE user_id = ?
                    AND source = 'wearable'
                    """,
                    (user_id,)
                ).fetchone()

                if not first["start_time"]:
                    continue

                period_start = parse_iso(
                    first["start_time"]
                )

            windows = 0

            while (
                now - period_start
                >= timedelta(hours=SUMMARY_HOURS)
                and windows < 24
            ):

                period_end = period_start + timedelta(
                    hours=SUMMARY_HOURS
                )

                rows = db.execute(
                    """
                    SELECT *
                    FROM sensor_readings
                    WHERE user_id = ?
                    AND source = 'wearable'
                    AND timestamp >= ?
                    AND timestamp < ?
                    """,
                    (
                        user_id,
                        iso(period_start),
                        iso(period_end)
                    )
                ).fetchall()

                if rows:

                    reliable = [
                        row for row in rows
                        if row["signal_quality"]
                        in ("good", "moderate")
                    ]

                    summary = {
                        "summary_id": str(uuid.uuid4()),
                        "user_id": user_id,
                        "period_start": iso(period_start),
                        "period_end": iso(period_end),
                        "reading_count": len(rows),
                        "reliable_reading_count": len(reliable),

                        "avg_heart_rate": average(
                            reliable, "heart_rate_bpm"
                        ),

                        "avg_hrv": average(
                            reliable, "hrv_ms"
                        ),

                        "avg_gsr": average(
                            reliable, "gsr_raw"
                        ),

                        "avg_skin_temperature": average(
                            reliable, "skin_temperature_c"
                        ),

                        "created_at": iso()
                    }

                    db.execute(
                        """
                        INSERT OR IGNORE INTO summaries (
                            summary_id,
                            user_id,
                            period_start,
                            period_end,
                            reading_count,
                            reliable_reading_count,
                            avg_heart_rate,
                            avg_hrv,
                            avg_gsr,
                            avg_skin_temperature,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            summary["summary_id"],
                            user_id,
                            summary["period_start"],
                            summary["period_end"],
                            summary["reading_count"],
                            summary["reliable_reading_count"],
                            summary["avg_heart_rate"],
                            summary["avg_hrv"],
                            summary["avg_gsr"],
                            summary["avg_skin_temperature"],
                            summary["created_at"]
                        )
                    )

                    firestore_write(
                        ("users", user_id, "summaries"),
                        summary["summary_id"],
                        summary
                    )

                    created += 1

                period_start = period_end
                windows += 1

    return created


async def summary_scheduler():

    while True:

        try:

            count = create_due_summaries()

            if count:
                logger.info(
                    "Created %s three-hour summary window(s).",
                    count
                )

        except Exception:

            logger.exception(
                "Three-hour summary scheduler failed."
            )

        await asyncio.sleep(60)


# ============================================================
# 15. FASTAPI STARTUP AND CORS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_database()
    initialize_firebase()
    initialize_ml_model()

    task = asyncio.create_task(
        summary_scheduler()
    )

    logger.info(
        "NeuroSense backend started in %s mode.",
        APP_ENV
    )

    yield

    task.cancel()

    try:
        await task

    except asyncio.CancelledError:
        pass

    logger.info("NeuroSense backend stopped.")


app = FastAPI(
    title="NeuroSense API",
    description=(
        "Personalized wearable wellness monitoring "
        "and recovery feedback API."
    ),
    version="3.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=[
        "GET", "POST", "PUT", "DELETE", "OPTIONS"
    ],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-Demo-User",
        "X-Device-Token"
    ]
)


# ============================================================
# 16. HOME AND HEALTH ENDPOINTS
# ============================================================

@app.get("/dashboard", include_in_schema=False)
def dashboard():
    """Serve the single-file NeuroSense web dashboard."""
    dashboard_file = BASE_DIR / "frontend" / "index.html"
    if not dashboard_file.is_file():
        raise HTTPException(
            status_code=503,
            detail="NeuroSense dashboard file is unavailable."
        )
    return FileResponse(
        dashboard_file,
        media_type="text/html",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/")
def home():

    return {
        "project": "NeuroSense",
        "version": "3.0.0",
        "status": "online"
    }


@app.get("/health")
def health():

    try:

        with db_connection() as db:
            db.execute("SELECT 1")

        database_status = "healthy"

    except Exception:

        database_status = "unavailable"

    return {
        "status": (
            "healthy"
            if database_status == "healthy"
            else "degraded"
        ),

        "database": database_status,
        "firebase_configured": FIREBASE_APP is not None,
        "ml_model_loaded": STRESS_MODEL is not None,
        "timestamp": iso()
    }


@app.get("/api/public-config")
def public_config():
    """Return Firebase browser configuration only; never expose Admin credentials."""
    web_ready = firebase_web_configured()
    admin_ready = FIREBASE_APP is not None

    return {
        "firebase_web_configured": web_ready,
        "firebase_auth_backend_ready": admin_ready,
        "firebase_sign_in_ready": web_ready and admin_ready,
        "firebase": (
            dict(FIREBASE_WEB_CONFIG)
            if web_ready and admin_ready
            else None
        )
    }


@app.get("/api/system/status")
def system_status():

    return {
        "backend": "online",
        "database": "sqlite",

        "authentication": (
            "firebase"
            if FIREBASE_APP
            else (
                "development-demo"
                if APP_ENV == "development"
                else "not_configured"
            )
        ),

        "firebase_firestore": (
            FIRESTORE_CLIENT is not None
        ),

        "firebase_web_configured": firebase_web_configured(),

        "firebase_sign_in_ready": (
            FIREBASE_APP is not None
            and firebase_web_configured()
        ),

        "ml_model_loaded": (
            STRESS_MODEL is not None
        ),

        "face_scan_analysis": "browser_local_landmark_detection_only",
        "timestamp": iso()
    }


# ============================================================
# 17. DEVICE PAIRING AND MANAGEMENT
# ============================================================

@app.post(
    "/api/devices/pair",
    status_code=status.HTTP_201_CREATED
)
def pair_device(
    payload: DevicePairIn,
    user_id: str = Depends(current_user_id)
):

    device_token = secrets.token_urlsafe(36)

    with db_connection() as db:

        existing = db.execute(
            """
            SELECT user_id
            FROM devices
            WHERE device_id = ?
            """,
            (payload.device_id,)
        ).fetchone()

        if (
            existing
            and existing["user_id"] != user_id
        ):

            raise HTTPException(
                status_code=409,
                detail="This device is already paired to another account."
            )

        db.execute(
            """
            INSERT INTO devices (
                device_id,
                user_id,
                device_name,
                token_hash,
                created_at,
                last_seen,
                active
            )
            VALUES (?, ?, ?, ?, ?, NULL, 1)

            ON CONFLICT(device_id)
            DO UPDATE SET
                device_name = excluded.device_name,
                token_hash = excluded.token_hash,
                created_at = excluded.created_at,
                active = 1
            """,
            (
                payload.device_id,
                user_id,
                payload.device_name,
                token_digest(device_token),
                iso()
            )
        )

    return {
        "success": True,
        "device_id": payload.device_id,
        "device_token": device_token,

        "warning": (
            "Copy this token now. It is shown only once. "
            "Store it securely on the ESP32; never commit it to GitHub."
        )
    }


@app.get("/api/devices")
def list_devices(
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT
                device_id,
                device_name,
                created_at,
                last_seen,
                active
            FROM devices
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,)
        ).fetchall()

    return {
        "devices": [
            dict(row) for row in rows
        ]
    }


@app.delete("/api/devices/{device_id}")
def unpair_device(
    device_id: str,
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        cursor = db.execute(
            """
            UPDATE devices
            SET active = 0
            WHERE device_id = ?
            AND user_id = ?
            """,
            (device_id, user_id)
        )

        if cursor.rowcount == 0:

            raise HTTPException(
                status_code=404,
                detail="Device not found for this account."
            )

    return {
        "success": True,
        "device_id": device_id,
        "active": False
    }


# ============================================================
# 18. SENSOR DATA APIS
# ============================================================

@app.post(
    "/api/sensors",
    status_code=status.HTTP_201_CREATED
)
def receive_user_sensor_reading(
    payload: SensorReadingIn,
    user_id: str = Depends(current_user_id)
):

    """Authenticated user endpoint for Swagger/manual test readings."""

    device_id = "manual-preview"

    result = store_sensor_reading(
        user_id,
        device_id,
        payload
    )

    return {
        "success": True,
        **result
    }


@app.post(
    "/api/devices/{device_id}/readings",
    status_code=status.HTTP_201_CREATED
)
def receive_device_reading(
    device_id: str,
    payload: SensorReadingIn,

    x_device_token: Annotated[
        str | None, Header()
    ] = None,
):

    if payload.source != "wearable":

        raise HTTPException(
            status_code=422,
            detail="Device endpoint accepts wearable readings only."
        )

    if x_device_token is None:

        raise HTTPException(
            status_code=401,
            detail="X-Device-Token header is required."
        )

    device = authenticated_device(
        device_id,
        x_device_token
    )

    result = store_sensor_reading(
        device["user_id"],
        device_id,
        payload
    )

    return {
        "success": True,
        **result
    }


@app.get("/api/sensors/latest")
def latest_reading(
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        row = db.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE user_id = ?
            ORDER BY timestamp DESC, received_at DESC
            LIMIT 1
            """,
            (user_id,)
        ).fetchone()

    data = row_dict(row)

    if data:

        data["ibi_ms"] = (
            json.loads(data.pop("ibi_ms_json"))
            if data.get("ibi_ms_json")
            else None
        )

    return {
        "data": data
    }


@app.get("/api/sensors/history")
def sensor_history(
    limit: int = Query(
        default=100,
        ge=1,
        le=500
    ),

    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (user_id, limit)
        ).fetchall()

    return {
        "total": len(rows),
        "readings": [
            dict(row) for row in rows
        ]
    }


# ============================================================
# 19. DEVICE CONNECTION STATUS
# ============================================================

@app.get("/api/device/status")
def device_status(
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT
                device_id,
                device_name,
                last_seen,
                active
            FROM devices
            WHERE user_id = ?
            AND active = 1
            """,
            (user_id,)
        ).fetchall()

    now = utc_now()
    devices = []

    for row in rows:

        last_seen = (
            parse_iso(row["last_seen"])
            if row["last_seen"]
            else None
        )

        connected = bool(
            last_seen
            and (
                now - last_seen
            ).total_seconds() <= 60
        )

        devices.append({
            **dict(row),
            "connected": connected
        })

    return {
        "devices": devices,
        "connected": any(
            device["connected"]
            for device in devices
        )
    }


# ============================================================
# 20. BASELINE AND STRESS ESTIMATION APIS
# ============================================================

@app.post("/api/baseline/calibrate")
def calibrate_user_baseline(
    lookback_minutes: int = Query(
        default=10,
        ge=5,
        le=30
    ),

    user_id: str = Depends(current_user_id)
):

    return {
        "success": True,
        "baseline": calibrate_baseline(
            user_id,
            lookback_minutes
        )
    }


@app.get("/api/baseline")
def get_user_baseline(
    user_id: str = Depends(current_user_id)
):

    baseline = baseline_for_user(user_id)

    return {
        "available": baseline is not None,
        "baseline": baseline
    }


@app.get("/api/stress/latest")
def latest_stress_estimate(
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        row = db.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE user_id = ?
            AND source = 'wearable'
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (user_id,)
        ).fetchone()

    if row is None:

        return {
            "status": "waiting_for_data",
            "stress_score": None
        }

    reading = dict(row)

    return predict_stress(
        user_id,
        reading
    )


# ============================================================
# 21. DASHBOARD OVERVIEW API
# ============================================================

@app.get("/api/dashboard/overview")
def dashboard_overview(
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        latest = db.execute(
            """
            SELECT *
            FROM sensor_readings
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (user_id,)
        ).fetchone()

        settings = db.execute(
            """
            SELECT haptic_enabled
            FROM user_settings
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

        device_rows = db.execute(
            """
            SELECT last_seen
            FROM devices
            WHERE user_id = ?
            AND active = 1
            """,
            (user_id,)
        ).fetchall()

    connected = any(
        row["last_seen"]
        and (
            utc_now() - parse_iso(row["last_seen"])
        ).total_seconds() <= 60
        for row in device_rows
    )

    reading = row_dict(latest)

    if (
        reading
        and reading["source"] == "wearable"
    ):

        prediction = predict_stress(
            user_id,
            reading
        )

    else:

        prediction = {
            "status": "waiting_for_wearable_data",
            "stress_score": None
        }

    return {
        "device_connected": connected,
        "latest_reading": reading,
        "stress": prediction,

        "haptic_enabled": (
            bool(settings["haptic_enabled"])
            if settings else False
        ),

        "updated_at": iso()
    }


# ============================================================
# 22. USER SETTINGS AND HAPTIC CONTROL
# ============================================================

@app.get("/api/settings")
def get_settings(
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        row = db.execute(
            """
            SELECT *
            FROM user_settings
            WHERE user_id = ?
            """,
            (user_id,)
        ).fetchone()

    if row is None:

        return {
            "haptic_enabled": False,
            "updated_at": None
        }

    return {
        "haptic_enabled": bool(
            row["haptic_enabled"]
        ),

        "updated_at": row["updated_at"]
    }


@app.put("/api/settings/haptic")
def set_haptic_setting(
    payload: HapticSettingIn,
    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        db.execute(
            """
            INSERT INTO user_settings (
                user_id,
                haptic_enabled,
                updated_at
            )
            VALUES (?, ?, ?)

            ON CONFLICT(user_id)
            DO UPDATE SET
                haptic_enabled=excluded.haptic_enabled,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                int(payload.haptic_enabled),
                iso()
            )
        )

    firestore_write(
        ("users", user_id, "settings"),
        "preferences",
        {
            "haptic_enabled": payload.haptic_enabled,
            "updated_at": iso()
        }
    )

    return {
        "success": True,
        "haptic_enabled": payload.haptic_enabled
    }


@app.post(
    "/api/haptic/commands",
    status_code=status.HTTP_201_CREATED
)
def create_haptic_command(
    payload: HapticCommandIn,
    user_id: str = Depends(current_user_id)
):

    settings = get_settings(user_id)

    if not settings["haptic_enabled"]:

        raise HTTPException(
            status_code=409,
            detail="Haptic feedback is OFF in user settings."
        )

    with db_connection() as db:

        device = db.execute(
            """
            SELECT device_id
            FROM devices
            WHERE device_id = ?
            AND user_id = ?
            AND active = 1
            """,
            (
                payload.device_id,
                user_id
            )
        ).fetchone()

        if device is None:

            raise HTTPException(
                status_code=404,
                detail="No active paired device found."
            )

        command_id = str(uuid.uuid4())

        db.execute(
            """
            INSERT INTO haptic_commands (
                command_id,
                user_id,
                device_id,
                duration_ms,
                command_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                command_id,
                user_id,
                payload.device_id,
                payload.duration_ms,
                "pending",
                iso()
            )
        )

    return {
        "success": True,
        "command_id": command_id,
        "status": "pending",
        "duration_ms": payload.duration_ms
    }


@app.get("/api/devices/{device_id}/commands")
def poll_device_commands(
    device_id: str,

    x_device_token: Annotated[
        str | None, Header()
    ] = None,
):

    if not x_device_token:

        raise HTTPException(
            status_code=401,
            detail="X-Device-Token header required."
        )

    device = authenticated_device(
        device_id,
        x_device_token
    )

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT
                command_id,
                duration_ms,
                created_at
            FROM haptic_commands
            WHERE device_id = ?
            AND user_id = ?
            AND command_status = 'pending'
            ORDER BY created_at ASC
            LIMIT 10
            """,
            (
                device_id,
                device["user_id"]
            )
        ).fetchall()

        for row in rows:

            db.execute(
                """
                UPDATE haptic_commands
                SET command_status = 'delivered',
                    delivered_at = ?
                WHERE command_id = ?
                """,
                (
                    iso(),
                    row["command_id"]
                )
            )

    return {
        "commands": [
            dict(row) for row in rows
        ]
    }


@app.post(
    "/api/devices/{device_id}/commands/{command_id}/ack"
)
def acknowledge_device_command(
    device_id: str,
    command_id: str,

    x_device_token: Annotated[
        str | None, Header()
    ] = None,
):

    if not x_device_token:

        raise HTTPException(
            status_code=401,
            detail="X-Device-Token header required."
        )

    device = authenticated_device(
        device_id,
        x_device_token
    )

    with db_connection() as db:

        cursor = db.execute(
            """
            UPDATE haptic_commands
            SET command_status = 'acknowledged',
                acknowledged_at = ?
            WHERE command_id = ?
            AND device_id = ?
            AND user_id = ?
            AND command_status = 'delivered'
            """,
            (
                iso(),
                command_id,
                device_id,
                device["user_id"]
            )
        )

        if cursor.rowcount == 0:

            raise HTTPException(
                status_code=404,
                detail="Command not found or already acknowledged."
            )

    return {
        "success": True,
        "status": "acknowledged"
    }


# ============================================================
# 23. RECOVERY SESSION AND BEFORE/AFTER COMPARISON
# ============================================================

@app.post(
    "/api/recovery/sessions",
    status_code=status.HTTP_201_CREATED
)
def save_recovery_session(
    payload: RecoverySessionIn,
    user_id: str = Depends(current_user_id)
):

    started = to_utc(
        payload.started_at
    )

    completed = (
        to_utc(payload.completed_at)
        if payload.completed_at
        else utc_now()
    )

    if completed < started:

        raise HTTPException(
            status_code=422,
            detail="completed_at cannot be earlier than started_at."
        )

    duration = int(
        (completed - started).total_seconds()
    )

    if duration > 7200:

        raise HTTPException(
            status_code=422,
            detail="Recovery session cannot exceed two hours."
        )

    with db_connection() as db:

        before = db.execute(
            """
            SELECT
                heart_rate_bpm,
                hrv_ms,
                gsr_raw,
                skin_temperature_c,
                timestamp
            FROM sensor_readings
            WHERE user_id = ?
            AND source = 'wearable'
            AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (
                user_id,
                iso(started)
            )
        ).fetchone()

        after = db.execute(
            """
            SELECT
                heart_rate_bpm,
                hrv_ms,
                gsr_raw,
                skin_temperature_c,
                timestamp
            FROM sensor_readings
            WHERE user_id = ?
            AND source = 'wearable'
            AND timestamp >= ?
            ORDER BY timestamp ASC
            LIMIT 1
            """,
            (
                user_id,
                iso(completed)
            )
        ).fetchone()

        session_id = str(uuid.uuid4())

        before_json = (
            json.dumps(dict(before))
            if before else None
        )

        after_json = (
            json.dumps(dict(after))
            if after else None
        )

        db.execute(
            """
            INSERT INTO recovery_sessions (
                session_id,
                user_id,
                activity,
                started_at,
                completed_at,
                duration_seconds,
                completed,
                before_reading_json,
                after_reading_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                user_id,
                payload.activity,
                iso(started),
                iso(completed),
                duration,
                int(payload.completed),
                before_json,
                after_json
            )
        )

    session_doc = {
        "session_id": session_id,
        "user_id": user_id,
        "activity": payload.activity,
        "started_at": iso(started),
        "completed_at": iso(completed),
        "duration_seconds": duration,
        "completed": payload.completed
    }

    firestore_write(
        ("users", user_id, "recovery_sessions"),
        session_id,
        session_doc
    )

    comparison = None

    if before and after:

        comparison = {}

        for field in (
            "heart_rate_bpm",
            "hrv_ms",
            "gsr_raw",
            "skin_temperature_c"
        ):

            before_value = before[field]
            after_value = after[field]

            comparison[field] = {
                "before": before_value,
                "after": after_value,

                "change": (
                    round(
                        after_value - before_value,
                        2
                    )
                    if (
                        before_value is not None
                        and after_value is not None
                    )
                    else None
                )
            }

    return {
        "success": True,
        "session": session_doc,
        "comparison": comparison
    }


@app.get("/api/recovery/history")
def recovery_history(
    limit: int = Query(
        default=50,
        ge=1,
        le=200
    ),

    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT
                session_id,
                activity,
                started_at,
                completed_at,
                duration_seconds,
                completed
            FROM recovery_sessions
            WHERE user_id = ?
            ORDER BY completed_at DESC
            LIMIT ?
            """,
            (
                user_id,
                limit
            )
        ).fetchall()

    return {
        "total": len(rows),
        "sessions": [
            dict(row) for row in rows
        ]
    }


# ============================================================
# 24. THREE-HOUR SUMMARY HISTORY
# ============================================================

@app.get("/api/summaries")
def summary_history(
    limit: int = Query(
        default=50,
        ge=1,
        le=200
    ),

    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT *
            FROM summaries
            WHERE user_id = ?
            ORDER BY period_end DESC
            LIMIT ?
            """,
            (
                user_id,
                limit
            )
        ).fetchall()

    return {
        "total": len(rows),
        "summaries": [
            dict(row) for row in rows
        ]
    }


# ============================================================
# 25. HIGH-STRESS EVENT HISTORY
# ============================================================

@app.get("/api/events/high-stress")
def high_stress_events(
    limit: int = Query(
        default=50,
        ge=1,
        le=200
    ),

    user_id: str = Depends(current_user_id)
):

    with db_connection() as db:

        rows = db.execute(
            """
            SELECT
                event_id,
                reading_id,
                stress_probability,
                occurred_at
            FROM high_stress_events
            WHERE user_id = ?
            ORDER BY occurred_at DESC
            LIMIT ?
            """,
            (
                user_id,
                limit
            )
        ).fetchall()

    return {
        "total": len(rows),
        "events": [
            dict(row) for row in rows
        ]
    }


# ============================================================
# 26. FACE SCAN STATUS AND PRIVACY
# ============================================================

@app.get("/api/face-scan/status")
def face_scan_status():

    return {
        "available": True,
        "status": "browser_local_landmark_detection_only",
        "processing": "on_device_browser",
        "camera_policy": (
            "Camera remains browser-side, "
            "user-triggered and OFF by default."
        ),
        "raw_frames_uploaded": False,
        "image_storage": False,
        "emotion_inference": False,
        "stress_inference": False,
        "message": (
            "Face landmarks and mesh overlays are processed "
            "locally in the browser. Webcam frames are not "
            "uploaded to the NeuroSense backend or stored. "
            "Emotion and stress inference are not included."
        )
    }


# ============================================================
# 27. SERVER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
