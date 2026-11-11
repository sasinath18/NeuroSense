
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
import re
import requests

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
from fastapi.responses import FileResponse, Response
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from pydantic import (
    BaseModel, ConfigDict, Field, model_validator
)

from scipy.signal import butter, find_peaks, sosfiltfilt

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials, firestore, storage as firebase_storage


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

FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET", "").strip()
MUSIC_STORAGE_DIR = Path(os.getenv(
    "NEUROSENSE_MUSIC_DIR", str(BASE_DIR / "data" / "music")
))
MAX_MUSIC_FILE_BYTES = 20 * 1024 * 1024
ALLOWED_MUSIC_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".webm"}

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
TWILIO_WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM", "").strip()
GOOGLE_DRIVE_API_KEY = os.getenv("GOOGLE_DRIVE_API_KEY", "").strip()
GOOGLE_DRIVE_APP_ID = os.getenv("GOOGLE_DRIVE_APP_ID", "").strip()

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
            music_mode TEXT NOT NULL DEFAULT 'manual',
            warning_sound_enabled INTEGER NOT NULL DEFAULT 0,
            notifications_enabled INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS music_tracks (
            track_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            file_name TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            storage_backend TEXT NOT NULL,
            storage_path TEXT NOT NULL,
            is_default INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_music_tracks_user
        ON music_tracks(user_id, created_at);

        CREATE TABLE IF NOT EXISTS blink_summaries (
            summary_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            blink_count INTEGER NOT NULL,
            duration_seconds INTEGER NOT NULL,
            blinks_per_minute REAL NOT NULL,
            recorded_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_blink_user_time
        ON blink_summaries(user_id, recorded_at);

        CREATE TABLE IF NOT EXISTS emergency_contacts (
            user_id TEXT PRIMARY KEY,
            phone TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 0,
            last_alert_at TEXT,
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

        # SQLite CREATE TABLE IF NOT EXISTS does not add columns to an
        # existing database. Apply additive migrations for user preferences.
        existing_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(user_settings)").fetchall()
        }
        migrations = {
            "music_mode": "TEXT NOT NULL DEFAULT 'manual'",
            "warning_sound_enabled": "INTEGER NOT NULL DEFAULT 0",
            "notifications_enabled": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, definition in migrations.items():
            if column not in existing_columns:
                db.execute(
                    f"ALTER TABLE user_settings ADD COLUMN {column} {definition}"
                )


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


def firestore_read_document(
    collection_path: tuple[str, ...],
    document_id: str
) -> dict | None:

    if FIRESTORE_CLIENT is None:
        return None

    if not collection_path or len(collection_path) % 2 == 0:
        logger.error("Invalid Firestore collection path: %s", collection_path)
        return None

    try:
        reference = FIRESTORE_CLIENT.collection(collection_path[0])

        for index, segment in enumerate(collection_path[1:], start=1):
            if index % 2 == 1:
                reference = reference.document(segment)
            else:
                reference = reference.collection(segment)

        snapshot = reference.document(document_id).get()

        if not snapshot.exists:
            return None

        return snapshot.to_dict() or {}

    except Exception:
        logger.exception(
            "Firestore document read failed for %s/%s",
            collection_path,
            document_id
        )
        return None


def firestore_read_collection(
    collection_path: tuple[str, ...],
    limit: int = 100,
    order_by: str | None = None,
    descending: bool = True
) -> list[dict]:

    if FIRESTORE_CLIENT is None:
        return []

    # A collection path alternates collection/document segments and
    # therefore contains an odd number of segments, e.g.
    # ("users", user_id, "sensor_readings").
    if not collection_path or len(collection_path) % 2 == 0:
        logger.error("Invalid Firestore collection path: %s", collection_path)
        return []

    try:
        reference = FIRESTORE_CLIENT.collection(collection_path[0])

        for index, segment in enumerate(collection_path[1:], start=1):
            if index % 2 == 1:
                reference = reference.document(segment)
            else:
                reference = reference.collection(segment)

        if order_by:
            direction = (
                firestore.Query.DESCENDING
                if descending
                else firestore.Query.ASCENDING
            )
            reference = reference.order_by(
                order_by,
                direction=direction
            )

        documents = reference.limit(limit).stream()
        results = []

        for document in documents:
            data = document.to_dict() or {}
            data.setdefault("document_id", document.id)
            results.append(data)

        return results

    except Exception:
        logger.exception(
            "Firestore collection read failed for %s",
            collection_path
        )
        return []




# ============================================================
# 7A. OPTIONAL CLOUD MUSIC AND WHATSAPP HELPERS
# ============================================================

def configured_music_bucket():
    if FIREBASE_APP is None or not FIREBASE_STORAGE_BUCKET:
        return None
    try:
        return firebase_storage.bucket(
            name=FIREBASE_STORAGE_BUCKET,
            app=FIREBASE_APP
        )
    except Exception:
        logger.exception("Firebase Storage bucket is unavailable.")
        return None


def send_whatsapp_message(phone: str, message: str) -> str:
    if not all((TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM)):
        return "not_configured"
    try:
        response = requests.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Messages.json",
            data={
                "From": TWILIO_WHATSAPP_FROM,
                "To": f"whatsapp:{phone}",
                "Body": message,
            },
            auth=(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
            timeout=12,
        )
        response.raise_for_status()
        return "sent"
    except requests.RequestException:
        logger.exception("WhatsApp message delivery failed.")
        return "failed"


def maybe_send_whatsapp_alert(user_id: str, probability: float) -> str:
    """Send only after repeated elevated estimates, with consent and cooldown."""
    if probability < HIGH_STRESS_THRESHOLD:
        return "not_triggered"

    with db_connection() as db:
        contact = db.execute(
            "SELECT * FROM emergency_contacts WHERE user_id = ? AND enabled = 1",
            (user_id,),
        ).fetchone()
        if contact is None:
            return "not_enabled"

        recent = db.execute(
            """
            SELECT COUNT(*) AS total
            FROM high_stress_events
            WHERE user_id = ? AND occurred_at >= ?
            """,
            (user_id, iso(utc_now() - timedelta(minutes=30))),
        ).fetchone()["total"]

        if int(recent) < 3:
            return "awaiting_repeated_readings"

        if contact["last_alert_at"]:
            last_sent = parse_iso(contact["last_alert_at"])
            if utc_now() - last_sent < timedelta(hours=6):
                return "cooldown"

        phone = contact["phone"]

    delivery = send_whatsapp_message(
        phone,
        "NeuroSense check-in: your account has recorded repeated elevated stress-related readings. "
        "If you are available, please check in with the user. This is an experimental wellness alert, "
        "not a medical diagnosis or emergency service.",
    )
    if delivery == "sent":
        with db_connection() as db:
            db.execute(
                "UPDATE emergency_contacts SET last_alert_at = ?, updated_at = ? WHERE user_id = ?",
                (iso(), iso(), user_id),
            )
    return delivery


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


class RecoveryPreferencesIn(StrictModel):
    music_mode: Literal["manual", "automatic"] = "manual"
    warning_sound_enabled: bool = False
    notifications_enabled: bool = False


class BlinkSummaryIn(StrictModel):
    blink_count: int = Field(ge=0, le=2000)
    duration_seconds: int = Field(ge=10, le=3600)


class EmergencyContactIn(StrictModel):
    phone: str = Field(pattern=r"^\+[1-9]\d{7,14}$")
    enabled: bool = False


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

def sync_device_record(device: dict) -> dict:
    """Back up device metadata without ever storing the one-time plaintext token."""
    device_id = str(device.get("device_id") or "")
    user_id = str(device.get("user_id") or "")
    if not device_id or not user_id:
        return {"global": "failed", "user": "failed"}
    payload = {
        "device_id": device_id,
        "user_id": user_id,
        "device_name": str(device.get("device_name") or "NeuroSense ESP32"),
        "token_hash": str(device.get("token_hash") or ""),
        "created_at": str(device.get("created_at") or iso()),
        "last_seen": device.get("last_seen"),
        "active": bool(device.get("active", 1)),
    }
    return {
        "global": firestore_write(("devices",), device_id, payload),
        "user": firestore_write(("users", user_id, "devices"), device_id, payload),
    }


def restore_cloud_devices(user_id: str) -> int:
    """Restore the signed-in user's device metadata after an ephemeral DB reset."""
    cloud_devices = firestore_read_collection(
        ("users", user_id, "devices"),
        limit=100,
    )
    restored = 0
    for item in cloud_devices:
        device_id = str(item.get("device_id") or item.get("document_id") or "")
        token_hash = str(item.get("token_hash") or "")
        owner = str(item.get("user_id") or user_id)
        if not device_id or not token_hash or owner != user_id:
            continue
        try:
            with db_connection() as db:
                db.execute(
                    """
                    INSERT INTO devices (
                        device_id, user_id, device_name, token_hash,
                        created_at, last_seen, active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(device_id) DO UPDATE SET
                        device_name=excluded.device_name,
                        token_hash=excluded.token_hash,
                        created_at=excluded.created_at,
                        last_seen=excluded.last_seen,
                        active=excluded.active
                    WHERE devices.user_id=excluded.user_id
                    """,
                    (
                        device_id, owner,
                        str(item.get("device_name") or "NeuroSense ESP32"),
                        token_hash, str(item.get("created_at") or iso()),
                        item.get("last_seen"),
                        int(bool(item.get("active", True))),
                    ),
                )
            restored += 1
        except Exception:
            logger.exception("Could not restore cloud device %s.", device_id)
    return restored


def authenticated_device(
    device_id: str,
    device_token: str
) -> sqlite3.Row | dict:
    if not device_token or len(device_token) < 24:
        raise HTTPException(
            status_code=401,
            detail="Valid X-Device-Token header required."
        )

    with db_connection() as db:
        row = db.execute(
            "SELECT * FROM devices WHERE device_id = ? AND active = 1",
            (device_id,),
        ).fetchone()

    # Restore a valid device record from the server-only cloud ownership index
    # when Render's ephemeral SQLite filesystem has been reset.
    if row is None:
        cloud_device = firestore_read_document(("devices",), device_id)
        if (
            cloud_device
            and bool(cloud_device.get("active", False))
            and cloud_device.get("token_hash")
        ):
            with db_connection() as db:
                db.execute(
                    """
                    INSERT OR IGNORE INTO devices (
                        device_id, user_id, device_name, token_hash,
                        created_at, last_seen, active
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        device_id,
                        str(cloud_device.get("user_id") or ""),
                        str(cloud_device.get("device_name") or "NeuroSense ESP32"),
                        str(cloud_device["token_hash"]),
                        str(cloud_device.get("created_at") or iso()),
                        cloud_device.get("last_seen"),
                    ),
                )
                row = db.execute(
                    "SELECT * FROM devices WHERE device_id = ? AND active = 1",
                    (device_id,),
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

    # Authenticated polling is also a device heartbeat. Throttle writes so
    # a one-second command poll does not create a database write each time.
    now = utc_now()
    last_seen = parse_iso(row["last_seen"]) if row["last_seen"] else None
    if last_seen is None or (now - last_seen).total_seconds() >= 15:
        with db_connection() as db:
            db.execute(
                """
                UPDATE devices SET last_seen = ?
                WHERE device_id = ? AND user_id = ? AND active = 1
                """,
                (iso(now), device_id, row["user_id"]),
            )
            updated = db.execute(
                "SELECT * FROM devices WHERE device_id = ? AND active = 1",
                (device_id,),
            ).fetchone()
        if updated is not None:
            row = updated
            sync_device_record(dict(row))

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

    local_baseline = row_dict(row)

    if local_baseline is not None:
        return local_baseline

    return firestore_read_document(
        ("users", user_id, "baselines"),
        "current"
    )


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
        cloud_rows = firestore_read_collection(
            ("users", user_id, "sensor_readings"),
            limit=500,
            order_by="timestamp",
        )
        local_ids = {str(row["id"]) for row in rows}
        cloud_recent = [
            item for item in cloud_rows
            if str(item.get("source") or "") == "wearable"
            and str(item.get("signal_quality") or "") in {"good", "moderate"}
            and item.get("timestamp")
            and str(item.get("timestamp")) >= since
            and str(item.get("id") or item.get("document_id") or "") not in local_ids
        ]
        rows = [dict(row) for row in rows] + cloud_recent
        rows.sort(key=lambda item: str(item.get("timestamp") or ""), reverse=True)

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
    """Build model inputs while preserving unavailable sensors as NaN.

    The fitted preprocessing pipeline contains a median imputer. Missing
    optional measurements must reach that imputer rather than being encoded
    as a physiological value of zero.
    """

    def number_or_nan(value) -> float:
        return float(value) if value is not None else float("nan")

    def delta(field: str) -> float:
        current = reading.get(field)
        base = baseline.get(field)
        if current is None or base is None:
            return float("nan")
        return float(current) - float(base)

    return {
        "heart_rate_bpm": number_or_nan(reading.get("heart_rate_bpm")),
        "hrv_ms": number_or_nan(reading.get("hrv_ms")),
        "gsr_raw": number_or_nan(reading.get("gsr_raw")),
        "skin_temperature_c": number_or_nan(reading.get("skin_temperature_c")),
        "heart_rate_delta": delta("heart_rate_bpm"),
        "hrv_delta": delta("hrv_ms"),
        "gsr_delta": delta("gsr_raw"),
        "skin_temperature_delta": delta("skin_temperature_c"),
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

                firestore_write(
                    ("users", user_id, "high_stress_events"),
                    event_id,
                    {
                        "event_id": event_id,
                        "user_id": user_id,
                        "reading_id": reading["id"],
                        "stress_probability": probability,
                        "occurred_at": reading["timestamp"]
                    }
                )

                result["whatsapp_alert"] = maybe_send_whatsapp_alert(
                    user_id, probability
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

    if device_id != "manual-preview":
        device_record = firestore_read_document(("devices",), device_id)
        if device_record and str(device_record.get("user_id") or "") == user_id:
            device_record["last_seen"] = payload["received_at"]
            sync_device_record(device_record)

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
        ),
        "google_drive_picker_configured": bool(
            GOOGLE_DRIVE_API_KEY and GOOGLE_DRIVE_APP_ID
        ),
        "google_drive_api_key": GOOGLE_DRIVE_API_KEY or None,
        "google_drive_app_id": GOOGLE_DRIVE_APP_ID or None,
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
            "SELECT user_id FROM devices WHERE device_id = ?",
            (payload.device_id,),
        ).fetchone()

    if existing is not None and existing["user_id"] != user_id:
        raise HTTPException(
            status_code=409,
            detail="This device is already paired to another account."
        )

    # Firestore's root ownership record prevents another account from
    # claiming the same device after an ephemeral SQLite reset.
    cloud_existing = firestore_read_document(("devices",), payload.device_id)
    if (
        existing is None
        and cloud_existing
        and cloud_existing.get("active", False)
        and str(cloud_existing.get("user_id") or "") != user_id
    ):
        raise HTTPException(
            status_code=409,
            detail="This device is already paired to another account."
        )

    record = {
        "device_id": payload.device_id,
        "user_id": user_id,
        "device_name": payload.device_name,
        "token_hash": token_digest(device_token),
        "created_at": iso(),
        "last_seen": None,
        "active": 1,
    }
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO devices (
                device_id, user_id, device_name, token_hash,
                created_at, last_seen, active
            )
            VALUES (?, ?, ?, ?, ?, NULL, 1)
            ON CONFLICT(device_id)
            DO UPDATE SET
                device_name=excluded.device_name,
                token_hash=excluded.token_hash,
                created_at=excluded.created_at,
                last_seen=NULL,
                active=1
            WHERE devices.user_id=excluded.user_id
            """,
            (
                record["device_id"], record["user_id"],
                record["device_name"], record["token_hash"],
                record["created_at"],
            ),
        )

    cloud_sync = sync_device_record(record)
    return {
        "success": True,
        "device_id": payload.device_id,
        "device_token": device_token,
        "cloud_sync": cloud_sync,
        "warning": (
            "Copy this token now. It is shown only once. "
            "Store it securely on the ESP32; never commit it to GitHub."
        ),
    }


@app.get("/api/devices")
def list_devices(
    user_id: str = Depends(current_user_id)
):
    restore_cloud_devices(user_id)
    with db_connection() as db:
        rows = db.execute(
            """
            SELECT device_id, device_name, created_at, last_seen, active
            FROM devices
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()
    return {"devices": [dict(row) for row in rows]}


@app.delete("/api/devices/{device_id}")
def unpair_device(
    device_id: str,
    user_id: str = Depends(current_user_id)
):
    with db_connection() as db:
        cursor = db.execute(
            """
            UPDATE devices SET active = 0
            WHERE device_id = ? AND user_id = ?
            """,
            (device_id, user_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Device not found for this account."
            )
        row = db.execute(
            "SELECT * FROM devices WHERE device_id = ? AND user_id = ?",
            (device_id, user_id),
        ).fetchone()

    cloud_sync = sync_device_record(dict(row))
    return {
        "success": True,
        "device_id": device_id,
        "active": False,
        "cloud_sync": cloud_sync,
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

    else:
        cloud_rows = firestore_read_collection(
            ("users", user_id, "sensor_readings"),
            limit=1,
            order_by="timestamp"
        )
        data = cloud_rows[0] if cloud_rows else None

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

    combined = []
    seen_ids = set()

    for row in rows:
        item = dict(row)
        item["ibi_ms"] = (
            json.loads(item.pop("ibi_ms_json"))
            if item.get("ibi_ms_json")
            else None
        )
        combined.append(item)
        seen_ids.add(item["id"])

    # Merge Firestore copies to recover readings after an ephemeral
    # Render filesystem restart; SQLite remains the preferred fast path.
    cloud_rows = firestore_read_collection(
        ("users", user_id, "sensor_readings"),
        limit=limit,
        order_by="timestamp"
    )

    for item in cloud_rows:
        reading_id = item.get("id") or item.get("document_id")
        if reading_id and reading_id not in seen_ids:
            item.setdefault("id", reading_id)
            combined.append(item)
            seen_ids.add(reading_id)

    combined.sort(
        key=lambda item: item.get("timestamp") or "",
        reverse=True
    )
    combined = combined[:limit]

    return {
        "total": len(combined),
        "readings": combined
    }


# ============================================================
# 19. DEVICE CONNECTION STATUS
# ============================================================

@app.get("/api/device/status")
def device_status(
    user_id: str = Depends(current_user_id)
):

    restore_cloud_devices(user_id)

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
        cloud_rows = firestore_read_collection(
            ("users", user_id, "sensor_readings"),
            limit=1,
            order_by="timestamp",
        )
        if not cloud_rows:
            return {
                "status": "waiting_for_data",
                "stress_score": None
            }
        reading = cloud_rows[0]
    else:
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

    restore_cloud_devices(user_id)

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

    if settings is None:
        cloud_settings = firestore_read_document(
            ("users", user_id, "settings"),
            "preferences"
        )
        if cloud_settings is not None:
            settings = {
                "haptic_enabled": int(
                    bool(cloud_settings.get("haptic_enabled", False))
                )
            }

    reading = row_dict(latest)

    if reading is None:
        cloud_readings = firestore_read_collection(
            ("users", user_id, "sensor_readings"),
            limit=1,
            order_by="timestamp"
        )
        reading = cloud_readings[0] if cloud_readings else None

    if (
        reading
        and reading.get("source") == "wearable"
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
            "SELECT * FROM user_settings WHERE user_id = ?",
            (user_id,)
        ).fetchone()

    if row is None:
        values = firestore_read_document(
            ("users", user_id, "settings"), "preferences"
        ) or {}
        return {
            "haptic_enabled": bool(values.get("haptic_enabled", False)),
            "music_mode": values.get("music_mode", "manual"),
            "warning_sound_enabled": bool(values.get("warning_sound_enabled", False)),
            "notifications_enabled": bool(values.get("notifications_enabled", False)),
            "updated_at": values.get("updated_at"),
        }

    return {
        "haptic_enabled": bool(row["haptic_enabled"]),
        "music_mode": row["music_mode"],
        "warning_sound_enabled": bool(row["warning_sound_enabled"]),
        "notifications_enabled": bool(row["notifications_enabled"]),
        "updated_at": row["updated_at"],
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

    current_preferences = get_settings(user_id)
    cloud_sync = firestore_write(
        ("users", user_id, "settings"),
        "preferences",
        {
            "haptic_enabled": payload.haptic_enabled,
            "music_mode": current_preferences.get("music_mode", "manual"),
            "warning_sound_enabled": current_preferences.get("warning_sound_enabled", False),
            "notifications_enabled": current_preferences.get("notifications_enabled", False),
            "updated_at": iso()
        }
    )

    return {
        "success": True,
        "haptic_enabled": payload.haptic_enabled,
        "cloud_sync": cloud_sync,
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
            SELECT device_id FROM devices
            WHERE device_id = ? AND user_id = ? AND active = 1
            """,
            (payload.device_id, user_id),
        ).fetchone()
        if device is None:
            raise HTTPException(
                status_code=404,
                detail="No active paired device found."
            )

        command_id = str(uuid.uuid4())
        created_at = iso()
        command = {
            "command_id": command_id,
            "user_id": user_id,
            "device_id": payload.device_id,
            "duration_ms": payload.duration_ms,
            "command_status": "pending",
            "created_at": created_at,
            "delivered_at": None,
            "acknowledged_at": None,
        }
        db.execute(
            """
            INSERT INTO haptic_commands (
                command_id, user_id, device_id, duration_ms,
                command_status, created_at, delivered_at, acknowledged_at
            )
            VALUES (?, ?, ?, ?, 'pending', ?, NULL, NULL)
            """,
            (
                command_id, user_id, payload.device_id,
                payload.duration_ms, created_at,
            ),
        )

    cloud_sync = firestore_write(
        ("users", user_id, "haptic_commands"),
        command_id,
        command,
    )
    return {
        "success": True,
        "command_id": command_id,
        "status": "pending",
        "duration_ms": payload.duration_ms,
        "cloud_sync": cloud_sync,
    }


@app.get("/api/devices/{device_id}/commands")
def poll_device_commands(
    device_id: str,
    x_device_token: Annotated[str | None, Header()] = None,
):
    if not x_device_token:
        raise HTTPException(
            status_code=401,
            detail="X-Device-Token header required."
        )

    device = authenticated_device(device_id, x_device_token)
    user_id = str(device["user_id"])

    # Re-check the user's current preference at delivery time as well as
    # creation time, so a queued command cannot fire after opting out.
    if not get_settings(user_id)["haptic_enabled"]:
        return {"commands": []}

    # Rehydrate the pending queue after a Render restart.
    cloud_commands = firestore_read_collection(
        ("users", user_id, "haptic_commands"),
        limit=100,
    )
    with db_connection() as db:
        for item in cloud_commands:
            if (
                str(item.get("device_id") or "") != device_id
                or str(item.get("command_status") or "") not in {"pending", "delivered"}
                or not item.get("command_id")
            ):
                continue
            db.execute(
                """
                INSERT OR IGNORE INTO haptic_commands (
                    command_id, user_id, device_id, duration_ms,
                    command_status, created_at, delivered_at, acknowledged_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(item["command_id"]), user_id, device_id,
                    int(item.get("duration_ms") or 500),
                    str(item.get("command_status") or "pending"),
                    str(item.get("created_at") or iso()),
                    item.get("delivered_at"),
                    item.get("acknowledged_at"),
                ),
            )

        expiry_cutoff = iso(utc_now() - timedelta(seconds=60))
        db.execute(
            """
            UPDATE haptic_commands
            SET command_status = 'expired'
            WHERE device_id = ?
            AND user_id = ?
            AND command_status = 'pending'
            AND created_at < ?
            """,
            (device_id, user_id, expiry_cutoff),
        )
        rows = db.execute(
            """
            SELECT command_id, duration_ms, created_at
            FROM haptic_commands
            WHERE device_id = ?
            AND user_id = ?
            AND command_status = 'pending'
            AND created_at >= ?
            ORDER BY created_at ASC
            LIMIT 10
            """,
            (device_id, user_id, expiry_cutoff),
        ).fetchall()

        delivered_at = iso()
        for row in rows:
            db.execute(
                """
                UPDATE haptic_commands
                SET command_status = 'delivered', delivered_at = ?
                WHERE command_id = ?
                """,
                (delivered_at, row["command_id"]),
            )

    response_commands = []
    for row in rows:
        firestore_write(
            ("users", user_id, "haptic_commands"),
            row["command_id"],
            {
                "command_id": row["command_id"],
                "user_id": user_id,
                "device_id": device_id,
                "duration_ms": row["duration_ms"],
                "command_status": "delivered",
                "created_at": row["created_at"],
                "delivered_at": delivered_at,
                "acknowledged_at": None,
            },
        )
        response_commands.append({
            "command_id": row["command_id"],
            "duration_ms": row["duration_ms"],
            "created_at": row["created_at"],
        })

    return {"commands": response_commands}


@app.get("/api/haptic/commands/{command_id}")
def haptic_command_status(
    command_id: str,
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        row = db.execute(
            """
            SELECT command_id, device_id, duration_ms, command_status,
                   created_at, delivered_at, acknowledged_at
            FROM haptic_commands
            WHERE command_id = ? AND user_id = ?
            """,
            (command_id, user_id),
        ).fetchone()

    command = dict(row) if row is not None else None
    if command is None:
        command = firestore_read_document(
            ("users", user_id, "haptic_commands"),
            command_id,
        )

    if command is None or str(command.get("user_id") or user_id) != user_id:
        raise HTTPException(
            status_code=404,
            detail="Haptic command not found for this account."
        )

    return {
        "command_id": command_id,
        "device_id": command.get("device_id"),
        "duration_ms": command.get("duration_ms"),
        "status": command.get("command_status", "unknown"),
        "created_at": command.get("created_at"),
        "delivered_at": command.get("delivered_at"),
        "acknowledged_at": command.get("acknowledged_at"),
    }


@app.post(
    "/api/devices/{device_id}/commands/{command_id}/ack"
)
def acknowledge_device_command(
    device_id: str,
    command_id: str,
    x_device_token: Annotated[str | None, Header()] = None,
):
    if not x_device_token:
        raise HTTPException(
            status_code=401,
            detail="X-Device-Token header required."
        )

    device = authenticated_device(device_id, x_device_token)
    user_id = str(device["user_id"])

    with db_connection() as db:
        cursor = db.execute(
            """
            UPDATE haptic_commands
            SET command_status = 'acknowledged', acknowledged_at = ?
            WHERE command_id = ? AND device_id = ?
            AND user_id = ? AND command_status = 'delivered'
            """,
            (iso(), command_id, device_id, user_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Command not found or already acknowledged."
            )
        command_row = db.execute(
            "SELECT * FROM haptic_commands WHERE command_id = ?",
            (command_id,),
        ).fetchone()

    cloud_sync = firestore_write(
        ("users", user_id, "haptic_commands"),
        command_id,
        dict(command_row),
    )
    return {
        "success": True,
        "status": "acknowledged",
        "cloud_sync": cloud_sync,
    }





# ============================================================
# 23. RECOVERY SESSION AND BEFORE/AFTER COMPARISON
# ============================================================

def build_recovery_comparison(before, after) -> dict | None:

    if before is None or after is None:
        return None

    comparison = {
        "before_timestamp": before["timestamp"],
        "after_timestamp": after["timestamp"]
    }

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
                round(after_value - before_value, 2)
                if (
                    before_value is not None
                    and after_value is not None
                )
                else None
            )
        }

    return comparison


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

    comparison = build_recovery_comparison(
        before,
        after
    )

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
                completed,
                before_reading_json,
                after_reading_json
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

        sessions = []

        for row in rows:

            item = dict(row)

            before = (
                json.loads(item["before_reading_json"])
                if item.get("before_reading_json")
                else None
            )

            after = (
                json.loads(item["after_reading_json"])
                if item.get("after_reading_json")
                else None
            )

            # A post-session reading may arrive after the recovery
            # record is saved. Re-query it when history is requested.
            if before is None:
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
                        item["started_at"]
                    )
                ).fetchone()

            if after is None:
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
                        item["completed_at"]
                    )
                ).fetchone()

            item.pop("before_reading_json", None)
            item.pop("after_reading_json", None)
            item["comparison"] = build_recovery_comparison(
                before,
                after
            )

            sessions.append(item)

    cloud_sessions = firestore_read_collection(
        ("users", user_id, "recovery_sessions"),
        limit=limit,
        order_by="completed_at"
    )
    merged_sessions = {
        str(item.get("session_id") or item.get("document_id")): item
        for item in cloud_sessions
        if item.get("session_id") or item.get("document_id")
    }
    for item in sessions:
        merged_sessions[str(item["session_id"])] = item
    sessions = sorted(
        merged_sessions.values(),
        key=lambda item: str(item.get("completed_at") or ""),
        reverse=True,
    )[:limit]
    for session in sessions:
        session.setdefault("comparison", None)

    return {
        "total": len(sessions),
        "sessions": sessions
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

    cloud_summaries = firestore_read_collection(
        ("users", user_id, "summaries"),
        limit=limit,
        order_by="period_end"
    )
    merged_summaries = {
        str(item.get("summary_id") or item.get("document_id")): item
        for item in cloud_summaries
        if item.get("summary_id") or item.get("document_id")
    }
    for item in [dict(row) for row in rows]:
        merged_summaries[str(item["summary_id"])] = item
    summaries = sorted(
        merged_summaries.values(),
        key=lambda item: str(item.get("period_end") or ""),
        reverse=True,
    )[:limit]

    return {
        "total": len(summaries),
        "summaries": summaries
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

    cloud_events = firestore_read_collection(
        ("users", user_id, "high_stress_events"),
        limit=limit,
        order_by="occurred_at"
    )
    merged_events = {
        str(item.get("event_id") or item.get("document_id")): item
        for item in cloud_events
        if item.get("event_id") or item.get("document_id")
    }
    for item in [dict(row) for row in rows]:
        merged_events[str(item["event_id"])] = item
    events = sorted(
        merged_events.values(),
        key=lambda item: str(item.get("occurred_at") or ""),
        reverse=True,
    )[:limit]

    return {
        "total": len(events),
        "events": events
    }




@app.put("/api/settings/recovery")
def set_recovery_preferences(
    payload: RecoveryPreferencesIn,
    user_id: str = Depends(current_user_id)
):
    updated_at = iso()
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO user_settings (
                user_id, haptic_enabled, music_mode,
                warning_sound_enabled, notifications_enabled, updated_at
            )
            VALUES (?, 0, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                music_mode=excluded.music_mode,
                warning_sound_enabled=excluded.warning_sound_enabled,
                notifications_enabled=excluded.notifications_enabled,
                updated_at=excluded.updated_at
            """,
            (
                user_id, payload.music_mode,
                int(payload.warning_sound_enabled),
                int(payload.notifications_enabled), updated_at,
            ),
        )
    cloud_sync = firestore_write(
        ("users", user_id, "settings"), "preferences",
        {
            "haptic_enabled": get_settings(user_id)["haptic_enabled"],
            "music_mode": payload.music_mode,
            "warning_sound_enabled": payload.warning_sound_enabled,
            "notifications_enabled": payload.notifications_enabled,
            "updated_at": updated_at,
        },
    )
    return {
        "success": True,
        "music_mode": payload.music_mode,
        "warning_sound_enabled": payload.warning_sound_enabled,
        "notifications_enabled": payload.notifications_enabled,
        "cloud_sync": cloud_sync,
    }


@app.get("/api/music")
def list_music_tracks(user_id: str = Depends(current_user_id)):
    with db_connection() as db:
        rows = db.execute(
            "SELECT * FROM music_tracks WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    tracks = {str(row["track_id"]): dict(row) for row in rows}
    cloud_tracks = firestore_read_collection(
        ("users", user_id, "music_tracks"), limit=200, order_by="created_at"
    )
    for item in cloud_tracks:
        track_id = str(item.get("track_id") or item.get("document_id") or "")
        if track_id:
            tracks.setdefault(track_id, item)
            if track_id not in {str(row["track_id"]) for row in rows}:
                try:
                    with db_connection() as db:
                        db.execute(
                            """
                            INSERT OR IGNORE INTO music_tracks (
                                track_id, user_id, file_name, content_type, size_bytes,
                                storage_backend, storage_path, is_default, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                track_id, user_id, item.get("file_name", "audio"),
                                item.get("content_type", "audio/mpeg"),
                                int(item.get("size_bytes", 0)),
                                item.get("storage_backend", "firebase_storage"),
                                item.get("storage_path", ""),
                                int(bool(item.get("is_default", False))),
                                item.get("created_at", iso()),
                            ),
                        )
                except Exception:
                    logger.exception("Could not restore cloud music metadata for %s", track_id)
    ordered = sorted(
        tracks.values(), key=lambda item: str(item.get("created_at") or ""), reverse=True
    )
    return {
        "tracks": [
            {
                "track_id": item["track_id"],
                "file_name": item["file_name"],
                "content_type": item["content_type"],
                "size_bytes": item["size_bytes"],
                "is_default": bool(item.get("is_default", False)),
                "created_at": item.get("created_at"),
                "storage_backend": item.get("storage_backend", "firebase_storage"),
            }
            for item in ordered
        ],
        "storage_configured": configured_music_bucket() is not None,
        "note": "Cloud music survives service restarts when Firebase Storage is configured; local fallback files may be ephemeral on hosted services.",
    }


@app.post("/api/music/upload", status_code=status.HTTP_201_CREATED)
async def upload_music_track(
    file: UploadFile = File(...),
    user_id: str = Depends(current_user_id),
):
    original_name = Path(file.filename or "track").name
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_MUSIC_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported audio format.")
    content = await file.read(MAX_MUSIC_FILE_BYTES + 1)
    if not content or len(content) > MAX_MUSIC_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Audio file must be between 1 byte and 20 MB.")
    content_type = file.content_type or "application/octet-stream"
    if not content_type.startswith("audio/") and content_type != "application/octet-stream":
        raise HTTPException(status_code=415, detail="Uploaded file is not recognized as audio.")

    track_id = str(uuid.uuid4())
    safe_user = re.sub(r"[^A-Za-z0-9_.-]", "_", user_id)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)[:120]
    storage_path = f"users/{safe_user}/music/{track_id}/{safe_name}"
    bucket = configured_music_bucket()

    if bucket is not None:
        try:
            bucket.blob(storage_path).upload_from_string(
                content, content_type=content_type
            )
            storage_backend = "firebase_storage"
        except Exception:
            logger.exception("Firebase Storage upload failed.")
            raise HTTPException(status_code=503, detail="Cloud music storage upload failed.")
    else:
        user_dir = MUSIC_STORAGE_DIR / hashlib.sha256(user_id.encode()).hexdigest()
        user_dir.mkdir(parents=True, exist_ok=True)
        local_path = user_dir / f"{track_id}{extension}"
        local_path.write_bytes(content)
        storage_path = str(local_path)
        storage_backend = "local_ephemeral" if APP_ENV == "production" else "local"

    track = {
        "track_id": track_id,
        "user_id": user_id,
        "file_name": safe_name,
        "content_type": content_type,
        "size_bytes": len(content),
        "storage_backend": storage_backend,
        "storage_path": storage_path,
        "is_default": False,
        "created_at": iso(),
    }
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO music_tracks (
                track_id, user_id, file_name, content_type, size_bytes,
                storage_backend, storage_path, is_default, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                track_id, user_id, safe_name, content_type, len(content),
                storage_backend, storage_path, track["created_at"],
            ),
        )
    cloud_sync = firestore_write(
        ("users", user_id, "music_tracks"), track_id, track
    )
    return {
        "success": True,
        "track": {
            "track_id": track_id,
            "file_name": safe_name,
            "content_type": content_type,
            "size_bytes": len(content),
            "is_default": False,
            "created_at": track["created_at"],
            "storage_backend": storage_backend,
        },
        "cloud_sync": cloud_sync,
    }


@app.put("/api/music/{track_id}/default")
def set_default_music_track(
    track_id: str,
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        selected = db.execute(
            "SELECT * FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        ).fetchone()
        if selected is None:
            raise HTTPException(status_code=404, detail="Music track not found.")
        db.execute(
            "UPDATE music_tracks SET is_default = 0 WHERE user_id = ?",
            (user_id,),
        )
        db.execute(
            "UPDATE music_tracks SET is_default = 1 WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        )
    cloud_tracks = firestore_read_collection(
        ("users", user_id, "music_tracks"), limit=200
    )
    for item in cloud_tracks:
        cloud_id = str(item.get("track_id") or item.get("document_id") or "")
        item["is_default"] = cloud_id == track_id
        if cloud_id:
            firestore_write(
                ("users", user_id, "music_tracks"), cloud_id, item
            )
    selected_doc = dict(selected)
    selected_doc["is_default"] = True
    firestore_write(
        ("users", user_id, "music_tracks"), track_id, selected_doc
    )
    return {"success": True, "default_track_id": track_id}


@app.get("/api/music/{track_id}/audio")
def stream_music_track(
    track_id: str,
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        row = db.execute(
            "SELECT * FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        ).fetchone()
    track = dict(row) if row else firestore_read_document(
        ("users", user_id, "music_tracks"), track_id
    )
    if not track:
        raise HTTPException(status_code=404, detail="Music track not found.")
    try:
        if track.get("storage_backend") == "firebase_storage":
            bucket = configured_music_bucket()
            if bucket is None:
                raise HTTPException(status_code=503, detail="Firebase Storage is not configured.")
            audio_bytes = bucket.blob(track["storage_path"]).download_as_bytes()
        else:
            audio_path = Path(track["storage_path"])
            if not audio_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail="Local audio is unavailable after a server restart. Upload it again or configure Firebase Storage.",
                )
            audio_bytes = audio_path.read_bytes()
    except HTTPException:
        raise
    except Exception:
        logger.exception("Music playback retrieval failed.")
        raise HTTPException(status_code=503, detail="Music file could not be retrieved.")
    return Response(
        content=audio_bytes,
        media_type=track.get("content_type") or "audio/mpeg",
        headers={
            "Content-Disposition": f'inline; filename="{Path(track["file_name"]).name}"',
            "Cache-Control": "private, no-store",
        },
    )


@app.get("/api/personalization/profile")
def personalization_profile(
    user_id: str = Depends(current_user_id),
):
    since = iso(utc_now() - timedelta(days=3))
    with db_connection() as db:
        rows = db.execute(
            """
            SELECT timestamp, heart_rate_bpm, hrv_ms, gsr_raw
            FROM sensor_readings
            WHERE user_id = ? AND source = 'wearable'
              AND signal_quality IN ('good', 'moderate')
              AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (user_id, since),
        ).fetchall()
    day_keys = {str(row["timestamp"])[:10] for row in rows}
    def med(field):
        return median_or_none([
            float(row[field]) for row in rows if row[field] is not None
        ])
    days = len(day_keys)
    return {
        "days_observed": days,
        "reading_count": len(rows),
        "minimum_days": 2,
        "recommended_days": 3,
        "ready_for_preliminary_profile": days >= 2 and len(rows) >= 10,
        "recommended_profile_ready": days >= 3 and len(rows) >= 20,
        "status": (
            "preliminary_profile_available" if days >= 2 and len(rows) >= 10
            else "collecting_personal_data"
        ),
        "personal_reference": {
            "heart_rate_bpm": med("heart_rate_bpm"),
            "hrv_ms": med("hrv_ms"),
            "gsr_raw": med("gsr_raw"),
        },
        "notice": "Two distinct days allow a preliminary personal reference; three or more days are recommended. This is baseline analysis, not a newly trained or clinically validated model.",
    }


@app.post("/api/face/blink-summary", status_code=status.HTTP_201_CREATED)
def save_blink_summary(
    payload: BlinkSummaryIn,
    user_id: str = Depends(current_user_id),
):
    summary_id = str(uuid.uuid4())
    recorded_at = iso()
    rate = round(payload.blink_count * 60.0 / payload.duration_seconds, 2)
    summary = {
        "summary_id": summary_id,
        "user_id": user_id,
        "blink_count": payload.blink_count,
        "duration_seconds": payload.duration_seconds,
        "blinks_per_minute": rate,
        "recorded_at": recorded_at,
    }
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO blink_summaries (
                summary_id, user_id, blink_count,
                duration_seconds, blinks_per_minute, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                summary_id, user_id, payload.blink_count,
                payload.duration_seconds, rate, recorded_at,
            ),
        )
    cloud_sync = firestore_write(
        ("users", user_id, "blink_summaries"), summary_id, summary
    )
    return {"success": True, "summary": summary, "cloud_sync": cloud_sync}


@app.get("/api/face/blink-history")
def blink_history(
    limit: int = Query(default=30, ge=1, le=200),
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        rows = db.execute(
            """
            SELECT summary_id, blink_count, duration_seconds,
                   blinks_per_minute, recorded_at
            FROM blink_summaries
            WHERE user_id = ?
            ORDER BY recorded_at DESC LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()

    # Merge cloud-backed summaries so per-user face history survives an
    # ephemeral hosted filesystem restart, just like sensor and recovery data.
    merged = {
        str(item["summary_id"]): dict(item)
        for item in rows
    }
    cloud_summaries = firestore_read_collection(
        ("users", user_id, "blink_summaries"),
        limit=limit,
        order_by="recorded_at",
    )
    for item in cloud_summaries:
        summary_id = str(item.get("summary_id") or item.get("document_id") or "")
        if summary_id:
            item.setdefault("summary_id", summary_id)
            merged.setdefault(summary_id, item)

    summaries = sorted(
        merged.values(),
        key=lambda item: str(item.get("recorded_at") or ""),
        reverse=True,
    )[:limit]
    return {"total": len(summaries), "summaries": summaries}


@app.get("/api/emergency-contact")
def get_emergency_contact(
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        row = db.execute(
            "SELECT phone, enabled, updated_at, last_alert_at FROM emergency_contacts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row:
        return {
            "phone": row["phone"], "enabled": bool(row["enabled"]),
            "updated_at": row["updated_at"], "last_alert_at": row["last_alert_at"],
            "whatsapp_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM),
        }
    cloud = firestore_read_document(
        ("users", user_id, "emergency_contact"), "primary"
    ) or {}
    if cloud.get("phone"):
        with db_connection() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO emergency_contacts (
                    user_id, phone, enabled, last_alert_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    user_id, cloud["phone"], int(bool(cloud.get("enabled", False))),
                    cloud.get("last_alert_at"), cloud.get("updated_at") or iso(),
                ),
            )
    return {
        "phone": cloud.get("phone", ""),
        "enabled": bool(cloud.get("enabled", False)),
        "updated_at": cloud.get("updated_at"),
        "last_alert_at": cloud.get("last_alert_at"),
        "whatsapp_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM),
    }


@app.put("/api/emergency-contact")
def save_emergency_contact(
    payload: EmergencyContactIn,
    user_id: str = Depends(current_user_id),
):
    updated_at = iso()
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO emergency_contacts (user_id, phone, enabled, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                phone=excluded.phone, enabled=excluded.enabled,
                updated_at=excluded.updated_at
            """,
            (user_id, payload.phone, int(payload.enabled), updated_at),
        )
    cloud_sync = firestore_write(
        ("users", user_id, "emergency_contact"), "primary",
        {"phone": payload.phone, "enabled": payload.enabled, "updated_at": updated_at},
    )
    return {
        "success": True, "phone": payload.phone, "enabled": payload.enabled,
        "whatsapp_configured": bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_WHATSAPP_FROM),
        "cloud_sync": cloud_sync,
    }


@app.post("/api/emergency-contact/test")
def test_emergency_contact(
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        row = db.execute(
            "SELECT phone, enabled FROM emergency_contacts WHERE user_id = ?",
            (user_id,),
        ).fetchone()
    if row is None or not row["enabled"]:
        raise HTTPException(status_code=409, detail="Save and enable an emergency contact first.")
    result = send_whatsapp_message(
        row["phone"],
        "NeuroSense test message: WhatsApp emergency-contact integration is configured. This is a test, not an emergency alert.",
    )
    if result == "not_configured":
        raise HTTPException(
            status_code=503,
            detail="Twilio WhatsApp credentials are not configured on the server.",
        )
    if result != "sent":
        raise HTTPException(status_code=502, detail="WhatsApp test message could not be delivered.")
    return {"success": True, "status": "sent"}


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
        "blink_count_available": True,
        "derived_blink_metrics_can_be_saved_per_user": True,
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
