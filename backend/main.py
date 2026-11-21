
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
import subprocess
import uuid
import re
import urllib.request
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np

from dotenv import load_dotenv

from backend.face_valence import (
    MAX_VIDEO_BYTES as FACE_VALENCE_MAX_VIDEO_BYTES,
    load_face_valence_model,
    predict_from_video_bytes,
)
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
MAX_MUSIC_TRACKS_PER_USER = 10
ALLOWED_MUSIC_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac", ".webm"}

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_BOT_USERNAME = os.getenv("TELEGRAM_BOT_USERNAME", "").strip().lstrip("@")
TELEGRAM_INVITE_TTL_DAYS = 7
TELEGRAM_MAX_CONTACTS = 10
NEUROSENSE_PUBLIC_URL = os.getenv(
    "NEUROSENSE_PUBLIC_URL",
    os.getenv("RENDER_EXTERNAL_URL", "https://neurosense-app.onrender.com")
).strip().rstrip("/")
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

DEFAULT_MONITORING_INTERVAL_SECONDS = 600
MIN_MONITORING_INTERVAL_SECONDS = 5
MAX_MONITORING_INTERVAL_SECONDS = 86400
MAX_SUMMARY_WINDOWS_PER_PASS = 24
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
FACE_VALENCE_MODEL = None
# This AFFEC model was trained on stimulus-locked video-phase windows, not
# arbitrary webcam selfies. Keep live webcam inference disabled until
# numerical parity and new-user validation have been completed.
FACE_VALENCE_WEBCAM_EXPERIMENT_ENABLED = (
    os.getenv("NEUROSENSE_FACE_VALENCE_WEBCAM_EXPERIMENT_ENABLED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)

# Enabling the experiment is not sufficient. Live inference stays unavailable
# until training-to-live feature parity and separate new-user validation are
# completed and explicitly recorded by the maintainer.
FACE_VALENCE_FEATURE_PARITY_VERIFIED = (
    os.getenv("NEUROSENSE_FACE_VALENCE_FEATURE_PARITY_VERIFIED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)
FACE_VALENCE_NEW_USER_VALIDATION_VERIFIED = (
    os.getenv("NEUROSENSE_FACE_VALENCE_NEW_USER_VALIDATION_VERIFIED", "false")
    .strip()
    .lower()
    in {"1", "true", "yes", "on"}
)


def face_valence_live_inference_enabled() -> bool:
    return (
        FACE_VALENCE_WEBCAM_EXPERIMENT_ENABLED
        and FACE_VALENCE_FEATURE_PARITY_VERIFIED
        and FACE_VALENCE_NEW_USER_VALIDATION_VERIFIED
    )


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


def select_latest_sensor_reading(
    local_reading,
    cloud_readings: list[dict],
    source: str | None = None,
) -> dict | None:
    """Select the newest valid timestamped reading from SQLite and Firestore."""
    candidates = []
    if local_reading is not None:
        candidates.append(dict(local_reading))

    candidates.extend(
        dict(item) for item in cloud_readings if isinstance(item, dict)
    )

    newest = None
    newest_timestamp = None
    for item in candidates:
        if source is not None and str(item.get("source") or "") != source:
            continue

        raw_timestamp = item.get("timestamp")
        if not raw_timestamp:
            continue

        try:
            parsed_timestamp = parse_iso(str(raw_timestamp))
        except (TypeError, ValueError):
            continue

        if newest is None or parsed_timestamp > newest_timestamp:
            newest = item
            newest_timestamp = parsed_timestamp

    return newest


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
            ibi_ms_json TEXT,
            stress_score REAL,
            stress_probability REAL,
            stress_status TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_readings_user_time
        ON sensor_readings(user_id, timestamp);

        CREATE INDEX IF NOT EXISTS idx_readings_device_time
        ON sensor_readings(device_id, timestamp);



        CREATE TABLE IF NOT EXISTS reading_feedback (
            feedback_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            reading_id TEXT NOT NULL,
            label TEXT NOT NULL,
            context_note TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, reading_id),
            FOREIGN KEY(reading_id) REFERENCES sensor_readings(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_feedback_user_time
        ON reading_feedback(user_id, created_at);

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
            monitoring_interval_seconds INTEGER NOT NULL DEFAULT 600,
            monitoring_interval_value INTEGER NOT NULL DEFAULT 10,
            monitoring_interval_unit TEXT NOT NULL DEFAULT 'minutes',
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

        CREATE TABLE IF NOT EXISTS telegram_emergency_invites (
            invite_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            contact_name TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            revoked_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_telegram_emergency_invites_user
        ON telegram_emergency_invites(user_id, created_at);

        CREATE TABLE IF NOT EXISTS telegram_emergency_contacts (
            contact_id TEXT PRIMARY KEY,
            invite_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            chat_id TEXT NOT NULL UNIQUE,
            telegram_user_id TEXT,
            username TEXT,
            first_name TEXT,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY(invite_id) REFERENCES telegram_emergency_invites(invite_id)
                ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_telegram_emergency_contacts_user
        ON telegram_emergency_contacts(user_id, active);

        CREATE TABLE IF NOT EXISTS telegram_emergency_alert_state (
            user_id TEXT PRIMARY KEY,
            last_alert_at TEXT NOT NULL
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
            avg_stress_score REAL,
            peak_stress_score REAL,
            peak_stress_at TEXT,
            high_stress_event_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, period_end)
        );

        CREATE TABLE IF NOT EXISTS monitoring_summary_state (
            user_id TEXT PRIMARY KEY,
            last_period_end TEXT NOT NULL
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

        # Retire the legacy single-phone contact table; emergency contacts
        # are now represented by opt-in push invitations and device tokens.
        db.execute("DROP TABLE IF EXISTS emergency_contacts")

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
            "monitoring_interval_seconds": "INTEGER NOT NULL DEFAULT 600",
            "monitoring_interval_value": "INTEGER NOT NULL DEFAULT 10",
            "monitoring_interval_unit": "TEXT NOT NULL DEFAULT 'minutes'",
        }
        for column, definition in migrations.items():
            if column not in existing_columns:
                db.execute(
                    f"ALTER TABLE user_settings ADD COLUMN {column} {definition}"
                )

        reading_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(sensor_readings)").fetchall()
        }
        reading_migrations = {
            "stress_score": "REAL",
            "stress_probability": "REAL",
            "stress_status": "TEXT",
        }
        for column, definition in reading_migrations.items():
            if column not in reading_columns:
                db.execute(
                    f"ALTER TABLE sensor_readings ADD COLUMN {column} {definition}"
                )

        summary_columns = {
            row["name"]
            for row in db.execute("PRAGMA table_info(summaries)").fetchall()
        }
        summary_migrations = {
            "avg_stress_score": "REAL",
            "peak_stress_score": "REAL",
            "peak_stress_at": "TEXT",
            "high_stress_event_count": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, definition in summary_migrations.items():
            if column not in summary_columns:
                db.execute(
                    f"ALTER TABLE summaries ADD COLUMN {column} {definition}"
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

        admin_credential = credentials.Certificate(certificate_source)
        admin_project_id = str(
            getattr(admin_credential, "project_id", "") or ""
        ).strip()
        web_project_id = str(
            FIREBASE_WEB_CONFIG.get("projectId") or ""
        ).strip()

        if (
            admin_project_id
            and web_project_id
            and admin_project_id != web_project_id
        ):
            FIREBASE_APP = None
            FIRESTORE_CLIENT = None
            logger.error(
                "Firebase Web project ID does not match the Admin service-account "
                "project ID; refusing to initialize Firebase."
            )
            return

        try:
            FIREBASE_APP = firebase_admin.get_app()

        except ValueError:
            FIREBASE_APP = firebase_admin.initialize_app(
                admin_credential
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


def initialize_face_valence_model() -> None:
    """Load the separate AFFEC felt-valence model when its artifact is configured."""
    global FACE_VALENCE_MODEL

    model_path = Path(os.getenv(
        "NEUROSENSE_FACE_VALENCE_MODEL_PATH",
        str(BASE_DIR / "ml_model" / "face_valence" / "neurosense_face_valence_rf.joblib")
    ))
    model_url = os.getenv("NEUROSENSE_FACE_VALENCE_MODEL_URL", "").strip() or None
    expected_sha256 = os.getenv(
        "NEUROSENSE_FACE_VALENCE_MODEL_SHA256", ""
    ).strip() or None

    try:
        FACE_VALENCE_MODEL = load_face_valence_model(
            model_path=model_path,
            model_url=model_url,
            expected_sha256=expected_sha256,
        )
        logger.info("AFFEC artifact verified for isolated worker at %s", FACE_VALENCE_MODEL)
    except FileNotFoundError:
        FACE_VALENCE_MODEL = None
        logger.warning(
            "AFFEC felt-valence model is not configured. "
            "Set NEUROSENSE_FACE_VALENCE_MODEL_URL or provide the model file."
        )
    except Exception:
        FACE_VALENCE_MODEL = None
        logger.exception("Could not load the AFFEC felt-valence model.")


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


def firestore_delete_document(
    collection_path: tuple[str, ...],
    document_id: str,
) -> str:
    """Delete a document from an authenticated user's Firestore subcollection."""
    if FIRESTORE_CLIENT is None:
        return "not_configured"
    if not collection_path or len(collection_path) % 2 == 0:
        logger.error("Invalid Firestore collection path: %s", collection_path)
        return "failed"
    try:
        reference = FIRESTORE_CLIENT.collection(collection_path[0])
        for index, segment in enumerate(collection_path[1:], start=1):
            if index % 2 == 1:
                reference = reference.document(segment)
            else:
                reference = reference.collection(segment)
        reference.document(document_id).delete()
        return "synced"
    except Exception:
        logger.exception("Firestore document delete failed for %s", document_id)
        return "failed"


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
# 7A. TELEGRAM ALERTS AND OPTIONAL CLOUD MUSIC
# ============================================================

def telegram_configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME)


def telegram_webhook_secret() -> str:
    """Derive a stable webhook secret without storing another credential."""
    return hashlib.sha256(
        ("neurosense-telegram-webhook:" + TELEGRAM_BOT_TOKEN).encode("utf-8")
    ).hexdigest()


def telegram_api_call(method: str, payload: dict) -> dict:
    """Call Telegram Bot API without logging or exposing the bot token."""
    if not TELEGRAM_BOT_TOKEN:
        return {"ok": False, "description": "Telegram bot token is not configured."}

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=12) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not result.get("ok"):
            logger.warning(
                "Telegram Bot API %s returned an error: %s",
                method,
                result.get("description", "unknown error"),
            )
        return result
    except Exception as error:
        logger.error("Telegram Bot API %s request failed: %s", method, type(error).__name__)
        return {"ok": False, "description": "Telegram API request failed."}


def configure_telegram_webhook() -> bool:
    """Register Telegram's HTTPS webhook on Render at application startup."""
    if not telegram_configured():
        logger.info("Telegram alerts are not configured; skipping webhook setup.")
        return False

    result = telegram_api_call(
        "setWebhook",
        {
            "url": NEUROSENSE_PUBLIC_URL + "/api/telegram/webhook",
            "secret_token": telegram_webhook_secret(),
            "allowed_updates": ["message"],
            "drop_pending_updates": False,
            "max_connections": 5,
        },
    )
    if result.get("ok"):
        logger.info("Telegram emergency-alert webhook configured.")
        return True
    logger.error("Telegram webhook setup failed.")
    return False


def send_telegram_message(chat_id: str | int, text: str) -> bool:
    result = telegram_api_call(
        "sendMessage",
        {"chat_id": str(chat_id), "text": text[:4000], "disable_web_page_preview": True},
    )
    return bool(result.get("ok"))


def telegram_chat_document_id(chat_id: str | int) -> str:
    """Stable Firestore-safe ID for a Telegram chat."""
    return hashlib.sha256(str(chat_id).encode("utf-8")).hexdigest()


def telegram_user_collection(user_id: str, collection_name: str):
    if FIRESTORE_CLIENT is None:
        raise RuntimeError("Firestore is not configured.")
    return (
        FIRESTORE_CLIENT.collection("users")
        .document(str(user_id))
        .collection(collection_name)
    )


def telegram_global_document(collection_name: str, document_id: str):
    if FIRESTORE_CLIENT is None:
        raise RuntimeError("Firestore is not configured.")
    return FIRESTORE_CLIENT.collection(collection_name).document(document_id)



def telegram_firestore_documents(collection_ref) -> list[dict]:
    """Return a Firestore collection with document IDs included."""
    return [
        {**(snapshot.to_dict() or {}), "document_id": snapshot.id}
        for snapshot in collection_ref.stream()
    ]


def sync_local_telegram_state_to_firestore(user_id: str) -> None:
    """
    One-time-safe migration of any still-visible SQLite Telegram state.
    Firestore remains authoritative; existing Firestore documents are never
    overwritten by a local cache copy.
    """
    if FIRESTORE_CLIENT is None:
        return

    with db_connection() as db:
        invite_rows = db.execute(
            """
            SELECT invite_id, user_id, contact_name, token_hash,
                   created_at, expires_at, revoked_at
            FROM telegram_emergency_invites
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
        contact_rows = db.execute(
            """
            SELECT contact_id, invite_id, user_id, chat_id, telegram_user_id,
                   username, first_name, created_at, last_seen_at, active
            FROM telegram_emergency_contacts
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

    invite_collection = telegram_user_collection(
        user_id, "telegram_emergency_invites"
    )
    contact_collection = telegram_user_collection(
        user_id, "telegram_emergency_contacts"
    )

    for row in invite_rows:
        invite = dict(row)
        invite_ref = invite_collection.document(str(invite["invite_id"]))
        if not invite_ref.get().exists:
            invite_ref.set(invite)
        if not invite.get("revoked_at"):
            token_ref = telegram_global_document(
                "telegram_invite_tokens", str(invite["token_hash"])
            )
            if not token_ref.get().exists:
                token_ref.set({
                    "user_id": user_id,
                    "invite_id": str(invite["invite_id"]),
                })

    for row in contact_rows:
        contact = dict(row)
        chat_id = str(contact["chat_id"])
        stable_id = telegram_chat_document_id(chat_id)
        contact_ref = contact_collection.document(stable_id)
        if not contact_ref.get().exists:
            contact["active"] = bool(contact.get("active"))
            contact["chat_id"] = chat_id
            contact_ref.set(contact)
        chat_ref = telegram_global_document(
            "telegram_chat_links", stable_id
        )
        if not chat_ref.get().exists:
            chat_ref.set({
                "user_id": user_id,
                "invite_id": str(contact["invite_id"]),
                "contact_id": stable_id,
                "active": bool(contact.get("active")),
            })


def telegram_firestore_active_chat_ids(user_id: str) -> list[str]:
    if FIRESTORE_CLIENT is None:
        return []
    contacts = telegram_firestore_documents(
        telegram_user_collection(user_id, "telegram_emergency_contacts")
    )
    return [
        str(item["chat_id"])
        for item in contacts
        if item.get("active") and item.get("chat_id") is not None
    ]


def send_user_telegram_alert(user_id: str, title: str, body: str) -> tuple[str, int]:
    if not telegram_configured():
        return "not_configured", 0

    if FIRESTORE_CLIENT is not None:
        try:
            chat_ids = telegram_firestore_active_chat_ids(user_id)
        except Exception:
            logger.exception("Could not load Telegram contacts from Firestore.")
            return "storage_failed", 0
    else:
        with db_connection() as db:
            rows = db.execute(
                """
                SELECT chat_id
                FROM telegram_emergency_contacts
                WHERE user_id = ? AND active = 1
                ORDER BY created_at
                """,
                (user_id,),
            ).fetchall()
        chat_ids = [str(row["chat_id"]) for row in rows]

    if not chat_ids:
        return "no_registered_contacts", 0

    message = f"{title}\n\n{body}"
    sent_count = sum(
        1 for chat_id in chat_ids
        if send_telegram_message(chat_id, message)
    )
    return ("sent", sent_count) if sent_count else ("failed", 0)

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



def maybe_send_emergency_telegram_alert(user_id: str, probability: float) -> str:
    """Send a Telegram check-in using cloud-backed event history and cooldown."""
    if probability < HIGH_STRESS_THRESHOLD:
        return "not_triggered"
    if not telegram_configured():
        return "not_configured"

    now = utc_now()
    now_text = iso(now)
    cutoff = now - timedelta(minutes=30)

    # Local SQLite is a development cache. In production, recover the rolling
    # event window and alert reservation from Firestore so a Render restart
    # cannot erase the three-reading threshold or cause a duplicate alert.
    if FIRESTORE_CLIENT is not None:
        try:
            local_events = []
            with db_connection() as db:
                local_events = [
                    dict(row) for row in db.execute(
                        """
                        SELECT event_id, reading_id, stress_probability, occurred_at
                        FROM high_stress_events
                        WHERE user_id = ? AND occurred_at >= ?
                        """,
                        (user_id, iso(cutoff)),
                    ).fetchall()
                ]

            event_ref = telegram_user_collection(user_id, "high_stress_events")
            event_query = event_ref.order_by(
                "occurred_at", direction=firestore.Query.DESCENDING
            ).limit(500)
            cloud_events = [
                {**(snapshot.to_dict() or {}), "document_id": snapshot.id}
                for snapshot in event_query.stream()
            ]

            state_ref = telegram_user_collection(
                user_id, "telegram_emergency_alert_state"
            ).document("current")
            state_snapshot = state_ref.get()
            cloud_state = (
                state_snapshot.to_dict() or {}
                if state_snapshot.exists else {}
            )
            active_chat_ids = telegram_firestore_active_chat_ids(user_id)
        except Exception:
            logger.exception("Could not recover Telegram alert history or cooldown from Firestore.")
            return "storage_failed"

        event_by_reading: dict[str, dict] = {}
        for event in [*local_events, *cloud_events]:
            reading_id = str(
                event.get("reading_id") or event.get("event_id")
                or event.get("document_id") or ""
            )
            raw_occurred_at = event.get("occurred_at")
            if not reading_id or not raw_occurred_at:
                continue
            try:
                occurred_at = parse_iso(str(raw_occurred_at))
            except (TypeError, ValueError):
                continue
            if cutoff <= occurred_at <= now:
                event_by_reading[reading_id] = event

        if len(event_by_reading) < 3:
            return "awaiting_repeated_readings"

        last_attempt = cloud_state.get("last_attempt_at") or cloud_state.get("last_alert_at")
        if last_attempt:
            try:
                if now - parse_iso(str(last_attempt)) < timedelta(hours=6):
                    return "cooldown"
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid Telegram alert cooldown timestamp for user %s.", user_id)

        if not active_chat_ids:
            return "no_registered_contacts"

        # Reserve the cooldown before contacting Telegram. If the process
        # stops after delivery but before the response is persisted, a restart
        # must not immediately send the same check-in again.
        reserved_state = {
            **cloud_state,
            "user_id": user_id,
            "last_attempt_at": now_text,
            "last_attempt_id": uuid.uuid4().hex,
            "last_attempt_status": "pending",
        }
        try:
            state_ref.set(reserved_state)
        except Exception:
            logger.exception("Could not reserve durable Telegram alert cooldown.")
            return "storage_failed"

        delivery, sent_count = send_user_telegram_alert(
            user_id,
            "NeuroSense wellness check-in",
            "Repeated elevated stress-related readings were recorded. Please check in with the NeuroSense user. "
            "This is an experimental wellness alert, not a medical diagnosis or emergency service.",
        )

        final_state = {
            **reserved_state,
            "last_attempt_status": delivery,
            "last_delivery_count": int(sent_count),
        }
        if delivery == "sent" and sent_count:
            final_state["last_alert_at"] = now_text

        try:
            state_ref.set(final_state)
        except Exception:
            # The pre-send reservation remains durable, preventing duplicates.
            logger.exception("Telegram alert sent/attempted but final cooldown metadata could not be saved.")

        if delivery == "sent" and sent_count:
            try:
                with db_connection() as db:
                    db.execute(
                        """
                        INSERT INTO telegram_emergency_alert_state (user_id, last_alert_at)
                        VALUES (?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET last_alert_at = excluded.last_alert_at
                        """,
                        (user_id, now_text),
                    )
            except Exception:
                logger.exception("Could not update the local Telegram alert cooldown cache.")

        return delivery

    # Local development fallback: Firestore is not configured.
    with db_connection() as db:
        recent = db.execute(
            """
            SELECT COUNT(DISTINCT reading_id) AS total
            FROM high_stress_events
            WHERE user_id = ? AND occurred_at >= ?
            """,
            (user_id, iso(cutoff)),
        ).fetchone()["total"]
        if int(recent) < 3:
            return "awaiting_repeated_readings"

        alert_state = db.execute(
            "SELECT last_alert_at FROM telegram_emergency_alert_state WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if alert_state and now - parse_iso(alert_state["last_alert_at"]) < timedelta(hours=6):
            return "cooldown"

    delivery, sent_count = send_user_telegram_alert(
        user_id,
        "NeuroSense wellness check-in",
        "Repeated elevated stress-related readings were recorded. Please check in with the NeuroSense user. "
        "This is an experimental wellness alert, not a medical diagnosis or emergency service.",
    )
    if delivery == "sent" and sent_count:
        with db_connection() as db:
            db.execute(
                """
                INSERT INTO telegram_emergency_alert_state (user_id, last_alert_at)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET last_alert_at = excluded.last_alert_at
                """,
                (user_id, now_text),
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



class PersonalFeedbackIn(StrictModel):
    reading_id: str = Field(min_length=1, max_length=80)
    label: Literal["calm", "stressed", "active", "uncertain"]
    context_note: str | None = Field(default=None, max_length=300)


class MonitoringIntervalIn(StrictModel):
    interval_value: int = Field(ge=1, le=86400)
    interval_unit: Literal["seconds", "minutes", "hours"]


class TelegramEmergencyInviteIn(StrictModel):
    contact_name: str = Field(min_length=2, max_length=60)


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

    # WESAD training uses EDA in microsiemens (uS), while current ESP32
    # firmware sends an uncalibrated 0-4095 ADC count. Do not feed those counts
    # into a model feature trained in a different physical unit. Keep the GSR
    # fields missing so the fitted imputer uses its training median until a
    # hardware-specific ADC-to-uS calibration path is implemented and verified.
    return {
        "heart_rate_bpm": number_or_nan(reading.get("heart_rate_bpm")),
        "hrv_ms": number_or_nan(reading.get("hrv_ms")),
        "gsr_raw": float("nan"),
        "skin_temperature_c": number_or_nan(reading.get("skin_temperature_c")),
        "heart_rate_delta": delta("heart_rate_bpm"),
        "hrv_delta": delta("hrv_ms"),
        "gsr_delta": float("nan"),
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
                "model_validated": False,
                "model_input_warnings": [
                    "GSR was excluded from this prediction because the ESP32 sends raw ADC counts while the training data uses EDA in microsiemens. Hardware calibration is required before GSR can be used."
                ]
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

                # NeuroSense uses Telegram as its only emergency-contact channel.
                # Do not dispatch the legacy Firebase Web Push path.
                result["telegram_alert"] = maybe_send_emergency_telegram_alert(
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

def stored_or_predicted_stress(user_id: str, reading: dict) -> dict:
    """Reuse the persisted prediction for a reading instead of re-triggering events."""
    if reading.get("stress_status"):
        return {
            "status": reading.get("stress_status"),
            "stress_score": reading.get("stress_score"),
            "stress_probability": reading.get("stress_probability"),
            "model_validated": False,
            "model_input_warnings": [
                "GSR was excluded from this prediction because the ESP32 sends raw ADC counts while the training data uses EDA in microsiemens. Hardware calibration is required before GSR can be used."
            ],
        }
    return predict_stress(user_id, reading)



def maybe_queue_personal_haptic_nudge(
    user_id: str,
    device_id: str,
    reading: dict,
) -> dict:
    """Queue a short opt-in haptic nudge only after sustained personal HR/HRV deviation.

    This is a conservative wellness cue, not stress detection. It requires a
    calibrated personal reference, three consecutive good-quality readings
    spanning at least 60 seconds, explicit haptic opt-in, and a 15-minute
    cooldown. GSR is intentionally excluded until sensor polarity/calibration
    is established for each hardware build.
    """
    if reading.get("source") != "wearable" or reading.get("signal_quality") != "good":
        return {"queued": False, "reason": "signal_quality"}
    settings = get_settings(user_id)
    if not settings.get("haptic_enabled"):
        return {"queued": False, "reason": "haptic_opt_out"}

    baseline = baseline_for_user(user_id)
    if not baseline or not baseline.get("heart_rate_bpm") or not baseline.get("hrv_ms"):
        return {"queued": False, "reason": "personal_baseline_required"}

    base_hr = float(baseline["heart_rate_bpm"])
    base_hrv = float(baseline["hrv_ms"])
    if base_hr <= 0 or base_hrv <= 0:
        return {"queued": False, "reason": "invalid_baseline"}

    with db_connection() as db:
        local_rows = db.execute(
            """
            SELECT id, timestamp, heart_rate_bpm, hrv_ms, signal_quality
            FROM sensor_readings
            WHERE user_id = ? AND source = 'wearable'
            ORDER BY timestamp DESC
            LIMIT 3
            """,
            (user_id,),
        ).fetchall()
        last_command = db.execute(
            """
            SELECT created_at FROM haptic_commands
            WHERE user_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        active_label = db.execute(
            """
            SELECT rf.label, rf.created_at
            FROM reading_feedback rf
            WHERE rf.user_id = ?
            ORDER BY rf.created_at DESC LIMIT 1
            """,
            (user_id,),
        ).fetchone()

    rows = [dict(row) for row in local_rows]
    if len(rows) < 3:
        # After an ephemeral Render reset, the current upload may be the only
        # local row. Rehydrate prior wearable samples before evaluating the
        # consecutive-reading and 60-second persistence gates.
        cloud_rows = firestore_read_collection(
            ("users", user_id, "sensor_readings"),
            limit=100,
            order_by="timestamp",
        )
        readings_by_id = {
            str(item.get("id") or ""): item
            for item in rows
            if item.get("id")
        }
        for item in cloud_rows:
            reading_id = str(item.get("id") or item.get("document_id") or "")
            if (
                not reading_id
                or str(item.get("source") or "") != "wearable"
                or reading_id in readings_by_id
            ):
                continue
            readings_by_id[reading_id] = dict(item)
        rows = sorted(
            readings_by_id.values(),
            key=lambda item: str(item.get("timestamp") or ""),
            reverse=True,
        )[:3]

    if last_command is None:
        cloud_commands = firestore_read_collection(
            ("users", user_id, "haptic_commands"),
            limit=1,
            order_by="created_at",
        )
        if cloud_commands:
            last_command = cloud_commands[0]

    if active_label is None:
        cloud_feedback = firestore_read_collection(
            ("users", user_id, "reading_feedback"),
            limit=1,
            order_by="created_at",
        )
        if cloud_feedback:
            active_label = cloud_feedback[0]

    if len(rows) < 3 or any(row["signal_quality"] != "good" for row in rows):
        return {"queued": False, "reason": "not_enough_consecutive_good_readings"}

    ordered = sorted(rows, key=lambda row: str(row["timestamp"] or ""))
    try:
        span = (parse_iso(ordered[-1]["timestamp"]) - parse_iso(ordered[0]["timestamp"])).total_seconds()
    except Exception:
        return {"queued": False, "reason": "invalid_reading_timestamps"}
    if span < 60:
        return {"queued": False, "reason": "change_not_sustained"}

    for row in ordered:
        hr, hrv = row["heart_rate_bpm"], row["hrv_ms"]
        if hr is None or hrv is None:
            return {"queued": False, "reason": "missing_hr_or_hrv"}
        if float(hr) < base_hr * 1.15 or float(hrv) > base_hrv * 0.80:
            return {"queued": False, "reason": "personal_deviation_not_met"}

    if active_label and active_label["label"] == "active":
        try:
            if (utc_now() - parse_iso(active_label["created_at"])).total_seconds() < 1800:
                return {"queued": False, "reason": "recent_activity_label"}
        except Exception:
            pass

    if last_command:
        try:
            if (utc_now() - parse_iso(last_command["created_at"])).total_seconds() < 900:
                return {"queued": False, "reason": "cooldown"}
        except Exception:
            return {"queued": False, "reason": "cooldown_timestamp_invalid"}

    with db_connection() as db:
        device = db.execute(
            """
            SELECT device_id FROM devices
            WHERE device_id = ? AND user_id = ? AND active = 1
            """,
            (device_id, user_id),
        ).fetchone()
    if device is None:
        return {"queued": False, "reason": "no_active_paired_device"}

    command_id = str(uuid.uuid4())
    created_at = iso()
    command = {
        "command_id": command_id,
        "user_id": user_id,
        "device_id": device_id,
        "duration_ms": 300,
        "command_status": "pending",
        "created_at": created_at,
        "delivered_at": None,
        "acknowledged_at": None,
        "trigger": "sustained_personal_hr_hrv_deviation",
    }
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO haptic_commands (
                command_id, user_id, device_id, duration_ms,
                command_status, created_at, delivered_at, acknowledged_at
            )
            VALUES (?, ?, ?, 300, 'pending', ?, NULL, NULL)
            """,
            (command_id, user_id, device_id, created_at),
        )
    firestore_write(("users", user_id, "haptic_commands"), command_id, command)
    return {"queued": True, "command_id": command_id, "duration_ms": 300,
            "reason": "sustained_personal_hr_hrv_deviation"}


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

    stored["stress_score"] = prediction.get("stress_score")
    stored["stress_probability"] = prediction.get("stress_probability")
    stored["stress_status"] = prediction.get("status")

    with db_connection() as db:
        db.execute(
            """
            UPDATE sensor_readings
            SET stress_score = ?, stress_probability = ?, stress_status = ?
            WHERE id = ? AND user_id = ?
            """,
            (
                stored["stress_score"],
                stored["stress_probability"],
                stored["stress_status"],
                reading_id,
                user_id,
            ),
        )

    cloud_status = firestore_write(
        ("users", user_id, "sensor_readings"),
        reading_id,
        stored
    )

    haptic_nudge = (
        maybe_queue_personal_haptic_nudge(user_id, device_id, stored)
        if payload["source"] == "wearable"
        else {"queued": False, "reason": "not_wearable"}
    )

    return {
        "reading": stored,
        "cloud_sync": cloud_status,
        "stress": prediction,
        "haptic_nudge": haptic_nudge,
    }


# ============================================================
# 14. USER-INTERVAL MONITORING SUMMARY GENERATION
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
    """Create per-user monitoring summaries from local and cloud-backed readings."""
    created = 0
    now = utc_now()

    with db_connection() as db:
        # A user is discovered once their device sends a post-restart reading.
        # At that point, old history can be restored from that user's cloud path.
        users = db.execute(
            """
            SELECT DISTINCT user_id
            FROM sensor_readings
            WHERE source = 'wearable'
            """
        ).fetchall()

        for user_row in users:
            user_id = str(user_row["user_id"])

            settings = db.execute(
                """
                SELECT monitoring_interval_seconds
                FROM user_settings
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

            if settings and settings["monitoring_interval_seconds"]:
                interval_seconds = int(settings["monitoring_interval_seconds"])
            else:
                cloud_preferences = firestore_read_document(
                    ("users", user_id, "settings"), "preferences"
                ) or {}
                interval_seconds = int(
                    cloud_preferences.get(
                        "monitoring_interval_seconds",
                        DEFAULT_MONITORING_INTERVAL_SECONDS,
                    )
                )
            interval_seconds = max(
                MIN_MONITORING_INTERVAL_SECONDS,
                min(MAX_MONITORING_INTERVAL_SECONDS, interval_seconds),
            )
            interval = timedelta(seconds=interval_seconds)

            state = db.execute(
                """
                SELECT last_period_end
                FROM monitoring_summary_state
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()
            local_last_summary = db.execute(
                """
                SELECT MAX(period_end) AS end_time
                FROM summaries
                WHERE user_id = ?
                """,
                (user_id,),
            ).fetchone()

            checkpoints = []
            if state and state["last_period_end"]:
                try:
                    checkpoints.append(parse_iso(state["last_period_end"]))
                except (TypeError, ValueError):
                    pass
            if local_last_summary and local_last_summary["end_time"]:
                try:
                    checkpoints.append(parse_iso(local_last_summary["end_time"]))
                except (TypeError, ValueError):
                    pass

            existing_period_ends = set()
            # Cloud checkpoint reconciliation is needed only after a local DB
            # reset (or before the first generated summary). Once restored, the
            # local checkpoint prevents repeatedly downloading history each second.
            if state is None:
                cloud_state = firestore_read_document(
                    ("users", user_id, "monitoring_summary_state"), "current"
                ) or {}
                cloud_checkpoint = cloud_state.get("last_period_end")
                if cloud_checkpoint:
                    try:
                        checkpoints.append(parse_iso(str(cloud_checkpoint)))
                    except (TypeError, ValueError):
                        pass

                cloud_summaries = firestore_read_collection(
                    ("users", user_id, "summaries"),
                    limit=2000,
                    order_by="period_end",
                )
                for item in cloud_summaries:
                    raw_period_end = str(item.get("period_end") or "")
                    if not raw_period_end:
                        continue
                    try:
                        normalized_period_end = iso(parse_iso(raw_period_end))
                        existing_period_ends.add(normalized_period_end)
                        checkpoints.append(parse_iso(raw_period_end))
                    except (TypeError, ValueError):
                        continue

            period_start = max(checkpoints) if checkpoints else None
            if period_start is not None and period_start > now:
                period_start = now

            # Most passes run every second. Do not fetch thousands of cloud
            # documents until a complete summary interval is actually due.
            if period_start is not None and now - period_start < interval:
                next_checkpoint = iso(period_start)
                db.execute(
                    """
                    INSERT INTO monitoring_summary_state (user_id, last_period_end)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_period_end = excluded.last_period_end
                    """,
                    (user_id, next_checkpoint),
                )
                if state is None:
                    firestore_write(
                        ("users", user_id, "monitoring_summary_state"),
                        "current",
                        {"user_id": user_id, "last_period_end": next_checkpoint},
                    )
                continue

            local_sensor_rows = db.execute(
                """
                SELECT *
                FROM sensor_readings
                WHERE user_id = ? AND source = 'wearable'
                ORDER BY timestamp ASC
                """,
                (user_id,),
            ).fetchall()
            cloud_sensor_rows = firestore_read_collection(
                ("users", user_id, "sensor_readings"),
                limit=5000,
                order_by="timestamp",
            )

            readings_by_id = {}
            for row in local_sensor_rows:
                item = dict(row)
                reading_id = str(item.get("id") or "")
                if reading_id:
                    readings_by_id[reading_id] = item
            for item in cloud_sensor_rows:
                if str(item.get("source") or "") != "wearable":
                    continue
                reading_id = str(item.get("id") or item.get("document_id") or "")
                if reading_id:
                    readings_by_id.setdefault(reading_id, dict(item))

            readings = []
            for item in readings_by_id.values():
                timestamp = str(item.get("timestamp") or "")
                if not timestamp:
                    continue
                try:
                    recorded_at = parse_iso(timestamp)
                except (TypeError, ValueError):
                    continue
                item["timestamp"] = timestamp
                readings.append((recorded_at, item))
            readings.sort(key=lambda pair: pair[0])
            if not readings:
                continue

            if period_start is None:
                period_start = readings[0][0]

            if now - period_start < interval:
                next_checkpoint = iso(period_start)
                db.execute(
                    """
                    INSERT INTO monitoring_summary_state (user_id, last_period_end)
                    VALUES (?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        last_period_end = excluded.last_period_end
                    """,
                    (user_id, next_checkpoint),
                )
                firestore_write(
                    ("users", user_id, "monitoring_summary_state"),
                    "current",
                    {"user_id": user_id, "last_period_end": next_checkpoint},
                )
                continue

            local_events = db.execute(
                """
                SELECT event_id, reading_id, stress_probability, occurred_at
                FROM high_stress_events
                WHERE user_id = ?
                ORDER BY occurred_at ASC
                """,
                (user_id,),
            ).fetchall()
            cloud_events = firestore_read_collection(
                ("users", user_id, "high_stress_events"),
                limit=5000,
                order_by="occurred_at",
            )
            events_by_id = {}
            for row in local_events:
                item = dict(row)
                events_by_id[str(item["event_id"])] = item
            for item in cloud_events:
                event_id = str(item.get("event_id") or item.get("document_id") or "")
                if event_id:
                    events_by_id.setdefault(event_id, dict(item))
            events = []
            for item in events_by_id.values():
                occurred_at = str(item.get("occurred_at") or "")
                if not occurred_at:
                    continue
                try:
                    events.append((parse_iso(occurred_at), item))
                except (TypeError, ValueError):
                    continue

            windows = 0
            while now - period_start >= interval and windows < MAX_SUMMARY_WINDOWS_PER_PASS:
                period_end = period_start + interval
                start_iso = iso(period_start)
                end_iso = iso(period_end)

                window_rows = [
                    item for recorded_at, item in readings
                    if period_start <= recorded_at < period_end
                ]

                if window_rows and end_iso not in existing_period_ends:
                    reliable = [
                        row for row in window_rows
                        if row.get("signal_quality") in ("good", "moderate")
                    ]
                    stress_rows = [
                        row for row in reliable
                        if row.get("stress_score") is not None
                    ]
                    peak_row = (
                        max(stress_rows, key=lambda row: float(row["stress_score"]))
                        if stress_rows else None
                    )
                    event_count = sum(
                        1 for occurred_at, _event in events
                        if period_start <= occurred_at < period_end
                    )

                    # Deterministic IDs make cloud writes idempotent if the
                    # process exits after Firestore commits but before SQLite.
                    summary_id = str(uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"neurosense:{user_id}:summary:{end_iso}",
                    ))
                    summary = {
                        "summary_id": summary_id,
                        "user_id": user_id,
                        "period_start": start_iso,
                        "period_end": end_iso,
                        "interval_seconds": interval_seconds,
                        "reading_count": len(window_rows),
                        "reliable_reading_count": len(reliable),
                        "avg_heart_rate": average(reliable, "heart_rate_bpm"),
                        "avg_hrv": average(reliable, "hrv_ms"),
                        "avg_gsr": average(reliable, "gsr_raw"),
                        "avg_skin_temperature": average(reliable, "skin_temperature_c"),
                        "avg_stress_score": average(stress_rows, "stress_score"),
                        "peak_stress_score": (
                            round(float(peak_row["stress_score"]), 1)
                            if peak_row else None
                        ),
                        "peak_stress_at": peak_row["timestamp"] if peak_row else None,
                        "high_stress_event_count": int(event_count),
                        "created_at": iso(),
                    }

                    # Persist the cloud summary before advancing the checkpoint.
                    # If Firestore is temporarily unavailable, leave this period
                    # uncommitted so the next scheduler pass can retry it. The
                    # deterministic summary ID makes a retry idempotent.
                    cloud_sync = firestore_write(
                        ("users", user_id, "summaries"),
                        summary_id,
                        summary,
                    )
                    if cloud_sync not in {"synced", "not_configured"}:
                        logger.warning(
                            "Could not persist monitoring summary for user %s at %s; "
                            "leaving the checkpoint unchanged for retry.",
                            user_id,
                            end_iso,
                        )
                        break

                    db.execute(
                        """
                        INSERT OR IGNORE INTO summaries (
                            summary_id, user_id, period_start, period_end,
                            reading_count, reliable_reading_count,
                            avg_heart_rate, avg_hrv, avg_gsr,
                            avg_skin_temperature, avg_stress_score,
                            peak_stress_score, peak_stress_at,
                            high_stress_event_count, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            summary["summary_id"], user_id,
                            summary["period_start"], summary["period_end"],
                            summary["reading_count"], summary["reliable_reading_count"],
                            summary["avg_heart_rate"], summary["avg_hrv"],
                            summary["avg_gsr"], summary["avg_skin_temperature"],
                            summary["avg_stress_score"], summary["peak_stress_score"],
                            summary["peak_stress_at"], summary["high_stress_event_count"],
                            summary["created_at"],
                        ),
                    )
                    created += 1
                    existing_period_ends.add(end_iso)

                period_start = period_end
                windows += 1

            next_checkpoint = iso(period_start)
            db.execute(
                """
                INSERT INTO monitoring_summary_state (user_id, last_period_end)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    last_period_end = excluded.last_period_end
                """,
                (user_id, next_checkpoint),
            )
            firestore_write(
                ("users", user_id, "monitoring_summary_state"),
                "current",
                {"user_id": user_id, "last_period_end": next_checkpoint},
            )

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

        await asyncio.sleep(1)


# ============================================================
# 15. FASTAPI STARTUP AND CORS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    initialize_database()
    initialize_firebase()
    initialize_ml_model()
    initialize_face_valence_model()
    configure_telegram_webhook()

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
        "firebase_sign_in_ready": (
            firebase_web_configured() and FIREBASE_APP is not None
        ),
        # These are configuration-presence indicators, not external service probes.
        "firebase_storage_configured": bool(
            FIREBASE_APP is not None and FIREBASE_STORAGE_BUCKET
        ),
        "telegram_alerts_configured": telegram_configured(),
        "ml_model_loaded": STRESS_MODEL is not None,
        "face_valence_model_loaded": FACE_VALENCE_MODEL is not None,
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
        # Safe capability flag only; never expose the bucket name or credentials.
        "firebase_storage_configured": configured_music_bucket() is not None,
        "telegram_alerts_configured": telegram_configured(),
        "telegram_bot_username": TELEGRAM_BOT_USERNAME or None,
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

        "face_scan_analysis": "browser_local_landmarks; AFFEC_webcam_inference_validation_pending",
        "face_valence_model_loaded": FACE_VALENCE_MODEL is not None,
        "face_valence_webcam_experiment_enabled": face_valence_live_inference_enabled(),
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

    local_data = row_dict(row)

    if local_data:
        local_data["ibi_ms"] = (
            json.loads(local_data.pop("ibi_ms_json"))
            if local_data.get("ibi_ms_json")
            else None
        )

    cloud_rows = firestore_read_collection(
        ("users", user_id, "sensor_readings"),
        limit=100,
        order_by="timestamp",
    )
    data = select_latest_sensor_reading(
        local_data,
        cloud_rows,
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

    # Always check the cloud copy: Render can retain an older local row
    # while Firestore already contains a newer wearable reading.
    cloud_rows = firestore_read_collection(
        ("users", user_id, "sensor_readings"),
        limit=100,
        order_by="timestamp",
    )
    reading = select_latest_sensor_reading(
        row,
        cloud_rows,
        source="wearable",
    )
    if reading is None:
        return {
            "status": "waiting_for_data",
            "stress_score": None
        }

    prediction = stored_or_predicted_stress(user_id, reading)
    return {
        **prediction,
        "reading_id": reading.get("id"),
        "reading_timestamp": reading.get("timestamp"),
    }


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

    cloud_readings = firestore_read_collection(
        ("users", user_id, "sensor_readings"),
        limit=1,
        order_by="timestamp",
    )
    reading = select_latest_sensor_reading(
        latest,
        cloud_readings,
    )

    if (
        reading
        and reading.get("source") == "wearable"
    ):

        prediction = stored_or_predicted_stress(
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
            "monitoring_interval_seconds": int(values.get("monitoring_interval_seconds", DEFAULT_MONITORING_INTERVAL_SECONDS)),
            "monitoring_interval_value": int(values.get("monitoring_interval_value", 10)),
            "monitoring_interval_unit": values.get("monitoring_interval_unit", "minutes"),
            "updated_at": values.get("updated_at"),
        }

    return {
        "haptic_enabled": bool(row["haptic_enabled"]),
        "music_mode": row["music_mode"],
        "warning_sound_enabled": bool(row["warning_sound_enabled"]),
        "notifications_enabled": bool(row["notifications_enabled"]),
        "monitoring_interval_seconds": int(row["monitoring_interval_seconds"]),
        "monitoring_interval_value": int(row["monitoring_interval_value"]),
        "monitoring_interval_unit": row["monitoring_interval_unit"],
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
            "monitoring_interval_seconds": current_preferences.get("monitoring_interval_seconds", DEFAULT_MONITORING_INTERVAL_SECONDS),
            "monitoring_interval_value": current_preferences.get("monitoring_interval_value", 10),
            "monitoring_interval_unit": current_preferences.get("monitoring_interval_unit", "minutes"),
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

    # When Render's ephemeral SQLite database is recreated, both recovery
    # sessions and sensor readings may exist only in Firestore. Rebuild a
    # missing before/after comparison from this user's cloud wearable data.
    if any(session.get("comparison") is None for session in sessions):
        cloud_sensor_rows = firestore_read_collection(
            ("users", user_id, "sensor_readings"),
            limit=500,
            order_by="timestamp",
        )
        cloud_wearable_readings = []
        for reading in cloud_sensor_rows:
            if str(reading.get("source") or "") != "wearable":
                continue
            timestamp = str(reading.get("timestamp") or "")
            if not timestamp:
                continue
            try:
                recorded_at = parse_iso(timestamp)
            except (TypeError, ValueError):
                continue
            cloud_wearable_readings.append((recorded_at, {
                "timestamp": timestamp,
                "heart_rate_bpm": reading.get("heart_rate_bpm"),
                "hrv_ms": reading.get("hrv_ms"),
                "gsr_raw": reading.get("gsr_raw"),
                "skin_temperature_c": reading.get("skin_temperature_c"),
            }))

        for session in sessions:
            if session.get("comparison") is not None:
                continue
            try:
                started_at = parse_iso(str(session.get("started_at") or ""))
                completed_at = parse_iso(str(session.get("completed_at") or ""))
            except (TypeError, ValueError):
                continue

            before_candidates = [
                (recorded_at, reading)
                for recorded_at, reading in cloud_wearable_readings
                if recorded_at <= started_at
            ]
            after_candidates = [
                (recorded_at, reading)
                for recorded_at, reading in cloud_wearable_readings
                if recorded_at >= completed_at
            ]
            before = max(before_candidates, key=lambda item: item[0])[1] if before_candidates else None
            after = min(after_candidates, key=lambda item: item[0])[1] if after_candidates else None
            session["comparison"] = build_recovery_comparison(before, after)

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




@app.put("/api/settings/monitoring-interval")
def set_monitoring_interval(
    payload: MonitoringIntervalIn,
    user_id: str = Depends(current_user_id),
):
    multipliers = {"seconds": 1, "minutes": 60, "hours": 3600}
    interval_seconds = payload.interval_value * multipliers[payload.interval_unit]
    if not (
        MIN_MONITORING_INTERVAL_SECONDS
        <= interval_seconds
        <= MAX_MONITORING_INTERVAL_SECONDS
    ):
        raise HTTPException(
            status_code=422,
            detail="Monitoring interval must be between 5 seconds and 24 hours.",
        )

    updated_at = iso()
    with db_connection() as db:
        db.execute(
            """
            INSERT INTO user_settings (
                user_id, monitoring_interval_seconds,
                monitoring_interval_value, monitoring_interval_unit, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                monitoring_interval_seconds = excluded.monitoring_interval_seconds,
                monitoring_interval_value = excluded.monitoring_interval_value,
                monitoring_interval_unit = excluded.monitoring_interval_unit,
                updated_at = excluded.updated_at
            """,
            (
                user_id, interval_seconds, payload.interval_value,
                payload.interval_unit, updated_at,
            ),
        )

    preferences = get_settings(user_id)
    cloud_sync = firestore_write(
        ("users", user_id, "settings"),
        "preferences",
        {
            "haptic_enabled": preferences["haptic_enabled"],
            "music_mode": preferences["music_mode"],
            "warning_sound_enabled": preferences["warning_sound_enabled"],
            "notifications_enabled": preferences["notifications_enabled"],
            "monitoring_interval_seconds": interval_seconds,
            "monitoring_interval_value": payload.interval_value,
            "monitoring_interval_unit": payload.interval_unit,
            "updated_at": updated_at,
        },
    )
    return {
        "success": True,
        "monitoring_interval_seconds": interval_seconds,
        "monitoring_interval_value": payload.interval_value,
        "monitoring_interval_unit": payload.interval_unit,
        "cloud_sync": cloud_sync,
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
            "monitoring_interval_seconds": get_settings(user_id).get("monitoring_interval_seconds", DEFAULT_MONITORING_INTERVAL_SECONDS),
            "monitoring_interval_value": get_settings(user_id).get("monitoring_interval_value", 10),
            "monitoring_interval_unit": get_settings(user_id).get("monitoring_interval_unit", "minutes"),
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
        "library_limit": MAX_MUSIC_TRACKS_PER_USER,
        "remaining_slots": max(0, MAX_MUSIC_TRACKS_PER_USER - len(ordered)),
        "storage_configured": configured_music_bucket() is not None,
        "note": "Cloud music survives service restarts when Firebase Storage is configured; local fallback files may be ephemeral on hosted services.",
    }


@app.post("/api/music/upload", status_code=status.HTTP_201_CREATED)
async def upload_music_track(
    file: UploadFile = File(...),
    user_id: str = Depends(current_user_id),
):
    # Enforce a per-Google-UID library quota. Merge local and cloud IDs so
    # Render restarts cannot reset the ten-song allowance.
    with db_connection() as db:
        local_ids = {
            str(row["track_id"])
            for row in db.execute(
                "SELECT track_id FROM music_tracks WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        }
    cloud_items = firestore_read_collection(
        ("users", user_id, "music_tracks"), limit=200
    )
    cloud_ids = {
        str(item.get("track_id") or item.get("document_id") or "")
        for item in cloud_items
    }
    existing_ids = local_ids | {track_id for track_id in cloud_ids if track_id}
    if len(existing_ids) >= MAX_MUSIC_TRACKS_PER_USER:
        raise HTTPException(
            status_code=409,
            detail=(
                "Your music library is full (10/10). Remove one or more songs "
                "before uploading another track."
            ),
        )

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

    if bucket is None and APP_ENV == "production":
        # Render's local filesystem is ephemeral. Do not report a successful
        # production upload when the file would disappear on the next restart.
        raise HTTPException(
            status_code=503,
            detail="Durable music storage is not configured. Set FIREBASE_STORAGE_BUCKET and enable Firebase Storage.",
        )

    storage_blob = None
    if bucket is not None:
        try:
            storage_blob = bucket.blob(storage_path)
            storage_blob.upload_from_string(
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
        storage_backend = "local"

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

    # Cloud audio without durable Firestore metadata is an orphan waiting to
    # happen: require the UID-scoped metadata write before acknowledging it.
    cloud_sync = "not_configured"
    if storage_backend == "firebase_storage":
        cloud_sync = firestore_write(
            ("users", user_id, "music_tracks"), track_id, track
        )
        if cloud_sync != "synced":
            try:
                storage_blob.delete()
            except Exception:
                logger.exception(
                    "Could not clean up Firebase Storage object after metadata sync failure."
                )
            raise HTTPException(
                status_code=503,
                detail="Music metadata could not be backed up to Firestore. The upload was not saved; please retry.",
            )

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

    if storage_backend != "firebase_storage":
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


@app.delete("/api/music/{track_id}")
def delete_music_track(
    track_id: str,
    user_id: str = Depends(current_user_id),
):
    """Delete a song only from the authenticated user's own music library."""
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

    storage_backend = track.get("storage_backend", "local")
    storage_path = str(track.get("storage_path") or "")

    # Delete the audio object before its cloud metadata. If Storage deletion
    # fails, the Firestore document remains available for a safe retry after
    # an ephemeral SQLite database reset.
    try:
        if storage_backend == "firebase_storage":
            bucket = configured_music_bucket()
            if bucket is None:
                raise HTTPException(
                    status_code=503,
                    detail="Firebase Storage is not configured; the audio file could not be removed.",
                )
            blob = bucket.blob(storage_path)
            if blob.exists():
                blob.delete()
        elif storage_path:
            Path(storage_path).unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Music file deletion failed for track %s", track_id)
        raise HTTPException(
            status_code=503,
            detail="The audio file could not be removed. Please retry.",
        )

    # If Firestore fails after the blob is removed, keep its metadata so the
    # song remains discoverable and a retry can finish cleanup. A missing blob
    # is harmless on retry because Storage deletion is idempotent here.
    cloud_delete = firestore_delete_document(
        ("users", user_id, "music_tracks"), track_id
    )
    if cloud_delete == "failed" or (
        storage_backend == "firebase_storage" and cloud_delete != "synced"
    ):
        raise HTTPException(
            status_code=503,
            detail="Could not remove the song from your cloud library. Please retry.",
        )

    with db_connection() as db:
        db.execute(
            "DELETE FROM music_tracks WHERE track_id = ? AND user_id = ?",
            (track_id, user_id),
        )

    return {
        "success": True,
        "deleted_track_id": track_id,
        "cloud_sync": cloud_delete,
    }


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



@app.post("/api/personalization/labels", status_code=status.HTTP_201_CREATED)
def save_personal_feedback(
    payload: PersonalFeedbackIn,
    user_id: str = Depends(current_user_id),
):
    """Store a user's own context label for one reliable wearable reading."""
    with db_connection() as db:
        reading = db.execute(
            """
            SELECT id, timestamp, heart_rate_bpm, hrv_ms, gsr_raw,
                   skin_temperature_c, signal_quality, source
            FROM sensor_readings
            WHERE id = ? AND user_id = ?
            """,
            (payload.reading_id, user_id),
        ).fetchone()
        if reading is None:
            raise HTTPException(status_code=404, detail="Reading not found for this account.")
        if reading["source"] != "wearable" or reading["signal_quality"] == "poor":
            raise HTTPException(status_code=422, detail="Feedback requires a good/moderate wearable reading.")
        feedback_id = str(uuid.uuid4())
        created_at = iso()
        db.execute(
            """
            INSERT INTO reading_feedback (
                feedback_id, user_id, reading_id, label, context_note, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, reading_id) DO UPDATE SET
                label=excluded.label,
                context_note=excluded.context_note,
                created_at=excluded.created_at
            """,
            (feedback_id, user_id, payload.reading_id, payload.label,
             payload.context_note, created_at),
        )
        saved = db.execute(
            """
            SELECT feedback_id, user_id, reading_id, label, context_note, created_at
            FROM reading_feedback WHERE user_id = ? AND reading_id = ?
            """,
            (user_id, payload.reading_id),
        ).fetchone()
    item = dict(saved)
    firestore_write(("users", user_id, "reading_feedback"), item["reading_id"], item)
    return {"success": True, "feedback": item,
            "notice": "This is user-provided context, not a clinical ground-truth label."}


@app.get("/api/personalization/labels")
def get_personal_feedback(
    limit: int = Query(default=50, ge=1, le=200),
    user_id: str = Depends(current_user_id),
):
    with db_connection() as db:
        rows = db.execute(
            """
            SELECT feedback_id, reading_id, label, context_note, created_at
            FROM reading_feedback
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
    items = [dict(row) for row in rows]
    merged = {str(item["reading_id"]): item for item in items}
    for item in firestore_read_collection(
        ("users", user_id, "reading_feedback"), limit=limit, order_by="created_at"
    ):
        reading_id = str(item.get("reading_id") or item.get("document_id") or "")
        if reading_id:
            merged.setdefault(reading_id, item)
    items = sorted(
        merged.values(), key=lambda item: str(item.get("created_at") or ""), reverse=True
    )[:limit]
    return {"total": len(items), "labels": items,
            "counts": {key: sum(1 for item in items if item.get("label") == key)
                       for key in ("calm", "stressed", "active", "uncertain")}}


@app.get("/api/personalization/profile")
def personalization_profile(
    user_id: str = Depends(current_user_id),
):
    since = iso(utc_now() - timedelta(days=3))
    with db_connection() as db:
        local_sensor_rows = db.execute(
            """
            SELECT id, timestamp, heart_rate_bpm, hrv_ms, gsr_raw,
                   source, signal_quality
            FROM sensor_readings
            WHERE user_id = ? AND source = 'wearable'
              AND signal_quality IN ('good', 'moderate')
              AND timestamp >= ?
            ORDER BY timestamp ASC
            """,
            (user_id, since),
        ).fetchall()

        local_blink_rows = db.execute(
            """
            SELECT summary_id, blink_count, duration_seconds,
                   blinks_per_minute, recorded_at
            FROM blink_summaries
            WHERE user_id = ? AND recorded_at >= ?
            ORDER BY recorded_at ASC
            """,
            (user_id, since),
        ).fetchall()

    # Render's filesystem is ephemeral. Rebuild the learning window from
    # Firestore as well as SQLite, deduplicating by the stable reading ID.
    sensor_by_id = {
        str(row["id"]): dict(row)
        for row in local_sensor_rows
        if row["id"]
    }
    cloud_sensor_rows = firestore_read_collection(
        ("users", user_id, "sensor_readings"),
        limit=500,
        order_by="timestamp",
    )
    for item in cloud_sensor_rows:
        reading_id = str(item.get("id") or item.get("document_id") or "")
        timestamp = str(item.get("timestamp") or "")
        if (
            not reading_id
            or not timestamp
            or timestamp < since
            or str(item.get("source") or "") != "wearable"
            or str(item.get("signal_quality") or "") not in {"good", "moderate"}
        ):
            continue
        sensor_by_id.setdefault(reading_id, item)

    rows = sorted(
        sensor_by_id.values(),
        key=lambda item: str(item.get("timestamp") or ""),
    )
    day_keys = {str(row["timestamp"])[:10] for row in rows}
    def med(field):
        return median_or_none([
            float(row[field]) for row in rows if row.get(field) is not None
        ])
    days = len(day_keys)

    # Build a separate descriptive blink reference from completed scans.
    # Blink rate is not treated as a stress label and is not fused into the
    # wearable Random Forest prediction.
    blink_by_id = {
        str(row["summary_id"]): dict(row)
        for row in local_blink_rows
        if row["summary_id"]
    }
    cloud_blink_rows = firestore_read_collection(
        ("users", user_id, "blink_summaries"),
        limit=200,
        order_by="recorded_at",
    )
    for item in cloud_blink_rows:
        summary_id = str(item.get("summary_id") or item.get("document_id") or "")
        recorded_at = str(item.get("recorded_at") or "")
        if not summary_id or not recorded_at or recorded_at < since:
            continue
        blink_by_id.setdefault(summary_id, item)

    blink_summaries = sorted(
        blink_by_id.values(),
        key=lambda item: str(item.get("recorded_at") or ""),
    )
    blink_day_keys = {
        str(item.get("recorded_at") or "")[:10]
        for item in blink_summaries
        if item.get("recorded_at")
    }
    blink_rates = []
    for item in blink_summaries:
        try:
            rate = float(item.get("blinks_per_minute"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(rate) and rate >= 0:
            blink_rates.append(rate)
    blink_days = len(blink_day_keys)
    blink_count = len(blink_summaries)
    blink_reference = {
        "summary_count": blink_count,
        "days_observed": blink_days,
        "median_blinks_per_minute": median_or_none(blink_rates),
        "ready_for_preliminary_reference": blink_days >= 2 and blink_count >= 2,
        "recommended_reference_ready": blink_days >= 3 and blink_count >= 3,
        "notice": (
            "Descriptive camera blink reference only; it is not used as a stress score."
        ),
    }
    with db_connection() as db:
        label_rows = db.execute(
            "SELECT label, COUNT(*) AS total FROM reading_feedback WHERE user_id = ? GROUP BY label",
            (user_id,),
        ).fetchall()
    label_counts = {str(row["label"]): int(row["total"]) for row in label_rows}
    cloud_feedback = firestore_read_collection(
        ("users", user_id, "reading_feedback"), limit=200, order_by="created_at"
    )
    local_feedback_ids = set()
    with db_connection() as db:
        local_feedback_ids = {
            str(row["reading_id"])
            for row in db.execute(
                "SELECT reading_id FROM reading_feedback WHERE user_id = ?", (user_id,)
            ).fetchall()
        }
    for item in cloud_feedback:
        if str(item.get("reading_id") or "") in local_feedback_ids:
            continue
        label = str(item.get("label") or "")
        if label in {"calm", "stressed", "active", "uncertain"}:
            label_counts[label] = label_counts.get(label, 0) + 1
    labeled_total = sum(label_counts.values())
    return {
        "days_observed": days,
        "labeled_readings": labeled_total,
        "label_counts": label_counts,
        "personalized_model_status": "data_collection_only",
        "personalized_model_ready": False,
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
        "blink_reference": blink_reference,
        "notice": "Two distinct days allow a preliminary wearable reference; three or more days are recommended. Camera blink history is reported separately and is not used to infer stress. These are descriptive references, not a newly trained or clinically validated model.",
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


@app.get("/api/telegram-contacts")
def list_telegram_emergency_contacts(user_id: str = Depends(current_user_id)):
    if FIRESTORE_CLIENT is not None:
        try:
            sync_local_telegram_state_to_firestore(user_id)
            invitations = telegram_firestore_documents(
                telegram_user_collection(user_id, "telegram_emergency_invites")
            )
            contacts = telegram_firestore_documents(
                telegram_user_collection(user_id, "telegram_emergency_contacts")
            )
            contact_by_invite = {
                str(item.get("invite_id")): item
                for item in contacts
            }
            now = utc_now()
            result_contacts = []
            for invite in invitations:
                if invite.get("revoked_at"):
                    continue
                invite_id = str(invite.get("invite_id") or invite.get("document_id"))
                linked = contact_by_invite.get(invite_id)
                if linked and linked.get("active"):
                    state = "connected"
                elif linked:
                    state = "stopped"
                else:
                    state = (
                        "expired"
                        if parse_iso(invite["expires_at"]) <= now
                        else "pending"
                    )
                result_contacts.append({
                    "id": invite_id,
                    "contact_name": invite.get("contact_name", "Emergency contact"),
                    "username": (linked or {}).get("username"),
                    "first_name": (linked or {}).get("first_name"),
                    "created_at": invite.get("created_at"),
                    "expires_at": invite.get("expires_at"),
                    "last_seen_at": (linked or {}).get("last_seen_at"),
                    "status": state,
                })
            result_contacts.sort(
                key=lambda item: str(item.get("created_at") or ""),
                reverse=True,
            )
        except Exception:
            logger.exception("Could not load durable Telegram contacts from Firestore.")
            raise HTTPException(
                status_code=503,
                detail="Telegram contact storage is temporarily unavailable. Please retry.",
            )
    else:
        if APP_ENV != "development":
            raise HTTPException(
                status_code=503,
                detail="Persistent Telegram contact storage is not configured.",
            )
        with db_connection() as db:
            rows = db.execute(
                """
                SELECT i.invite_id, i.contact_name, i.created_at, i.expires_at,
                       c.username, c.first_name, c.active AS contact_active,
                       c.last_seen_at
                FROM telegram_emergency_invites i
                LEFT JOIN telegram_emergency_contacts c
                    ON c.invite_id = i.invite_id AND c.user_id = i.user_id
                WHERE i.user_id = ? AND i.revoked_at IS NULL
                ORDER BY i.created_at DESC
                """,
                (user_id,),
            ).fetchall()
        now = utc_now()
        result_contacts = []
        for row in rows:
            if row["contact_active"] == 1:
                state = "connected"
            elif row["contact_active"] == 0:
                state = "stopped"
            else:
                state = "expired" if parse_iso(row["expires_at"]) <= now else "pending"
            result_contacts.append({
                "id": row["invite_id"],
                "contact_name": row["contact_name"],
                "username": row["username"],
                "first_name": row["first_name"],
                "created_at": row["created_at"],
                "expires_at": row["expires_at"],
                "last_seen_at": row["last_seen_at"],
                "status": state,
            })

    return {
        "contacts": result_contacts,
        "telegram_configured": telegram_configured(),
        "bot_username": TELEGRAM_BOT_USERNAME or None,
    }


@app.post("/api/telegram-contacts/invites")
def create_telegram_emergency_invite(
    payload: TelegramEmergencyInviteIn,
    user_id: str = Depends(current_user_id),
):
    if not telegram_configured():
        raise HTTPException(
            status_code=503,
            detail="Telegram bot is not configured. Add TELEGRAM_BOT_TOKEN and TELEGRAM_BOT_USERNAME in Render Environment.",
        )
    if FIRESTORE_CLIENT is None and APP_ENV != "development":
        raise HTTPException(
            status_code=503,
            detail="Persistent Telegram contact storage is not configured.",
        )

    contact_name = payload.contact_name.strip()
    if len(contact_name) < 2:
        raise HTTPException(status_code=422, detail="Enter a contact name with at least 2 characters.")

    now = utc_now()
    expires_at = now + timedelta(days=TELEGRAM_INVITE_TTL_DAYS)
    invite_token = secrets.token_urlsafe(24)
    invite_id = uuid.uuid4().hex
    token_hash = token_digest(invite_token)
    invite_data = {
        "invite_id": invite_id,
        "user_id": user_id,
        "contact_name": contact_name,
        "token_hash": token_hash,
        "created_at": iso(now),
        "expires_at": iso(expires_at),
        "revoked_at": None,
    }

    if FIRESTORE_CLIENT is not None:
        try:
            sync_local_telegram_state_to_firestore(user_id)
            existing_invites = telegram_firestore_documents(
                telegram_user_collection(user_id, "telegram_emergency_invites")
            )
            active_count = sum(
                1 for item in existing_invites
                if not item.get("revoked_at")
                and parse_iso(item["expires_at"]) > now
            )
            if active_count >= TELEGRAM_MAX_CONTACTS:
                raise HTTPException(
                    status_code=429,
                    detail=f"You can have up to {TELEGRAM_MAX_CONTACTS} active Telegram contacts.",
                )
            telegram_user_collection(
                user_id, "telegram_emergency_invites"
            ).document(invite_id).set(invite_data)
            telegram_global_document(
                "telegram_invite_tokens", token_hash
            ).set({"user_id": user_id, "invite_id": invite_id})
        except HTTPException:
            raise
        except Exception:
            logger.exception("Could not save Telegram invitation in Firestore.")
            try:
                telegram_user_collection(
                    user_id, "telegram_emergency_invites"
                ).document(invite_id).delete()
                telegram_global_document(
                    "telegram_invite_tokens", token_hash
                ).delete()
            except Exception:
                logger.exception("Could not roll back partial Telegram invitation.")
            raise HTTPException(
                status_code=503,
                detail="Telegram invite could not be saved safely. Please retry.",
            )

    # Keep SQLite as a local cache/development fallback, not the production source of truth.
    try:
        with db_connection() as db:
            db.execute(
                """
                INSERT INTO telegram_emergency_invites (
                    invite_id, user_id, contact_name, token_hash, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(invite_id) DO NOTHING
                """,
                (invite_id, user_id, contact_name, token_hash, iso(now), iso(expires_at)),
            )
    except Exception:
        logger.exception("Could not update local Telegram invite cache.")

    invite_url = f"https://t.me/{TELEGRAM_BOT_USERNAME}?start={invite_token}"
    return {
        "success": True,
        "id": invite_id,
        "contact_name": contact_name,
        "invite_url": invite_url,
        "expires_at": iso(expires_at),
    }


@app.delete("/api/telegram-contacts/{invite_id}")
def revoke_telegram_emergency_contact(invite_id: str, user_id: str = Depends(current_user_id)):
    if FIRESTORE_CLIENT is not None:
        try:
            invite_ref = telegram_user_collection(
                user_id, "telegram_emergency_invites"
            ).document(invite_id)
            invite_snapshot = invite_ref.get()
            if not invite_snapshot.exists:
                raise HTTPException(status_code=404, detail="Telegram emergency contact was not found.")
            invite_data = invite_snapshot.to_dict() or {}
            token_hash = invite_data.get("token_hash")
            contacts_ref = telegram_user_collection(
                user_id, "telegram_emergency_contacts"
            )
            for snapshot in contacts_ref.where("invite_id", "==", invite_id).stream():
                contact = snapshot.to_dict() or {}
                chat_id = contact.get("chat_id")
                if chat_id is not None:
                    telegram_global_document(
                        "telegram_chat_links",
                        telegram_chat_document_id(str(chat_id)),
                    ).delete()
                snapshot.reference.delete()
            invite_ref.delete()
            if token_hash:
                telegram_global_document(
                    "telegram_invite_tokens", str(token_hash)
                ).delete()
        except HTTPException:
            raise
        except Exception:
            logger.exception("Could not revoke Telegram contact in Firestore.")
            raise HTTPException(
                status_code=503,
                detail="Telegram contact could not be removed right now. Please retry.",
            )

    with db_connection() as db:
        db.execute(
            "DELETE FROM telegram_emergency_invites WHERE invite_id = ? AND user_id = ?",
            (invite_id, user_id),
        )
    return {"success": True}


@app.post("/api/telegram-contacts/test")
def test_telegram_emergency_contact(user_id: str = Depends(current_user_id)):
    if not telegram_configured():
        raise HTTPException(status_code=503, detail="Telegram bot is not configured on the server.")
    if FIRESTORE_CLIENT is not None:
        try:
            chat_ids = telegram_firestore_active_chat_ids(user_id)
        except Exception:
            logger.exception("Could not load Telegram test recipients from Firestore.")
            raise HTTPException(
                status_code=503,
                detail="Telegram contact storage is temporarily unavailable. Please retry.",
            )
    else:
        if APP_ENV != "development":
            raise HTTPException(status_code=503, detail="Persistent Telegram contact storage is not configured.")
        with db_connection() as db:
            rows = db.execute(
                """
                SELECT chat_id
                FROM telegram_emergency_contacts
                WHERE user_id = ? AND active = 1
                """,
                (user_id,),
            ).fetchall()
        chat_ids = [str(row["chat_id"]) for row in rows]

    if not chat_ids:
        raise HTTPException(status_code=409, detail="No Telegram contact has started the bot from an active invitation yet.")

    message = (
        "NeuroSense test alert\n\n"
        "Telegram alerts are connected. This is a test, not an emergency alert."
    )
    sent_count = sum(1 for chat_id in chat_ids if send_telegram_message(chat_id, message))
    if not sent_count:
        raise HTTPException(status_code=502, detail="Telegram could not deliver the test message to any registered contact.")
    return {"success": True, "status": "sent", "recipient_count": sent_count}


@app.post("/api/telegram/webhook")
def telegram_webhook(
    update: dict,
    x_telegram_bot_api_secret_token: Annotated[
        str | None, Header(alias="X-Telegram-Bot-Api-Secret-Token")
    ] = None,
):
    expected_secret = telegram_webhook_secret()
    if not TELEGRAM_BOT_TOKEN or not x_telegram_bot_api_secret_token or not secrets.compare_digest(
        x_telegram_bot_api_secret_token, expected_secret
    ):
        raise HTTPException(status_code=401, detail="Invalid Telegram webhook secret.")

    message = update.get("message") or {}
    chat = message.get("chat") or {}
    sender = message.get("from") or {}
    chat_type = chat.get("type")
    chat_id = chat.get("id")
    text_value = str(message.get("text") or "").strip()
    if chat_type != "private" or chat_id is None or not text_value:
        return {"ok": True}

    parts = text_value.split(maxsplit=1)
    command = parts[0].split("@", 1)[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    chat_id_text = str(chat_id)
    chat_document_id = telegram_chat_document_id(chat_id_text)

    if command == "/stop":
        if FIRESTORE_CLIENT is not None:
            try:
                chat_ref = telegram_global_document(
                    "telegram_chat_links", chat_document_id
                )
                chat_snapshot = chat_ref.get()
                if chat_snapshot.exists:
                    link_data = chat_snapshot.to_dict() or {}
                    owner_id = str(link_data.get("user_id") or "")
                    contact_id = str(link_data.get("contact_id") or chat_document_id)
                    if owner_id:
                        telegram_user_collection(
                            owner_id, "telegram_emergency_contacts"
                        ).document(contact_id).set(
                            {"active": False, "last_seen_at": iso()},
                            merge=True,
                        )
                    chat_ref.set({"active": False}, merge=True)
            except Exception:
                logger.exception("Could not persist Telegram /stop in Firestore.")
                raise HTTPException(status_code=503, detail="Telegram contact storage is temporarily unavailable.")
        with db_connection() as db:
            db.execute(
                "UPDATE telegram_emergency_contacts SET active = 0 WHERE chat_id = ?",
                (chat_id_text,),
            )
        send_telegram_message(chat_id, "NeuroSense alerts stopped. Your Telegram chat is no longer receiving alerts.")
        return {"ok": True}

    if command != "/start":
        return {"ok": True}

    if not argument:
        send_telegram_message(
            chat_id,
            "To connect this Telegram account, open the personal invitation link shared by your NeuroSense contact and press Start.",
        )
        return {"ok": True}

    now = utc_now()
    now_text = iso(now)

    if FIRESTORE_CLIENT is not None:
        try:
            token_hash = token_digest(argument)
            token_snapshot = telegram_global_document(
                "telegram_invite_tokens", token_hash
            ).get()
            token_data = token_snapshot.to_dict() if token_snapshot.exists else None

            # A local invite may predate this Firestore-backed release. Migrate it
            # when this webhook reaches the same live instance that created it.
            if not token_data:
                with db_connection() as db:
                    invite_row = db.execute(
                        """
                        SELECT invite_id, user_id, contact_name, token_hash,
                               created_at, expires_at, revoked_at
                        FROM telegram_emergency_invites
                        WHERE token_hash = ? AND revoked_at IS NULL
                        """,
                        (token_hash,),
                    ).fetchone()
                if invite_row:
                    legacy_invite = dict(invite_row)
                    owner_id = str(legacy_invite["user_id"])
                    invite_id = str(legacy_invite["invite_id"])
                    telegram_user_collection(
                        owner_id, "telegram_emergency_invites"
                    ).document(invite_id).set(legacy_invite)
                    telegram_global_document(
                        "telegram_invite_tokens", token_hash
                    ).set({"user_id": owner_id, "invite_id": invite_id})
                    token_data = {"user_id": owner_id, "invite_id": invite_id}

            if not token_data:
                send_telegram_message(chat_id, "This NeuroSense invitation is invalid or has been revoked. Ask the NeuroSense user for a new link.")
                return {"ok": True}

            owner_id = str(token_data["user_id"])
            invite_id = str(token_data["invite_id"])
            invite_snapshot = telegram_user_collection(
                owner_id, "telegram_emergency_invites"
            ).document(invite_id).get()
            if not invite_snapshot.exists:
                send_telegram_message(chat_id, "This NeuroSense invitation is invalid or has been revoked. Ask the NeuroSense user for a new link.")
                return {"ok": True}
            invite = invite_snapshot.to_dict() or {}
            if invite.get("revoked_at") or parse_iso(invite["expires_at"]) <= now:
                send_telegram_message(chat_id, "This NeuroSense invitation has expired. Ask the NeuroSense user to create a new Telegram link.")
                return {"ok": True}

            chat_link_ref = telegram_global_document(
                "telegram_chat_links", chat_document_id
            )
            existing_snapshot = chat_link_ref.get()
            existing = existing_snapshot.to_dict() if existing_snapshot.exists else None
            if existing and (
                str(existing.get("invite_id")) != invite_id
                or str(existing.get("user_id")) != owner_id
            ):
                send_telegram_message(chat_id, "This Telegram account is already linked to another NeuroSense contact. Ask the owner to remove the previous link first.")
                return {"ok": True}

            contacts_ref = telegram_user_collection(
                owner_id, "telegram_emergency_contacts"
            )
            contact_ref = contacts_ref.document(chat_document_id)
            contact_snapshot = contact_ref.get()
            if not contact_snapshot.exists:
                active_contacts = [
                    item for item in telegram_firestore_documents(contacts_ref)
                    if item.get("active")
                ]
                if len(active_contacts) >= TELEGRAM_MAX_CONTACTS:
                    send_telegram_message(chat_id, "This NeuroSense account has reached its Telegram contact limit.")
                    return {"ok": True}

            contact_data = {
                "contact_id": chat_document_id,
                "invite_id": invite_id,
                "user_id": owner_id,
                "chat_id": chat_id_text,
                "telegram_user_id": (
                    str(sender.get("id"))
                    if sender.get("id") is not None else None
                ),
                "username": str(sender.get("username") or "")[:64] or None,
                "first_name": str(sender.get("first_name") or "")[:120] or None,
                "created_at": (
                    (contact_snapshot.to_dict() or {}).get("created_at")
                    if contact_snapshot.exists else now_text
                ),
                "last_seen_at": now_text,
                "active": True,
            }
            contact_ref.set(contact_data)
            chat_link_ref.set({
                "user_id": owner_id,
                "invite_id": invite_id,
                "contact_id": chat_document_id,
                "active": True,
            })
        except Exception:
            logger.exception("Could not persist Telegram /start connection in Firestore.")
            raise HTTPException(
                status_code=503,
                detail="Telegram contact storage is temporarily unavailable. Telegram will retry this connection.",
            )

        # Best-effort SQLite mirror for local development and older code paths.
        try:
            with db_connection() as db:
                existing_row = db.execute(
                    "SELECT contact_id FROM telegram_emergency_contacts WHERE chat_id = ?",
                    (chat_id_text,),
                ).fetchone()
                if existing_row:
                    db.execute(
                        """
                        UPDATE telegram_emergency_contacts
                        SET invite_id = ?, user_id = ?, telegram_user_id = ?,
                            username = ?, first_name = ?, last_seen_at = ?, active = 1
                        WHERE chat_id = ?
                        """,
                        (
                            invite_id, owner_id, contact_data["telegram_user_id"],
                            contact_data["username"], contact_data["first_name"],
                            now_text, chat_id_text,
                        ),
                    )
                else:
                    db.execute(
                        """
                        INSERT INTO telegram_emergency_contacts (
                            contact_id, invite_id, user_id, chat_id, telegram_user_id,
                            username, first_name, created_at, last_seen_at, active
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """,
                        (
                            chat_document_id, invite_id, owner_id, chat_id_text,
                            contact_data["telegram_user_id"], contact_data["username"],
                            contact_data["first_name"], contact_data["created_at"], now_text,
                        ),
                    )
        except Exception:
            logger.exception("Could not update local Telegram contact cache.")

        send_telegram_message(
            chat_id,
            f"Connected to NeuroSense alerts for {invite.get('contact_name', 'your contact')}. You will receive experimental wellness check-in alerts here. Send /stop anytime to disconnect.",
        )
        return {"ok": True}

    # Development-only SQLite path when Firebase is not configured.
    with db_connection() as db:
        invite = db.execute(
            """
            SELECT invite_id, user_id, contact_name, expires_at
            FROM telegram_emergency_invites
            WHERE token_hash = ? AND revoked_at IS NULL
            """,
            (token_digest(argument),),
        ).fetchone()
        if invite is None:
            send_telegram_message(chat_id, "This NeuroSense invitation is invalid or has been revoked. Ask the NeuroSense user for a new link.")
            return {"ok": True}
        if parse_iso(invite["expires_at"]) <= now:
            send_telegram_message(chat_id, "This NeuroSense invitation has expired. Ask the NeuroSense user to create a new Telegram link.")
            return {"ok": True}
        existing = db.execute(
            "SELECT invite_id, user_id FROM telegram_emergency_contacts WHERE chat_id = ?",
            (chat_id_text,),
        ).fetchone()
        if existing and (
            existing["invite_id"] != invite["invite_id"]
            or existing["user_id"] != invite["user_id"]
        ):
            send_telegram_message(chat_id, "This Telegram account is already linked to another NeuroSense contact. Ask the owner to remove the previous link first.")
            return {"ok": True}
        if existing is None:
            count = db.execute(
                "SELECT COUNT(*) AS total FROM telegram_emergency_contacts WHERE user_id = ? AND active = 1",
                (invite["user_id"],),
            ).fetchone()["total"]
            if int(count) >= TELEGRAM_MAX_CONTACTS:
                send_telegram_message(chat_id, "This NeuroSense account has reached its Telegram contact limit.")
                return {"ok": True}
            db.execute(
                """
                INSERT INTO telegram_emergency_contacts (
                    contact_id, invite_id, user_id, chat_id, telegram_user_id,
                    username, first_name, created_at, last_seen_at, active
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    uuid.uuid4().hex, invite["invite_id"], invite["user_id"],
                    chat_id_text,
                    str(sender.get("id")) if sender.get("id") is not None else None,
                    str(sender.get("username") or "")[:64] or None,
                    str(sender.get("first_name") or "")[:120] or None,
                    now_text, now_text,
                ),
            )
        else:
            db.execute(
                """
                UPDATE telegram_emergency_contacts
                SET telegram_user_id = ?, username = ?, first_name = ?,
                    last_seen_at = ?, active = 1
                WHERE chat_id = ?
                """,
                (
                    str(sender.get("id")) if sender.get("id") is not None else None,
                    str(sender.get("username") or "")[:64] or None,
                    str(sender.get("first_name") or "")[:120] or None,
                    now_text, chat_id_text,
                ),
            )

    send_telegram_message(
        chat_id,
        f"Connected to NeuroSense alerts for {invite['contact_name']}. You will receive experimental wellness check-in alerts here. Send /stop anytime to disconnect.",
    )
    return {"ok": True}




# ============================================================
# 26. FACE SCAN STATUS AND PRIVACY
# ============================================================

@app.get("/api/face-scan/status")
def face_scan_status():
    openface_executable = Path(os.getenv(
        "OPENFACE_FEATURE_EXTRACTION_PATH",
        "/opt/OpenFace/build/bin/FeatureExtraction",
    ))
    webcam_inference_available = (
        face_valence_live_inference_enabled()
        and FACE_VALENCE_MODEL is not None
        and openface_executable.is_file()
    )

    return {
        "available": True,
        "status": (
            "browser_local_scan_with_research_valence_opt_in"
            if webcam_inference_available
            else "browser_local_scan_affec_validation_pending"
        ),
        "processing": (
            "browser_local_rppg_blink_and_explicit_research_valence"
            if webcam_inference_available
            else "browser_local_rppg_and_blink_only"
        ),
        "camera_policy": (
            "Camera remains user-triggered and OFF by default. Raw frames are "
            "not uploaded by the browser-local rPPG/blink scan. AFFEC webcam "
            "inference is disabled until its stimulus-window feature parity "
            "and new-user validation are completed."
        ),
        "rppg_frames_uploaded": False,
        "valence_clip_uploaded_on_explicit_action": webcam_inference_available,
        "valence_clip_retained": False,
        "openface_available": openface_executable.is_file(),
        "face_valence_model_loaded": FACE_VALENCE_MODEL is not None,
        "blink_count_available": True,
        "derived_blink_metrics_can_be_saved_per_user": True,
        "felt_valence_inference": webcam_inference_available,
        "stress_inference": False,
        "message": (
            "The rPPG and blink scan remains browser-local. AFFEC was trained "
            "on stimulus-locked video-phase windows (variable duration; "
            "median 4.61 seconds), not arbitrary 4.8-second webcam selfies. "
            "Its live webcam output is disabled until numerical feature parity "
            "and new-user validation are completed. This is not a stress "
            "score or medical assessment."
        )
    }


@app.get("/api/face/valence/status")
def face_valence_status():
    openface_executable = Path(os.getenv(
        "OPENFACE_FEATURE_EXTRACTION_PATH",
        "/opt/OpenFace/build/bin/FeatureExtraction",
    ))
    parity_verified = FACE_VALENCE_FEATURE_PARITY_VERIFIED
    new_user_validation_verified = FACE_VALENCE_NEW_USER_VALIDATION_VERIFIED
    effective_enabled = face_valence_live_inference_enabled()
    return {
        "model_loaded": FACE_VALENCE_MODEL is not None,
        "openface_available": openface_executable.is_file(),
        "webcam_experiment_requested": FACE_VALENCE_WEBCAM_EXPERIMENT_ENABLED,
        "webcam_experiment_enabled": effective_enabled,
        "inference_available": bool(
            effective_enabled
            and FACE_VALENCE_MODEL is not None
            and openface_executable.is_file()
        ),
        "feature_parity_verified": parity_verified,
        "new_user_validation_verified": new_user_validation_verified,
        "requested_capture_seconds": 4.8,
        "live_window_context": "first 4.8 seconds of the submitted clip",
        "training_window_context": (
            "stimulus-locked video phase from video onset to last_frame_video; "
            "variable duration (metadata median 4.61 seconds, observed range "
            "3.53 to approximately 5.3 seconds)"
        ),
        "window_alignment_verified": parity_verified,
        "minimum_valid_frames": 40,
        "confidence_threshold": 0.8,
        "feature_count": 50,
        "target": "beh_f_emotion_v",
        "interpretation": "felt_valence_only_not_stress",
        "cross_validation_mean_fold_r2": 0.0795,
        "high_low_valence_balanced_accuracy": 0.5971,
        "validation_note": (
            "Offline subject-wise cross-validation only; no new-user validation. "
            "Live-window numerical parity is not yet verified."
        ),
    }


@app.post("/api/face/valence/predict")
async def predict_face_valence(
    video: UploadFile = File(...),
    user_id: str = Depends(current_user_id),
):
    """Run the gated experimental OpenFace felt-valence inference for one user."""
    del user_id  # Authentication gates access; no raw video or identity is persisted.

    if not face_valence_live_inference_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "AFFEC webcam inference is disabled. Live inference requires "
                "explicit experiment opt-in, verified training-to-live feature "
                "parity, and completed new-user validation. The saved model "
                "was trained on variable stimulus-locked video-phase windows, "
                "not arbitrary 4.8-second webcam clips."
            ),
        )

    if FACE_VALENCE_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The felt-valence model is not loaded. Configure "
                "NEUROSENSE_FACE_VALENCE_MODEL_URL in the deployment environment."
            ),
        )

    content_type = (video.content_type or "").lower()
    if not content_type.startswith("video/"):
        raise HTTPException(
            status_code=415,
            detail="Upload a browser-recorded video clip.",
        )

    payload = await video.read(FACE_VALENCE_MAX_VIDEO_BYTES + 1)
    await video.close()
    if len(payload) > FACE_VALENCE_MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail="The camera clip exceeds the 12 MB upload limit.",
        )

    openface_executable = Path(os.getenv(
        "OPENFACE_FEATURE_EXTRACTION_PATH",
        "/opt/OpenFace/build/bin/FeatureExtraction",
    ))
    openface_root = Path(os.getenv("OPENFACE_ROOT", "/opt/OpenFace"))

    if not openface_executable.is_file():
        raise HTTPException(
            status_code=503,
            detail="OpenFace FeatureExtraction is unavailable on this backend.",
        )

    try:
        result = await asyncio.to_thread(
            predict_from_video_bytes,
            FACE_VALENCE_MODEL,
            payload,
            openface_executable,
            openface_root,
        )
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=504,
            detail="OpenFace processing timed out. Please retry with a clear, short clip.",
        ) from error
    except Exception as error:
        logger.exception("AFFEC felt-valence inference failed.")
        raise HTTPException(
            status_code=502,
            detail="Face-valence feature extraction or inference failed.",
        ) from error

    return {
        **result,
        "target": "beh_f_emotion_v",
        "label": "felt_valence",
        "research_only": True,
        "model_warning": (
            "Experimental felt-valence estimate only; not a stress score, "
            "diagnosis, or validated emotion classification. CV mean-fold "
            "R2 is approximately 0.08; new-user validation has not been performed."
        ),
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
