"""
NeuroSense - Backend API Tests

Run from the project root:
    python -m pytest backend/tests/test_main.py -v

Tests use an isolated temporary SQLite database. They do not modify
the developer's local NeuroSense database or require real hardware.
"""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from backend import main as api


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Start the API with an isolated test database and no cloud services."""
    monkeypatch.setattr(api, "DB_PATH", tmp_path / "neurosense_test.db")
    monkeypatch.setattr(api, "APP_ENV", "development")
    monkeypatch.setattr(api, "FIREBASE_APP", None)
    monkeypatch.setattr(api, "FIRESTORE_CLIENT", None)
    monkeypatch.setattr(api, "STRESS_MODEL", None)

    # Keep API tests independent from external Firebase credentials
    # and from loading a locally trained model into the test process.
    monkeypatch.setattr(api, "initialize_firebase", lambda: None)
    monkeypatch.setattr(api, "initialize_ml_model", lambda: None)

    with TestClient(api.app) as test_client:
        yield test_client


def test_application_starts(client):
    """Check that the FastAPI application responds."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_health_endpoint(client):
    """Check backend and isolated database health."""
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "healthy"
    assert body["database"] == "healthy"


def test_system_status_endpoint(client):
    """Check system status endpoint."""
    response = client.get("/api/system/status")

    assert response.status_code == 200
    body = response.json()

    assert body["backend"] == "online"
    assert body["database"] == "sqlite"


@pytest.mark.parametrize(
    ("path", "method"),
    [
        ("/health", "get"),
        ("/api/system/status", "get"),
        ("/api/sensors", "post"),
        ("/api/sensors/latest", "get"),
        ("/api/sensors/history", "get"),
        ("/api/device/status", "get"),
        ("/api/devices", "get"),
        ("/api/devices/pair", "post"),
    ],
)
def test_core_api_routes_are_registered(client, path, method):
    """Verify that the expected API routes exist in FastAPI."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json().get("paths", {})

    assert path in paths, f"API route is missing: {path}"
    assert method in paths[path], (
        f"HTTP method {method.upper()} is missing for {path}"
    )


def test_sensor_endpoint_rejects_empty_payload(client):
    """An empty sensor payload should fail Pydantic validation."""
    response = client.post(
        "/api/sensors",
        json={},
        headers={"X-Demo-User": "neurosense-test-user"},
    )

    assert response.status_code == 422


def test_sensor_post_accepts_valid_simulated_reading_and_persists_it(client):
    """API Test 1: valid sensor POST returns 201 and stores the reading."""
    headers = {"X-Demo-User": "neurosense-api-test-user"}
    payload = {
        "heart_rate_bpm": 76,
        "hrv_ms": 42,
        "gsr_raw": 1850,
        "skin_temperature_c": 33.2,
        "signal_quality": "moderate",
        "source": "simulated",
    }

    # The isolated database should start empty for this test.
    before = client.get("/api/sensors/latest", headers=headers)
    assert before.status_code == 200
    assert before.json()["data"] is None

    response = client.post(
        "/api/sensors",
        json=payload,
        headers=headers,
    )

    assert response.status_code == 201
    body = response.json()

    assert body["success"] is True
    assert body["reading"]["heart_rate_bpm"] == 76
    assert body["reading"]["hrv_ms"] == 42
    assert body["reading"]["gsr_raw"] == 1850
    assert body["reading"]["skin_temperature_c"] == 33.2
    assert body["reading"]["source"] == "simulated"
    assert body["stress"]["status"] == "simulated_data"

    # Confirm the same reading can be retrieved from SQLite.
    latest = client.get("/api/sensors/latest", headers=headers)
    assert latest.status_code == 200
    saved = latest.json()["data"]

    assert saved is not None
    assert saved["id"] == body["reading"]["id"]
    assert saved["heart_rate_bpm"] == 76
    assert saved["gsr_raw"] == 1850

    history = client.get("/api/sensors/history", headers=headers)
    assert history.status_code == 200
    assert history.json()["total"] == 1
    assert history.json()["readings"][0]["id"] == saved["id"]


def test_recovery_session_rejects_completion_before_start(client):
    """Recovery sessions must not complete before their start time."""
    response = client.post(
        "/api/recovery/sessions",
        json={
            "activity": "breathing",
            "started_at": "2026-11-08T09:00:00+00:00",
            "completed_at": "2026-11-08T08:59:00+00:00",
            "duration_seconds": 60,
            "completed": True,
        },
        headers={"X-Demo-User": "api-test-recovery-invalid-time"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "completed_at cannot be earlier than started_at."


def test_recovery_session_returns_before_after_comparison(client):
    """Recovery API should compare wearable readings around the session."""
    headers = {"X-Demo-User": "api-test-recovery-comparison"}
    now = datetime.now(timezone.utc)
    started = now - timedelta(seconds=90)
    completed = now - timedelta(seconds=30)
    before_time = started - timedelta(seconds=15)
    after_time = completed + timedelta(seconds=15)

    before = client.post(
        "/api/sensors",
        json={
            "timestamp": before_time.isoformat(),
            "heart_rate_bpm": 82,
            "hrv_ms": 32,
            "gsr_raw": 1900,
            "skin_temperature_c": 33.2,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    after = client.post(
        "/api/sensors",
        json={
            "timestamp": after_time.isoformat(),
            "heart_rate_bpm": 75,
            "hrv_ms": 40,
            "gsr_raw": 1600,
            "skin_temperature_c": 33.0,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    assert before.status_code == 201
    assert after.status_code == 201

    response = client.post(
        "/api/recovery/sessions",
        json={
            "activity": "breathing",
            "started_at": started.isoformat(),
            "completed_at": completed.isoformat(),
            "duration_seconds": 60,
            "completed": True,
        },
        headers=headers,
    )

    assert response.status_code == 201
    comparison = response.json()["comparison"]
    assert comparison is not None
    assert comparison["heart_rate_bpm"]["before"] == 82
    assert comparison["heart_rate_bpm"]["after"] == 75
    assert comparison["heart_rate_bpm"]["change"] == -7
    assert comparison["hrv_ms"]["change"] == 8
    assert comparison["gsr_raw"]["change"] == -300


def test_due_summary_generation_is_visible_in_summary_history(client):
    """A completed three-hour window should appear through the summaries API."""
    headers = {"X-Demo-User": "api-test-three-hour-summary"}
    old_timestamp = datetime.now(timezone.utc) - timedelta(hours=7)

    reading = client.post(
        "/api/sensors",
        json={
            "timestamp": old_timestamp.isoformat(),
            "heart_rate_bpm": 80,
            "hrv_ms": 36,
            "gsr_raw": 1700,
            "skin_temperature_c": 33.1,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers=headers,
    )
    assert reading.status_code == 201

    created_count = api.create_due_summaries()
    assert created_count >= 1

    response = client.get("/api/summaries", headers=headers)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] >= 1
    assert body["summaries"][0]["reading_count"] == 1
    assert body["summaries"][0]["reliable_reading_count"] == 1
    assert body["summaries"][0]["avg_heart_rate"] == 80
    assert body["summaries"][0]["avg_hrv"] == 36
    assert body["summaries"][0]["avg_gsr"] == 1700




def test_sensor_post_rejects_out_of_range_heart_rate(client):
    """API Test 2a: reject physiological values outside the schema range."""
    response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 221,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-validation-user"},
    )

    assert response.status_code == 422


def test_sensor_post_rejects_timestamp_too_far_in_future(client):
    """API Test 2b: reject readings more than five minutes ahead."""
    from datetime import datetime, timedelta, timezone

    future_timestamp = (
        datetime.now(timezone.utc) + timedelta(minutes=10)
    ).isoformat()

    response = client.post(
        "/api/sensors",
        json={
            "timestamp": future_timestamp,
            "heart_rate_bpm": 72,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-validation-user"},
    )

    assert response.status_code == 422
    assert "future" in response.json()["detail"].lower()


def test_sensor_post_rejects_reading_older_than_24_hours(client):
    """API Test 2c: reject stale sensor readings."""
    from datetime import datetime, timedelta, timezone

    old_timestamp = (
        datetime.now(timezone.utc) - timedelta(hours=25)
    ).isoformat()

    response = client.post(
        "/api/sensors",
        json={
            "timestamp": old_timestamp,
            "heart_rate_bpm": 72,
            "source": "simulated",
            "signal_quality": "good",
        },
        headers={"X-Demo-User": "neurosense-validation-user"},
    )

    assert response.status_code == 422
    assert "24-hour" in response.json()["detail"]


def test_sensor_readings_are_isolated_between_demo_users(client):
    """API Test 2d: one demo user must not see another user's readings."""
    alice_headers = {"X-Demo-User": "neurosense-alice"}
    bob_headers = {"X-Demo-User": "neurosense-bob"}

    created = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 74,
            "hrv_ms": 40,
            "gsr_raw": 1700,
            "source": "simulated",
            "signal_quality": "moderate",
        },
        headers=alice_headers,
    )

    assert created.status_code == 201

    alice_latest = client.get(
        "/api/sensors/latest",
        headers=alice_headers,
    )
    bob_latest = client.get(
        "/api/sensors/latest",
        headers=bob_headers,
    )

    assert alice_latest.status_code == 200
    assert alice_latest.json()["data"] is not None
    assert bob_latest.status_code == 200
    assert bob_latest.json()["data"] is None
