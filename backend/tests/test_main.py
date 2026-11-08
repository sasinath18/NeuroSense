"""
NeuroSense - Backend API Tests

Run from the project root:
    python -m pytest backend/tests/test_main.py -v

Tests use an isolated temporary SQLite database. They do not modify
the developer's local NeuroSense database or require real hardware.
"""

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
