
"""
NeuroSense - Backend API Tests
Run from the project root:
    python -m pytest backend/tests/test_main.py -v
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


def test_application_starts(client):
    """Check that the FastAPI application responds."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_health_endpoint(client):
    """Check backend health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


def test_system_status_endpoint(client):
    """Check system status endpoint."""
    response = client.get("/api/system/status")

    assert response.status_code == 200
    assert isinstance(response.json(), dict)


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
    """An empty sensor payload should fail validation."""
    response = client.post(
        "/api/sensors",
        json={},
        headers={"X-Demo-User": "neurosense-test-user"},
    )

    assert response.status_code == 422


def test_valid_sensor_reading_is_accepted_and_stored(client):
    """POST a valid simulated sensor reading and verify the stored response."""
    response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 76,
            "hrv_ms": 42,
            "gsr_raw": 1800,
            "skin_temperature_c": 33.2,
            "signal_quality": "good",
            "source": "simulated",
        },
        headers={"X-Demo-User": "api-test-sensor-post"},
    )

    assert response.status_code == 201

    body = response.json()
    assert body["success"] is True
    assert body["reading"]["user_id"] == "api-test-sensor-post"
    assert body["reading"]["device_id"] == "manual-preview"
    assert body["reading"]["heart_rate_bpm"] == 76
    assert body["reading"]["hrv_ms"] == 42
    assert body["reading"]["gsr_raw"] == 1800
    assert body["reading"]["skin_temperature_c"] == 33.2
    assert body["reading"]["signal_quality"] == "good"
    assert body["reading"]["source"] == "simulated"
    assert body["stress"]["status"] == "simulated_data"


def test_latest_sensor_reading_returns_saved_record(client):
    """GET the latest reading for the same development user after POST."""
    headers = {"X-Demo-User": "api-test-latest"}

    create_response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 81,
            "hrv_ms": 38,
            "gsr_raw": 1920,
            "signal_quality": "moderate",
            "source": "simulated",
        },
        headers=headers,
    )

    assert create_response.status_code == 201
    created_id = create_response.json()["reading"]["id"]

    latest_response = client.get(
        "/api/sensors/latest",
        headers=headers,
    )

    assert latest_response.status_code == 200
    body = latest_response.json()
    assert body["data"] is not None
    assert body["data"]["id"] == created_id
    assert body["data"]["heart_rate_bpm"] == 81
    assert body["data"]["hrv_ms"] == 38
    assert body["data"]["gsr_raw"] == 1920
    assert body["data"]["source"] == "simulated"


def test_sensor_history_returns_saved_readings(client):
    """GET sensor history and verify the record created for this user."""
    headers = {"X-Demo-User": "api-test-history"}

    create_response = client.post(
        "/api/sensors",
        json={
            "heart_rate_bpm": 73,
            "hrv_ms": 45,
            "gsr_raw": 1650,
            "signal_quality": "good",
            "source": "simulated",
        },
        headers=headers,
    )

    assert create_response.status_code == 201
    created_id = create_response.json()["reading"]["id"]

    history_response = client.get(
        "/api/sensors/history?limit=10",
        headers=headers,
    )

    assert history_response.status_code == 200
    body = history_response.json()
    assert body["total"] >= 1
    assert any(
        item["id"] == created_id
        and item["heart_rate_bpm"] == 73
        and item["hrv_ms"] == 45
        and item["gsr_raw"] == 1650
        for item in body["readings"]
    )


def test_device_pairing_creates_device_and_returns_one_time_token(client):
    """Pair a device and verify a usable token is returned only in the response."""
    device_id = "test-device-pairing-01"
    response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "NeuroSense Test Wearable",
        },
        headers={"X-Demo-User": "api-test-device-pairing"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["success"] is True
    assert body["device_id"] == device_id
    assert isinstance(body["device_token"], str)
    assert len(body["device_token"]) >= 24
    assert "shown only once" in body["warning"]

