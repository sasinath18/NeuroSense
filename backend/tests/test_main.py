
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


def test_device_list_returns_devices_for_authenticated_user(client):
    """Verify a newly paired device appears in that user's device list."""
    headers = {"X-Demo-User": "api-test-device-list"}
    device_id = "test-device-list-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Device List Test",
        },
        headers=headers,
    )

    assert pair_response.status_code == 201

    list_response = client.get(
        "/api/devices",
        headers=headers,
    )

    assert list_response.status_code == 200
    devices = list_response.json()["devices"]
    matching = [
        item for item in devices
        if item["device_id"] == device_id
    ]

    assert len(matching) == 1
    assert matching[0]["device_name"] == "Device List Test"
    assert matching[0]["active"] == 1


def test_paired_device_can_upload_reading_with_valid_device_token(client):
    """A paired device must upload wearable data using X-Device-Token."""
    user_headers = {"X-Demo-User": "api-test-device-upload"}
    device_id = "test-device-upload-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Device Upload Test",
        },
        headers=user_headers,
    )

    assert pair_response.status_code == 201
    device_token = pair_response.json()["device_token"]

    upload_response = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 78,
            "hrv_ms": 44,
            "gsr_raw": 1770,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": device_token},
    )

    assert upload_response.status_code == 201
    body = upload_response.json()
    assert body["success"] is True
    assert body["reading"]["device_id"] == device_id
    assert body["reading"]["source"] == "wearable"
    assert body["reading"]["heart_rate_bpm"] == 78
    assert body["reading"]["hrv_ms"] == 44
    assert body["reading"]["gsr_raw"] == 1770


def test_device_upload_rejects_missing_device_token(client):
    """The wearable readings endpoint must reject requests without a token."""
    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": "test-device-missing-token",
            "device_name": "Missing Token Test",
        },
        headers={"X-Demo-User": "api-test-missing-token"},
    )

    assert pair_response.status_code == 201

    response = client.post(
        "/api/devices/test-device-missing-token/readings",
        json={
            "heart_rate_bpm": 76,
            "signal_quality": "good",
            "source": "wearable",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "X-Device-Token header is required."


def test_device_upload_rejects_invalid_device_token(client):
    """A token with the wrong secret must not authorize device uploads."""
    device_id = "test-device-invalid-token"
    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Invalid Token Test",
        },
        headers={"X-Demo-User": "api-test-invalid-token"},
    )

    assert pair_response.status_code == 201

    response = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 76,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": "x" * 48},
    )

    assert response.status_code == 401


def test_device_status_reports_recently_seen_device_as_connected(client):
    """A device that uploaded a reading within 60 seconds is connected."""
    user_headers = {"X-Demo-User": "api-test-device-status"}
    device_id = "test-device-status-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Device Status Test",
        },
        headers=user_headers,
    )
    assert pair_response.status_code == 201

    token = pair_response.json()["device_token"]
    upload_response = client.post(
        f"/api/devices/{device_id}/readings",
        json={
            "heart_rate_bpm": 74,
            "signal_quality": "good",
            "source": "wearable",
        },
        headers={"X-Device-Token": token},
    )
    assert upload_response.status_code == 201

    status_response = client.get(
        "/api/device/status",
        headers=user_headers,
    )

    assert status_response.status_code == 200
    body = status_response.json()
    assert body["connected"] is True
    assert any(
        item["device_id"] == device_id and item["connected"] is True
        for item in body["devices"]
    )


def test_unpair_device_marks_device_inactive(client):
    """DELETE device endpoint should deactivate an owned paired device."""
    user_headers = {"X-Demo-User": "api-test-unpair"}
    device_id = "test-device-unpair-01"

    pair_response = client.post(
        "/api/devices/pair",
        json={
            "device_id": device_id,
            "device_name": "Unpair Test",
        },
        headers=user_headers,
    )
    assert pair_response.status_code == 201

    delete_response = client.delete(
        f"/api/devices/{device_id}",
        headers=user_headers,
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["success"] is True
    assert delete_response.json()["active"] is False

    list_response = client.get("/api/devices", headers=user_headers)
    assert list_response.status_code == 200
    matching = [
        item for item in list_response.json()["devices"]
        if item["device_id"] == device_id
    ]
    assert len(matching) == 1
    assert matching[0]["active"] == 0

